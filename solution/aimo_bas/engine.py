import gc
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
)

SYSTEM_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, then output "
    "your final answer on the last line in the exact format:\nANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)
PROBE = "Hello world 12345 \\frac{1}{2} x^2 + y_1 = 0.\n"
THINK_END = "</think>"
THINK_FORCE = "\n</think>\n\nANSWER: "
HARMONY_END = "<|end|>"
HARMONY_FORCE = "<|end|><|start|>assistant<|channel|>final<|message|>ANSWER: "


class WallClock(StoppingCriteria):
    def __init__(self, deadline):
        self.deadline = deadline

    def __call__(self, input_ids, scores, **kwargs):
        flag = time.monotonic() >= self.deadline
        return torch.full((input_ids.shape[0],), flag, dtype=torch.bool, device=input_ids.device)


def _roundtrip_ok(tokenizer):
    try:
        ids = tokenizer(PROBE, add_special_tokens=False)["input_ids"]
        return tokenizer.decode(ids, skip_special_tokens=False) == PROBE
    except Exception:
        return False


def load_tokenizer(model_id):
    candidates = []
    for loader in (PreTrainedTokenizerFast, AutoTokenizer):
        try:
            candidates.append(loader.from_pretrained(model_id, local_files_only=True))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("no tokenizer")
    ranked = sorted(
        candidates,
        key=lambda tok: (not _roundtrip_ok(tok), getattr(tok, "chat_template", None) is None),
    )
    tokenizer = ranked[0]
    if getattr(tokenizer, "chat_template", None) is None:
        for other in ranked[1:]:
            if getattr(other, "chat_template", None) is not None:
                tokenizer.chat_template = other.chat_template
                break
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_id, cuda):
    kwargs = {"local_files_only": True}
    if cuda:
        kwargs.update(dtype=torch.bfloat16, device_map="auto")
    else:
        kwargs.update(dtype=torch.float32)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    return model


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_harmony(tokenizer):
    template = getattr(tokenizer, "chat_template", None) or ""
    return "<|channel|>" in template


def render(tokenizer, problem, harmony, effort):
    extra = {"reasoning_effort": effort} if harmony else {"enable_thinking": True}
    attempts = [
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem}],
        [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + problem}],
    ]
    for messages in attempts:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **extra
            )
        except Exception:
            continue
        if isinstance(text, str) and text:
            return text
    return SYSTEM_PROMPT + "\n\n" + problem + "\n\n"


def stop_ids(model, tokenizer, harmony):
    ids = set()
    for value in (getattr(model.generation_config, "eos_token_id", None), tokenizer.eos_token_id):
        if value is None:
            continue
        if isinstance(value, int):
            ids.add(value)
        else:
            ids.update(int(item) for item in value)
    if harmony:
        vocab = tokenizer.get_vocab()
        for token in ("<|return|>", "<|call|>"):
            if token in vocab:
                ids.add(vocab[token])
    return sorted(ids)


def kv_bytes_per_sequence(model, length):
    config = getattr(model.config, "text_config", None) or model.config
    layers = int(getattr(config, "num_hidden_layers", 32))
    heads = int(getattr(config, "num_attention_heads", 32))
    kv_heads = int(getattr(config, "num_key_value_heads", None) or heads)
    head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // heads)
    types = list(getattr(config, "layer_types", None) or ["full_attention"] * layers)
    window = int(getattr(config, "sliding_window", None) or length)
    per_token = 2 * kv_heads * head_dim * 2
    total = 0
    for kind in types:
        if kind == "sliding_attention":
            total += per_token * min(window, length)
        elif kind == "linear_attention":
            total += 8 * 1024 * 1024
        else:
            total += per_token * length
    return total


def _device(model):
    embeddings = model.get_input_embeddings()
    if embeddings is not None:
        return embeddings.weight.device
    return next(model.parameters()).device


def _encode(tokenizer, texts, device):
    encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


@torch.inference_mode()
def sample(model, tokenizer, prompts, eos, max_new_tokens, temperature, top_p, seed, deadline):
    encoded = _encode(tokenizer, prompts, _device(model))
    width = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos,
        stopping_criteria=StoppingCriteriaList([WallClock(deadline)]),
    )
    rows = output[:, width:].tolist()
    del output, encoded
    eos_set = set(eos)
    results = []
    for row in rows:
        cut = next((index for index, token in enumerate(row) if token in eos_set), None)
        ended = cut is not None
        kept = row if cut is None else row[:cut]
        text = tokenizer.decode(kept, skip_special_tokens=False)
        results.append({"text": text, "tokens": len(kept), "eos": ended})
    return results


def split_reasoning(text, harmony):
    marker = HARMONY_END if harmony else THINK_END
    index = text.find(marker)
    if index < 0:
        return text, "", False
    return text[:index], text[index + len(marker):], True


def forced_prompt(prompt, reasoning, harmony):
    return prompt + reasoning + (HARMONY_FORCE if harmony else THINK_FORCE)


@torch.inference_mode()
def force_answers(model, tokenizer, texts, eos, max_new_tokens):
    encoded = _encode(tokenizer, texts, _device(model))
    width = encoded["input_ids"].shape[1]
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos,
        output_scores=True,
        return_dict_in_generate=True,
    )
    sequences = output.sequences[:, width:]
    probabilities = []
    for step, scores in enumerate(output.scores):
        chosen = sequences[:, step : step + 1]
        probabilities.append(torch.softmax(scores.float(), dim=-1).gather(-1, chosen).squeeze(-1))
    stacked = torch.stack(probabilities, dim=1).cpu().tolist() if probabilities else [[] for _ in texts]
    token_rows = sequences.cpu().tolist()
    del output, encoded, sequences
    eos_set = set(eos)
    results = []
    for tokens, probs in zip(token_rows, stacked):
        kept_tokens = []
        kept_probs = []
        for token, prob in zip(tokens, probs):
            if token in eos_set:
                break
            piece = tokenizer.decode([token], skip_special_tokens=False)
            if "\n" in piece and kept_tokens:
                break
            kept_tokens.append(token)
            kept_probs.append(prob)
        text = tokenizer.decode(kept_tokens, skip_special_tokens=True)
        confidence = sum(kept_probs) / len(kept_probs) if kept_probs else 0.0
        results.append({"text": text, "confidence": confidence})
    return results
