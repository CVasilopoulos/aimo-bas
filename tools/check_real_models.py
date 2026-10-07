import json
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, "/app/ingested_program")

from transformers import AutoConfig, GenerationConfig

from aimo_bas import engine, perturb
from aimo_bas.profiles import MAIN_MODELS

problem = "Alice picks $n \\le 1000$. Find the number of integers $n$ such that $7 \\mid n$."
for model_id in MAIN_MODELS:
    row = {"model": model_id}
    try:
        tokenizer = engine.load_tokenizer(model_id)
        config = AutoConfig.from_pretrained(model_id, local_files_only=True)
        try:
            generation = GenerationConfig.from_pretrained(model_id, local_files_only=True)
        except Exception:
            generation = GenerationConfig()
        stub = SimpleNamespace(config=config, generation_config=generation)
        harmony = engine.is_harmony(tokenizer)
        prompt = engine.render(tokenizer, perturb.variant(problem, "typos"), harmony, "low")
        forced = engine.forced_prompt(prompt, "some reasoning", harmony)
        row.update(
            tokenizer=type(tokenizer).__name__,
            roundtrip=engine._roundtrip_ok(tokenizer),
            harmony=harmony,
            eos=engine.stop_ids(stub, tokenizer, harmony),
            pad=tokenizer.pad_token_id,
            system_in_prompt="precise math problem solver" in prompt,
            prompt_head=prompt[:60],
            prompt_tail=prompt[-70:],
            forced_tail=forced[-80:],
            prompt_tokens=len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
            kv_gb_per_seq_6k=round(engine.kv_bytes_per_sequence(stub, 6500) / 1024**3, 3),
        )
    except Exception as error:
        row["error"] = repr(error)[:300]
    print(json.dumps(row, ensure_ascii=False))

tokenizer = engine.load_tokenizer("Qwen/Qwen3-0.6B")
model = engine.load_model("Qwen/Qwen3-0.6B", False)
eos = engine.stop_ids(model, tokenizer, False)
prompts = [engine.render(tokenizer, "Prove that there are infinitely many primes, then count to 5000.", False, "low")] * 2
started = time.monotonic()
out = engine.sample(model, tokenizer, prompts, eos, 4000, 0.6, 0.95, 1, time.monotonic() + 15)
print(json.dumps({"deadline_test_s": round(time.monotonic() - started, 1), "tokens": [o["tokens"] for o in out], "eos": [o["eos"] for o in out]}))
