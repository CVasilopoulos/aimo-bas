import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STARTED = time.monotonic()
SEEN = []


def _track():
    for path in ROOT.iterdir():
        if path.is_file() and path.name.casefold() == "small.txt":
            return "small"
    return "main"


def _profile_name():
    marker = ROOT / "profile.txt"
    if marker.is_file():
        name = marker.read_text(encoding="utf-8").strip()
        if name:
            return name
    return None


def _log(settings, payload):
    if settings.get("log"):
        print("[aimo-bas] " + json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def allot_seconds(settings, model_id, track, now):
    from aimo_bas.profiles import expected_models

    weights = settings["model_weights"]
    remaining = [m for m in expected_models(track) if m not in SEEN and m != model_id]
    own = float(weights.get(model_id, 1.0))
    rest = sum(float(weights.get(m, 1.0)) for m in remaining)
    left = settings["total_budget_s"] - settings["safety_s"] - (now - STARTED)
    return max(0.0, left * own / (own + rest))


def length_prior(problems):
    lengths = [len(p) for p in problems]
    ordered = sorted(lengths)
    median = ordered[(len(ordered) - 1) // 2]
    return [length <= median for length in lengths]


def plan_variants(settings, count):
    for limit, kinds in settings["variant_plans"]:
        if count <= limit:
            return list(kinds)
    return ["orig"]


def problem_features(records, kinds, weights):
    answers = [r["answer"] for r in records]
    valid = [a for a in answers if a]
    modal, modal_count = (Counter(valid).most_common(1)[0] if valid else ("", 0))
    agree = modal_count / len(records) if records else 0.0
    finish = sum(1 for r in records if r["closed"]) / len(records) if records else 0.0
    confidence = sum(r["confidence"] for r in records) / len(records) if records else 0.0
    originals = [r["answer"] for r, k in zip(records, kinds) if k == "orig"]
    perturbed = [r["answer"] for r, k in zip(records, kinds) if k != "orig"]
    stable = (
        sum(1 for a in perturbed if a and a == modal) / len(perturbed) if perturbed else agree
    )
    score = weights[0] * agree + weights[1] * finish + weights[2] * confidence
    return {
        "agree": agree,
        "finish": finish,
        "confidence": confidence,
        "stable": stable,
        "orig_agree": (Counter(a for a in originals if a).most_common(1)[0][1] / len(originals))
        if any(originals)
        else 0.0,
        "score": score,
    }


def decide(features, problems, settings):
    count = len(features)
    scores = [f["score"] for f in features]
    order = sorted(range(count), key=lambda i: (-scores[i], len(problems[i]), i))
    keep = count // 2
    if count % 2 == 1 and scores[order[keep]] >= 0.5:
        keep += 1
    chosen = set(order[:keep])
    predictions = []
    for index in range(count):
        value = index in chosen
        if scores[index] >= settings["hi"]:
            value = True
        if scores[index] <= settings["lo"]:
            value = False
        predictions.append(value)
    return predictions


def _is_oom(error):
    import torch

    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _run_chunks(total, chunk, deadline, work):
    from aimo_bas.engine import cleanup

    start = 0
    size = chunk
    while start < total:
        now = time.monotonic()
        if now >= deadline:
            break
        end = min(total, start + size)
        share = (end - start) / (total - start)
        chunk_deadline = now + (deadline - now) * share
        oom = False
        try:
            work(start, end, chunk_deadline)
        except Exception as error:
            if not _is_oom(error):
                raise
            oom = True
        if oom:
            cleanup()
            if size == 1:
                raise RuntimeError("out of memory with a single sequence")
            size = max(1, size // 2)
            continue
        start = end
    return start


def predict(model_id, problems):
    import torch

    from aimo_bas import engine, perturb
    from aimo_bas.answers import explicit_answer, normalize
    from aimo_bas.profiles import active

    import os

    cuda = torch.cuda.is_available()
    profile = _profile_name()
    if profile and not os.environ.get("AIMO_BAS_PROFILE"):
        os.environ["AIMO_BAS_PROFILE"] = profile
    settings = active(cuda)
    track = _track()
    began = time.monotonic()
    allotted = allot_seconds(settings, model_id, track, began)
    SEEN.append(model_id)
    summary = {"model": model_id, "n": len(problems), "track": track, "profile": settings["name"], "allotted_s": round(allotted, 1)}
    if allotted < settings["min_model_s"]:
        summary["mode"] = "prior-no-time"
        _log(settings, summary)
        return length_prior(problems)
    model_deadline = began + allotted
    tokenizer = engine.load_tokenizer(model_id)
    model = engine.load_model(model_id, cuda)
    try:
        harmony = engine.is_harmony(tokenizer)
        eos = engine.stop_ids(model, tokenizer, harmony)
        kinds = plan_variants(settings, len(problems))
        prompts = []
        owners = []
        kind_of = []
        for index, problem in enumerate(problems):
            for kind in kinds:
                prompts.append(engine.render(tokenizer, perturb.variant(problem, kind), harmony, settings["reasoning_effort"]))
                owners.append(index)
                kind_of.append(kind)
        temperature, top_p = settings["sampling_overrides"].get(model_id, [settings["temperature"], settings["top_p"]])
        longest = max(len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts)
        per_sequence = engine.kv_bytes_per_sequence(model, longest + settings["think_tokens"] + settings["answer_tokens"] + 16)
        chunk = max(1, min(settings["max_chunk"], int(settings["max_kv_gb"] * 1024**3 // max(1, per_sequence))))
        reserve = max(settings["force_reserve_min_s"], allotted * settings["force_reserve_frac"])
        generation_deadline = model_deadline - reserve
        generations = [None] * len(prompts)

        def generate_work(start, end, deadline):
            outputs = engine.sample(
                model,
                tokenizer,
                prompts[start:end],
                eos,
                settings["think_tokens"],
                temperature,
                top_p,
                settings["seed"] + start,
                deadline,
            )
            generations[start:end] = outputs

        t_gen = time.monotonic()
        done = _run_chunks(len(prompts), chunk, generation_deadline, generate_work)
        gen_seconds = time.monotonic() - t_gen
        records = []
        forced_texts = []
        for index, prompt in enumerate(prompts):
            generated = generations[index] or {"text": "", "tokens": 0, "eos": False}
            reasoning, tail, closed = engine.split_reasoning(generated["text"], harmony)
            natural = explicit_answer(tail) if closed else ""
            records.append({"answer": natural, "closed": closed, "confidence": 0.0, "tokens": generated["tokens"], "ran": generations[index] is not None})
            forced_texts.append(engine.forced_prompt(prompt, reasoning, harmony))
        forced = [None] * len(prompts)

        def force_work(start, end, deadline):
            outputs = engine.force_answers(model, tokenizer, forced_texts[start:end], eos, settings["answer_tokens"])
            forced[start:end] = outputs

        t_force = time.monotonic()
        force_chunk = max(1, chunk)
        _run_chunks(len(prompts), force_chunk, model_deadline, force_work)
        force_seconds = time.monotonic() - t_force
        for index, record in enumerate(records):
            if forced[index] is not None:
                forced_answer = normalize(forced[index]["text"])
                record["confidence"] = forced[index]["confidence"]
                if not record["answer"]:
                    record["answer"] = forced_answer
        features = []
        for index in range(len(problems)):
            mine = [r for r, owner in zip(records, owners) if owner == index]
            mine_kinds = [k for k, owner in zip(kind_of, owners) if owner == index]
            features.append(problem_features(mine, mine_kinds, settings["score_weights"]))
        predictions = decide(features, problems, settings)
        summary.update(
            {
                "mode": "probe",
                "variants": len(kinds),
                "chunk": chunk,
                "generated_sequences": done,
                "total_sequences": len(prompts),
                "gen_s": round(gen_seconds, 1),
                "force_s": round(force_seconds, 1),
                "tokens": sum(r["tokens"] for r in records),
                "mean_finish": round(sum(f["finish"] for f in features) / len(features), 3),
                "mean_agree": round(sum(f["agree"] for f in features) / len(features), 3),
                "positive_rate": round(sum(predictions) / len(predictions), 3),
                "elapsed_s": round(time.monotonic() - began, 1),
            }
        )
        _log(settings, summary)
        return predictions
    finally:
        del model
        engine.cleanup()


def are_robust(model_id, problems):
    problems = list(problems)
    if not problems:
        return []
    try:
        predictions = predict(model_id, problems)
        if isinstance(predictions, list) and len(predictions) == len(problems):
            return [bool(value) for value in predictions]
    except Exception:
        traceback.print_exc(file=sys.stderr)
    return [bool(value) for value in length_prior(problems)]
