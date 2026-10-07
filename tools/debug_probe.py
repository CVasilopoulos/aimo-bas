import json
import sys
import time

sys.path.insert(0, "/app/ingested_program")

from aimo_bas import engine, perturb
from aimo_bas.answers import explicit_answer, normalize

model_id = sys.argv[1]
think_tokens = int(sys.argv[2])
problems = [
    "What is the sum of all positive divisors of 12?",
    "Find the remainder when $2^{10}$ is divided by $7$.",
]
tokenizer = engine.load_tokenizer(model_id)
model = engine.load_model(model_id, False)
harmony = engine.is_harmony(tokenizer)
eos = engine.stop_ids(model, tokenizer, harmony)
print("tokenizer", type(tokenizer).__name__, "harmony", harmony, "eos", eos, "model", type(model).__name__)
kinds = ["orig", "orig", "typos", "distract", "rephrase"]
prompts = []
for problem in problems:
    for kind in kinds:
        prompts.append(engine.render(tokenizer, perturb.variant(problem, kind), harmony, "low"))
print("prompt tail", repr(prompts[0][-120:]))
start = time.time()
generations = engine.sample(model, tokenizer, prompts, eos, think_tokens, 0.6, 0.95, 7, time.monotonic() + 3000)
print("gen seconds", round(time.time() - start, 1))
forced_texts = []
for prompt, generated in zip(prompts, generations):
    reasoning, tail, closed = engine.split_reasoning(generated["text"], harmony)
    forced_texts.append(engine.forced_prompt(prompt, reasoning, harmony))
    generated["closed"] = closed
    generated["natural"] = explicit_answer(tail) if closed else ""
forced = engine.force_answers(model, tokenizer, forced_texts, eos, 12)
for index, (generated, forced_row) in enumerate(zip(generations, forced)):
    print(json.dumps({
        "i": index,
        "kind": kinds[index % len(kinds)],
        "tokens": generated["tokens"],
        "eos": generated["eos"],
        "closed": generated["closed"],
        "natural": generated["natural"],
        "forced_raw": forced_row["text"],
        "forced": normalize(forced_row["text"]),
        "conf": round(forced_row["confidence"], 3),
        "tail": generated["text"][-160:],
    }))
