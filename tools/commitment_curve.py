import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "solution")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=10240)
    parser.add_argument("--cuts", default="512,1024,2048,3072,4096,6144,8192,10240")
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--remap", action="store_true", help="run every case on --model, ignoring the case file's model_id")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import torch

    from aimo_bas import engine
    from aimo_bas.answers import explicit_answer, normalize

    cuda = torch.cuda.is_available()
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    cases = cases if args.remap else [c for c in cases if c["model_id"] == args.model]
    if args.limit:
        cases = cases[: args.limit]
    tokenizer = engine.load_tokenizer(args.model)
    model = engine.load_model(args.model, cuda)
    harmony = engine.is_harmony(tokenizer)
    eos = engine.stop_ids(model, tokenizer, harmony)
    cuts = [int(x) for x in args.cuts.split(",")]
    prompts = []
    owners = []
    for case in cases:
        prompt = engine.render(tokenizer, case["problem"], harmony, "low")
        for _ in range(args.samples):
            prompts.append(prompt)
            owners.append(case)
    generations = []
    started = time.monotonic()
    for start in range(0, len(prompts), args.chunk):
        generations.extend(engine.sample(model, tokenizer, prompts[start : start + args.chunk], eos, args.max_tokens, args.temperature, args.top_p, args.seed + start, time.monotonic() + 10**7))
    gen_seconds = time.monotonic() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as stream:
        for index, (prompt, generated) in enumerate(zip(prompts, generations)):
            reasoning, tail, closed = engine.split_reasoning(generated["text"], harmony)
            natural = explicit_answer(tail) if closed else ""
            ids = tokenizer(reasoning, add_special_tokens=False)["input_ids"]
            texts = []
            used = []
            for cut in cuts:
                if cut > len(ids) and used and used[-1] >= len(ids):
                    break
                prefix = tokenizer.decode(ids[: min(cut, len(ids))], skip_special_tokens=False)
                texts.append(engine.forced_prompt(prompt, prefix, harmony))
                used.append(min(cut, len(ids)))
            forced = []
            for start in range(0, len(texts), args.chunk):
                forced.extend(engine.force_answers(model, tokenizer, texts[start : start + args.chunk], eos, 12))
            record = {
                "model_id": args.model,
                "case_id": owners[index]["id"],
                "problem_sha256": hashlib.sha256(owners[index]["problem"].encode()).hexdigest()[:16],
                "sample": index % args.samples,
                "reasoning_tokens": len(ids),
                "closed": closed,
                "natural_answer": natural,
                "gen_seconds_total": round(gen_seconds, 1),
                "cuts": [{"tokens": u, "answer": normalize(f["text"]), "confidence": round(f["confidence"], 4)} for u, f in zip(used, forced)],
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
