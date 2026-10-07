# /// script
# dependencies = ["pandas>=2.2", "pyarrow>=17", "matplotlib>=3.9", "numpy>=2"]
# ///
import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from common import ROOT, SHORT, SUBMISSIONS, save_table, save_text, solution_path

solution_path()
from aimo_bas import profiles


def engine_without_torch():
    torch = types.ModuleType("torch")
    torch.inference_mode = lambda: (lambda f: f)
    transformers = types.ModuleType("transformers")
    for name in ("AutoModelForCausalLM", "AutoTokenizer", "PreTrainedTokenizerFast", "StoppingCriteriaList"):
        setattr(transformers, name, type(name, (), {}))
    transformers.StoppingCriteria = type("StoppingCriteria", (), {})
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("transformers", transformers)
    from aimo_bas import engine

    return engine


def hyperparameters():
    rows = []
    for key, value in profiles.BASE.items():
        rows.append({"profile": "gpu (BASE)", "setting": key, "value": json.dumps(value)})
    for name, overrides in profiles.PROFILES.items():
        for key, value in overrides.items():
            rows.append({"profile": name, "setting": key, "value": json.dumps(value)})
    return pd.DataFrame(rows)


def models(engine):
    rows = []
    for model_id in profiles.MAIN_MODELS:
        path = ROOT / "research" / "modelcfg" / (model_id.replace("/", "_") + ".config.json")
        raw = json.loads(path.read_text())
        config = raw.get("text_config") or raw
        ns = SimpleNamespace(**config)
        stub = SimpleNamespace(config=ns)
        layers = int(config.get("num_hidden_layers", 0))
        types_ = config.get("layer_types") or ["full_attention"] * layers
        length = 6500
        rows.append(
            {
                "model_id": model_id,
                "short": SHORT[model_id],
                "architecture": ",".join(raw.get("architectures", [])),
                "hidden_layers": layers,
                "attention_heads": config.get("num_attention_heads"),
                "kv_heads": config.get("num_key_value_heads"),
                "head_dim": config.get("head_dim") or (config.get("hidden_size", 0) // max(1, config.get("num_attention_heads", 1))),
                "full_attention_layers": sum(1 for t in types_ if t == "full_attention"),
                "sliding_attention_layers": sum(1 for t in types_ if t == "sliding_attention"),
                "linear_attention_layers": sum(1 for t in types_ if t == "linear_attention"),
                "sliding_window": config.get("sliding_window"),
                "max_position_embeddings": config.get("max_position_embeddings"),
                "kv_gb_per_sequence_at_6500": round(engine.kv_bytes_per_sequence(stub, length) / 1024**3, 3),
                "sequences_per_48gb_chunk": min(profiles.BASE["max_chunk"], int(profiles.BASE["max_kv_gb"] * 1024**3 // engine.kv_bytes_per_sequence(stub, length))),
                "small_track": model_id in profiles.SMALL_MODELS,
            }
        )
    return pd.DataFrame(rows)


def submissions():
    rows = []
    for track, meta in SUBMISSIONS.items():
        bundle = ROOT / "submissions" / meta["bundle"]
        rows.append(
            {
                "codabench_id": meta["id"],
                "track": track,
                "bundle": meta["bundle"],
                "bytes": bundle.stat().st_size,
                "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "profile": "gpu",
                "uploaded_utc": "2026-09-16",
                "correct": meta["correct"],
                "n": meta["n"],
                "accuracy": round(meta["correct"] / meta["n"], 3),
                "coverage": 1.0,
                "invalid_predictions": 0,
            }
        )
    return pd.DataFrame(rows)


def prompts(engine):
    solve = (ROOT / "research" / "baselines" / "prompts" / "solve.txt").read_text()
    same = solve.strip() == engine.SYSTEM_PROMPT.strip()
    lines = [
        "**System prompt** (`solution/aimo_bas/engine.py:SYSTEM_PROMPT`; identical to the organisers' `baselines/prompts/solve.txt`: " + str(same) + ")",
        "",
        "```",
        engine.SYSTEM_PROMPT,
        "```",
        "",
        "**Forced-answer suffix, `</think>` models** (`THINK_FORCE`)",
        "",
        "```",
        engine.THINK_FORCE,
        "```",
        "",
        "**Forced-answer suffix, harmony (gpt-oss)** (`HARMONY_FORCE`)",
        "",
        "```",
        engine.HARMONY_FORCE,
        "```",
        "",
        "**Reasoning split markers**: `" + engine.THINK_END + "` and `" + engine.HARMONY_END + "`.",
        "",
    ]
    return "\n".join(lines)


def main():
    engine = engine_without_torch()
    save_table(hyperparameters(), "hyperparameters")
    save_table(models(engine), "models")
    save_table(submissions(), "submissions")
    save_text(prompts(engine), "prompts.md")


if __name__ == "__main__":
    main()
