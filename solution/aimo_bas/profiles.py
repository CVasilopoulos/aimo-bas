import os

SMALL_MODELS = [
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
]
MAIN_MODELS = SMALL_MODELS + ["openai/gpt-oss-120b"]

BASE = {
    "total_budget_s": 3600.0,
    "safety_s": 480.0,
    "min_model_s": 45.0,
    "force_reserve_frac": 0.12,
    "force_reserve_min_s": 20.0,
    "think_tokens": 6144,
    "answer_tokens": 12,
    "variant_plans": [
        [8, ["orig", "orig", "orig", "typos", "distract", "rephrase"]],
        [16, ["orig", "orig", "typos", "distract"]],
        [1000000, ["orig", "distract"]],
    ],
    "max_kv_gb": 48.0,
    "max_chunk": 48,
    "temperature": 0.6,
    "top_p": 0.95,
    "sampling_overrides": {"openai/gpt-oss-120b": [1.0, 1.0]},
    "model_weights": {"openai/gpt-oss-120b": 1.5},
    "seed": 20261101,
    "score_weights": [0.5, 0.3, 0.2],
    "hi": 0.9,
    "lo": 0.3,
    "reasoning_effort": "low",
    "log": True,
}

PROFILES = {
    "gpu": {},
    "gpu-long": {"think_tokens": 10240, "safety_s": 420.0},
    "gpu-fast": {
        "think_tokens": 3072,
        "variant_plans": [[8, ["orig", "orig", "typos", "distract"]], [1000000, ["orig", "distract"]]],
    },
    "gpu-faster": {
        "think_tokens": 2048,
        "variant_plans": [[8, ["orig", "orig", "typos", "distract"]], [1000000, ["orig", "distract"]]],
    },
    "gpu-fast-wide": {
        "think_tokens": 3072,
        "variant_plans": [
            [8, ["orig", "orig", "typos", "distract", "rephrase"]],
            [16, ["orig", "orig", "typos", "distract"]],
            [1000000, ["orig", "distract"]],
        ],
    },
    "gpu-fast-tight": {
        "think_tokens": 3072,
        "safety_s": 300.0,
        "variant_plans": [[8, ["orig", "orig", "typos", "distract"]], [1000000, ["orig", "distract"]]],
    },
    "cpu-smoke": {
        "total_budget_s": 3000.0,
        "safety_s": 120.0,
        "min_model_s": 5.0,
        "force_reserve_min_s": 5.0,
        "think_tokens": 24,
        "answer_tokens": 6,
        "variant_plans": [[8, ["orig", "typos", "distract"]], [1000000, ["orig"]]],
        "max_chunk": 12,
    },
}


def active(cuda_available):
    name = os.environ.get("AIMO_BAS_PROFILE") or ("gpu" if cuda_available else "cpu-smoke")
    merged = dict(BASE)
    merged.update(PROFILES.get(name, {}))
    merged["name"] = name if name in PROFILES else "gpu"
    return merged


def expected_models(track):
    return list(SMALL_MODELS if track == "small" else MAIN_MODELS)
