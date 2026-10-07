from aimo_bas.runner import are_robust as _are_robust


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    return _are_robust(model_id, problems)
