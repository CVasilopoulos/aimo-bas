import re

INTEGER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
BOXED = re.compile(r"\\boxed\s*\{")
COMPOSITE = re.compile(r"\d\s*/\s*\d|\\frac|\\sqrt|\\pi|\^")


def _strip_markup(text):
    text = text.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    text = BOXED.sub("", text)
    for token in ("$", "\\(", "\\)", "\\[", "\\]", "**", "`"):
        text = text.replace(token, "")
    return text.strip()


def normalize(raw):
    if raw is None:
        return ""
    text = _strip_markup(raw.split("\n", 1)[0])
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("nan") or "no solution" in lowered or lowered.startswith("none"):
        return "NaN"
    if COMPOSITE.search(text):
        return re.sub(r"[\s$]", "", text)[:32].lower()
    match = INTEGER.search(text)
    if match:
        number = match.group(0).replace(",", "")
        try:
            value = float(number)
        except ValueError:
            return ""
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.6g}"
    compact = re.sub(r"[\s{}]", "", text)
    return compact[:24].lower()


def last_boxed(text):
    starts = [m.end() for m in BOXED.finditer(text)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return text[start:]


def explicit_answer(text):
    lines = [line for line in text.splitlines() if "ANSWER:" in line.upper()]
    if lines:
        line = lines[-1]
        position = line.upper().rfind("ANSWER:")
        return normalize(line[position + len("ANSWER:"):])
    boxed = last_boxed(text)
    if boxed is not None:
        return normalize(boxed)
    return ""
