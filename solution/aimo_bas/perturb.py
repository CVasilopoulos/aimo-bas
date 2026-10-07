import hashlib
import random
import re

MATH_SPAN = re.compile(
    r"(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)|\\begin\{[a-zA-Z*]+\}.*?\\end\{[a-zA-Z*]+\})",
    re.S,
)
WORD = re.compile(r"(?<![\\A-Za-z])[A-Za-z]{6,}(?![A-Za-z])")
NAMES = [
    "Alice", "Bob", "Carol", "Charlie", "Dave", "David", "Eve", "Frank", "Grace",
    "Heidi", "Ivan", "Judy", "Mallory", "Oscar", "Peggy", "Trent", "Victor", "Walter",
    "Anna", "Ben", "Emma", "John", "Mary", "Tom", "Sarah", "Peter", "Paul", "Lucy",
]
SUBSTITUTE_NAMES = [
    "Irene", "Mateo", "Noor", "Kenji", "Leila", "Tomas", "Ingrid", "Ravi", "Olga",
    "Diego", "Hana", "Pavel", "Zara", "Emil", "Yara", "Soren", "Nadia", "Farid",
]
LEADS = [
    (re.compile(r"\bFind\b"), "Determine"),
    (re.compile(r"\bCompute\b"), "Calculate"),
    (re.compile(r"\bDetermine\b"), "Find"),
    (re.compile(r"\bWhat is\b"), "Determine"),
    (re.compile(r"\bHow many\b"), "Determine how many"),
]
DISTRACTOR = (
    "\n\n(Unrelated note: this problem was typed up by a volunteer on a quiet afternoon "
    "and has been checked by two readers.)"
)
REFRAME_PREFIX = "Here is a competition problem.\n\n"
REFRAME_SUFFIX = "\n\nSolve it carefully."


def _seed(text, tag):
    digest = hashlib.sha256((tag + "\x00" + text).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _map_prose(text, fn):
    parts = MATH_SPAN.split(text)
    out = []
    for index, part in enumerate(parts):
        out.append(part if index % 2 == 1 else fn(part))
    return "".join(out)


def typos(text, count=2):
    rng = random.Random(_seed(text, "typos"))
    prose_words = []
    for index, part in enumerate(MATH_SPAN.split(text)):
        if index % 2 == 0:
            prose_words.extend(match.group(0) for match in WORD.finditer(part))
    if not prose_words:
        return text
    chosen = set(rng.sample(sorted(set(prose_words)), min(count, len(set(prose_words)))))

    def swap(match):
        word = match.group(0)
        if word not in chosen:
            return word
        position = 1 + _seed(word, "pos") % (len(word) - 3)
        letters = list(word)
        letters[position], letters[position + 1] = letters[position + 1], letters[position]
        return "".join(letters)

    return _map_prose(text, lambda prose: WORD.sub(swap, prose))


def rename(text):
    present = [name for name in NAMES if re.search(r"\b" + name + r"\b", text)]
    if not present:
        return None
    rng = random.Random(_seed(text, "rename"))
    pool = [name for name in SUBSTITUTE_NAMES if name not in text]
    rng.shuffle(pool)
    mapping = dict(zip(present, pool))
    if not mapping:
        return None
    pattern = re.compile(r"\b(" + "|".join(re.escape(name) for name in mapping) + r")\b")
    return _map_prose(text, lambda prose: pattern.sub(lambda m: mapping[m.group(1)], prose))


def rephrase(text):
    def lead(prose):
        for pattern, replacement in LEADS:
            new = pattern.sub(replacement, prose, count=1)
            if new != prose:
                return new
        return prose

    changed = _map_prose(text, lead)
    return REFRAME_PREFIX + changed + REFRAME_SUFFIX


def distract(text):
    return text + DISTRACTOR


def variant(text, kind):
    if kind == "orig":
        return text
    if kind == "typos":
        return typos(text)
    if kind == "distract":
        return distract(text)
    if kind == "rephrase":
        return rephrase(text)
    if kind == "rename":
        renamed = rename(text)
        return renamed if renamed is not None else rephrase(typos(text, count=1))
    raise ValueError(kind)
