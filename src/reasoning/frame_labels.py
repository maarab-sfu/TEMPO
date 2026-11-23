"""A small output vocabulary for visual attribution."""
from itertools import combinations
import re


FRAME_LABELS = "ABCDEFGH"


def parse_frame_labels(raw, frame_count):
    """Accept only displayed frame labels or an explicit NONE verdict."""
    answer = raw.strip().upper().rstrip(".")
    if answer == "NONE":
        return []
    if answer.startswith("[") and answer.endswith("]"):
        answer = answer[1:-1].replace('"', "").replace("'", "").strip()
    if not re.fullmatch(r"[A-H](?:[ ,]+[A-H])*", answer):
        return None
    indices = sorted({FRAME_LABELS.index(label) for label in re.findall(r"[A-H]", answer)})
    return indices if all(index < frame_count for index in indices) else None


def citation_token_prefix(tokenizer, labels, eos_ids, prompt_length):
    """Constrain generation to NONE or an ordered subset of the visible labels."""
    options = ["NONE"]
    for count in range(1, len(labels) + 1):
        options.extend(" ".join(selected) for selected in combinations(labels, count))
    encoded = tokenizer(options, add_special_tokens=False)["input_ids"]
    terminal = [eos_ids] if isinstance(eos_ids, int) else eos_ids
    allowed = {}
    for tokens in encoded:
        for index, token in enumerate(tokens):
            allowed.setdefault(tuple(tokens[:index]), set()).add(token)
        allowed.setdefault(tuple(tokens), set()).update(terminal)
    allowed = {prefix: sorted(tokens) for prefix, tokens in allowed.items()}

    def next_tokens(_batch_id, input_ids):
        return allowed[tuple(input_ids[prompt_length:].tolist())]

    return next_tokens
