from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Span:
    """
    Entity span in character/token indices (inclusive).
    For your pipeline, tokens are single Chinese characters, so indices map to char positions.
    """
    ent_type: str
    start: int
    end: int
    text: str


def _split_tag(tag: str) -> (str, Optional[str]):
    """
    'B-company' -> ('B', 'company')
    'O' -> ('O', None)
    """
    if tag == "O" or tag is None:
        return "O", None
    if "-" not in tag:
        # unexpected tag, treat as O
        return "O", None
    prefix, ent_type = tag.split("-", 1)
    return prefix, ent_type


def bioes_to_spans(tokens: List[str], tags: List[str]) -> List[Span]:
    """
    Convert BIOES tags to entity spans.
    tokens: list of characters (length N)
    tags: list of BIOES tags (length N), e.g. 'B-name', 'I-name', 'S-company', 'O'

    Returns:
        List[Span] sorted by start index.

    Robustness:
      - If an 'I-X' appears without a preceding 'B-X', we treat it as a 'B-X' (start a new span).
      - If a 'B-X' is followed by 'O' or different type, we close it as a single-token span.
      - 'E-X' is treated as closing tag (BIOES includes E, but CLUENER uses S for single and I for inside)
    """
    assert len(tokens) == len(tags), f"tokens and tags length mismatch: {len(tokens)} vs {len(tags)}"

    spans: List[Span] = []
    cur_type: Optional[str] = None
    cur_start: Optional[int] = None

    def close_span(end_idx: int):
        nonlocal cur_type, cur_start
        if cur_type is None or cur_start is None:
            return
        text = "".join(tokens[cur_start:end_idx + 1])
        spans.append(Span(ent_type=cur_type, start=cur_start, end=end_idx, text=text))
        cur_type, cur_start = None, None

    for i, tag in enumerate(tags):
        prefix, ent_type = _split_tag(tag)

        if prefix == "O":
            # close any open span
            close_span(i - 1)
            continue

        if prefix == "S":
            # single-token entity, close previous and add this
            close_span(i - 1)
            spans.append(Span(ent_type=ent_type, start=i, end=i, text=tokens[i]))
            continue

        if prefix == "B":
            # begin a new span
            close_span(i - 1)
            cur_type = ent_type
            cur_start = i
            continue

        if prefix in ("I", "E"):
            # inside or end
            if cur_type is None:
                # malformed: I-X without B-X; start a new span
                cur_type = ent_type
                cur_start = i
            elif cur_type != ent_type:
                # type switched unexpectedly; close previous and start new
                close_span(i - 1)
                cur_type = ent_type
                cur_start = i

            if prefix == "E":
                # close at i
                close_span(i)
            continue

        # unknown prefix: treat as O
        close_span(i - 1)

    # close if still open
    close_span(len(tags) - 1)
    return spans


def spans_to_dicts(spans: List[Span]) -> List[dict]:
    """
    Helper: convert Span objects to plain dicts for JSON output.
    """
    return [
        {"type": s.ent_type, "text": s.text, "start": s.start, "end": s.end}
        for s in spans
    ]