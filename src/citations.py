"""Citation validation against retrieved chunk IDs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.retrieval import RetrievedPassage, is_dev_fixture

CITATION_PATTERN = re.compile(r"\[cite:([^\]]+)\]", re.IGNORECASE)


@dataclass
class CitationCheckResult:
    valid_ids: list[str]
    invalid_ids: list[str]
    cleaned_answer: str
    all_valid: bool


def _split_ids(raw: str) -> list[str]:
    """``[cite:a, b]`` → ``["a", "b"]`` (models sometimes group citations)."""
    parts = [p.strip() for p in raw.split(",")]
    return [p.removeprefix("cite:").strip() for p in parts if p.strip()]


def extract_citation_ids(text: str) -> list[str]:
    out: list[str] = []
    for m in CITATION_PATTERN.finditer(text or ""):
        out.extend(_split_ids(m.group(1)))
    return out


def strip_citation_markers(text: str) -> str:
    """Remove all [cite:…] markers from visible answer text."""
    cleaned = CITATION_PATTERN.sub("", text or "")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r" +\.", ".", cleaned)
    return cleaned.strip()


def validate_citations(
    answer: str,
    passages: list[RetrievedPassage],
) -> CitationCheckResult:
    """Validate [cite:ID] against retrieved chunks; strip markers from visible text."""
    allowed = {p.chunk_id for p in passages}
    found = extract_citation_ids(answer)
    valid: list[str] = []
    invalid: list[str] = []
    for cid in found:
        if cid in allowed:
            if cid not in valid:
                valid.append(cid)
        else:
            if cid not in invalid:
                invalid.append(cid)

    cleaned = strip_citation_markers(answer or "")

    return CitationCheckResult(
        valid_ids=valid,
        invalid_ids=invalid,
        cleaned_answer=cleaned,
        all_valid=len(invalid) == 0,
    )


_STOPWORDS = {
    "the", "and", "for", "are", "what", "how", "does", "can", "with", "that",
    "this", "from", "your", "you", "have", "will", "into", "about", "when",
    "which", "who", "why", "is", "a", "an", "of", "to", "in", "on", "or", "my",
    "i", "do", "be", "at", "as", "by", "it", "if", "student", "students", "ucm",
}


def _query_terms(question: str | None) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9'-]+", (question or "").lower())
    return [w for w in dict.fromkeys(words) if w not in _STOPWORDS and len(w) > 2]


def best_excerpt(text: str, question: str | None = None, *, width: int = 360) -> str:
    """Pick the ~width-char window of ``text`` that best matches the question.

    Falls back to the start of the passage when no question terms occur.
    Windows snap to sentence/word boundaries so excerpts read cleanly.
    """
    text = " ".join((text or "").split())
    if len(text) <= width:
        return text
    terms = _query_terms(question)
    lower = text.lower()
    best_start, best_hits = 0, 0
    if terms:
        step = max(20, width // 6)
        for start in range(0, max(1, len(text) - width) + step, step):
            window = lower[start : start + width]
            hits = sum(window.count(t) for t in terms)
            if hits > best_hits:
                best_start, best_hits = start, hits
    start = best_start
    if start > 0:
        # Prefer starting at a sentence boundary just before the window.
        sentence = max(text.rfind(". ", max(0, start - 120), start + 40), -1)
        start = sentence + 2 if sentence >= 0 else text.find(" ", start) + 1
    end = min(len(text), start + width)
    if end < len(text):
        space = text.rfind(" ", start, end)
        end = space if space > start else end
    snippet = text[start:end].strip()
    return ("…" if start > 0 else "") + snippet + ("…" if end < len(text) else "")


def _source_link(p: RetrievedPassage) -> str:
    url = p.chunk.url or ""
    if p.chunk.content_type == "pdf" and p.chunk.page and url.lower().endswith(".pdf"):
        return f"{url}#page={p.chunk.page}"
    return url


def _source_key(p: RetrievedPassage) -> tuple[str, str]:
    """Passages from the same document page are one source for the reader."""
    loc = str(p.chunk.page) if p.chunk.page is not None else (p.chunk.section or p.chunk_id)
    return (p.chunk.source_id or p.chunk.title, loc)


def format_source_block(
    passages: list[RetrievedPassage],
    cited_ids: list[str] | None = None,
    *,
    question: str | None = None,
) -> list[dict]:
    """Build UI-friendly source cards; prefer cited IDs, else all retrieved.

    Passages from the same document page are merged into one card, numbered
    with ``ref`` (1-based) in display order. Development fixtures are never
    shown as production sources.
    """
    by_id = {p.chunk_id: p for p in passages if not is_dev_fixture(p.chunk)}
    order = cited_ids if cited_ids else [p.chunk_id for p in passages]
    cards: dict[tuple[str, str], dict] = {}
    for cid in order:
        p = by_id.get(cid)
        if p is None:
            continue
        key = _source_key(p)
        card = cards.get(key)
        if card is None:
            cards[key] = {
                "ref": len(cards) + 1,
                "chunk_id": p.chunk_id,
                "chunk_ids": [p.chunk_id],
                "title": p.chunk.title,
                "url": _source_link(p),
                "location": p.location_label(),
                "page": p.chunk.page,
                "section": p.chunk.section,
                "year": p.chunk.year,
                "level": p.chunk.level,
                "score": round(p.score, 4),
                "excerpt": best_excerpt(p.chunk.text, question),
            }
        elif cid not in card["chunk_ids"]:
            card["chunk_ids"].append(cid)
    return list(cards.values())


def number_citations(answer: str, sources: list[dict]) -> str:
    """Replace ``[cite:ID]`` markers with ``[n]`` matching each source's ``ref``.

    Unknown IDs are dropped; repeated refs inside one marker are collapsed, and
    adjacent duplicates such as ``[1][1]`` become ``[1]``.
    """
    ref_by_id = {cid: s["ref"] for s in sources for cid in s.get("chunk_ids", [s["chunk_id"]])}

    def repl(match: re.Match[str]) -> str:
        refs: list[int] = []
        for cid in _split_ids(match.group(1)):
            ref = ref_by_id.get(cid)
            if ref is not None and ref not in refs:
                refs.append(ref)
        return "".join(f"[{r}]" for r in refs)

    text = CITATION_PATTERN.sub(repl, answer or "")
    text = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", text)
    text = re.sub(r"[ \t]+(\[\d+\])", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
