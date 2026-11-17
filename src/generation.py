"""Evidence gate, grounded LLM generation, and safe abstention."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterator

from src.citations import (
    CITATION_PATTERN,
    _source_key,
    _split_ids,
    format_source_block,
    number_citations,
    validate_citations,
)
from src.config import Settings, get_settings
from src.conversation import (
    conversational_reply,
    detect_session_level_hint,
    rewrite_program_overview_query,
)
from src.retrieval import RetrievedPassage, is_dev_fixture, load_default_store, retrieve
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

ABSTAIN_MESSAGE = (
    "I couldn't find this in the catalogs. "
    "Please check the Undergraduate or Graduate Catalog, "
    "the Student Handbook, or contact the Office of the Registrar for guidance."
)
ABSTAIN_TITLE = "I couldn't find this in the catalogs"

SYSTEM_PROMPT = """You are CampusAsk, a university policy assistant for the University of Central Missouri (UCM).
Answer ONLY using the provided evidence passages. Rules:
1. Every substantive policy claim must include a citation in the form [cite:CHUNK_ID] using an ID from the evidence list.
2. If the evidence is insufficient, conflicting, or does not address the question, reply with exactly: ABSTAIN
3. Do not invent deadlines, requirements, office names, exceptions, or page numbers.
4. If undergraduate and graduate rules differ and the user level is unclear, say so and cite both when present.
5. Keep answers concise and student-friendly.
6. Never invent citation IDs that are not in the evidence list.
7. For thesis vs non-thesis, GPA, probation, or credit-hour questions, synthesize the relevant catalog rules from the evidence rather than abstaining when passages clearly discuss those topics.
8. Never mention student ID numbers; you will not be given any.
"""


@dataclass
class AnswerResult:
    status: str  # answered | abstained | conversational | error
    answer: str
    passages: list[RetrievedPassage] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    invalid_citation_ids: list[str] = field(default_factory=list)
    max_score: float | None = None
    error: str | None = None
    intent: str | None = None
    suggested_academic_level: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "answer": self.answer,
            "sources": self.sources,
            "cited_ids": self.cited_ids,
            "invalid_citation_ids": self.invalid_citation_ids,
            "max_score": self.max_score,
            "error": self.error,
            "intent": self.intent,
            "suggested_academic_level": self.suggested_academic_level,
            "passage_ids": [p.chunk_id for p in self.passages],
        }


def evidence_is_sufficient(
    passages: list[RetrievedPassage],
    *,
    min_score: float,
) -> bool:
    if not passages:
        return False
    return max(p.score for p in passages) >= min_score


def _format_evidence(passages: list[RetrievedPassage]) -> str:
    blocks: list[str] = []
    for p in passages:
        loc = p.location_label()
        blocks.append(
            f"CHUNK_ID: {p.chunk_id}\n"
            f"TITLE: {p.chunk.title}\n"
            f"LEVEL: {p.chunk.level} | YEAR: {p.chunk.year}\n"
            f"LOCATION: {loc}\n"
            f"URL: {p.chunk.url}\n"
            f"TEXT: {p.chunk.text}\n"
        )
    return "\n---\n".join(blocks)


LLM_UNAVAILABLE_MESSAGE = (
    "CampusAsk couldn't reach its language model just now. "
    "Please try again in a moment."
)


class LLMNotConfiguredError(RuntimeError):
    """No credentials for the selected LLM provider."""


@lru_cache(maxsize=4)
def _anthropic_client(api_key: str, timeout: float):
    import anthropic

    kwargs: dict[str, Any] = {"timeout": timeout, "max_retries": 2}
    if api_key:
        kwargs["api_key"] = api_key
    return anthropic.Anthropic(**kwargs)


@lru_cache(maxsize=4)
def _openai_client(api_key: str, base_url: str, timeout: float):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=2)


def _call_anthropic(prompt: str, settings: Settings) -> str:
    if not settings.anthropic_api_key:
        raise LLMNotConfiguredError(
            "ANTHROPIC_API_KEY is not set. Add it to .env (LLM_PROVIDER=anthropic)."
        )
    client = _anthropic_client(settings.anthropic_api_key, settings.llm_timeout_seconds)
    # Thinking is always on for Claude Opus 5.5; effort controls its depth.
    # Server-side fallback re-runs a safety-declined request on another model.
    response = client.beta.messages.create(
        model=settings.llm_model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": settings.llm_effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        logger.warning("Claude declined the request (request_id=%s)", response._request_id)
        return "ABSTAIN"
    if response.stop_reason == "max_tokens":
        logger.warning("Claude hit max_tokens (request_id=%s)", response._request_id)
    text = "".join(block.text for block in response.content if block.type == "text")
    return text.strip()


def _call_openai_compatible(prompt: str, settings: Settings) -> str:
    if not settings.llm_api_key:
        raise LLMNotConfiguredError(
            "LLM_API_KEY is not set. Add it to .env (LLM_PROVIDER=openai)."
        )
    client = _openai_client(
        settings.llm_api_key, settings.llm_base_url, settings.llm_timeout_seconds
    )
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def _call_llm(prompt: str, settings: Settings) -> str:
    if settings.llm_provider == "anthropic":
        return _call_anthropic(prompt, settings)
    return _call_openai_compatible(prompt, settings)


def _is_abstain_response(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if re.fullmatch(r"ABSTAIN\.?", cleaned, flags=re.IGNORECASE):
        return True
    if cleaned.upper().startswith("ABSTAIN") and len(cleaned) < 40:
        return True
    return False


def generate_answer(
    question: str,
    passages: list[RetrievedPassage],
    *,
    settings: Settings | None = None,
    academic_level: str = "all",
    min_score: float | None = None,
) -> AnswerResult:
    settings = settings or get_settings()
    max_score = max((p.score for p in passages), default=None)
    gate = settings.evidence_min_score if min_score is None else min_score

    if not evidence_is_sufficient(passages, min_score=gate):
        return AnswerResult(
            status="abstained",
            answer=ABSTAIN_MESSAGE,
            passages=passages,
            sources=format_source_block(passages, question=question),
            max_score=max_score,
        )

    prompt = _build_prompt(question, passages, academic_level)
    try:
        raw = _call_llm(prompt, settings)
    except Exception as exc:  # noqa: BLE001
        return _llm_failure(exc, question, passages, max_score)
    return _finalize_answer(raw, question, passages, max_score)


def _build_prompt(question: str, passages: list[RetrievedPassage], academic_level: str) -> str:
    return (
        f"Academic level filter selected by user: {academic_level}\n\n"
        f"Question: {question}\n\n"
        f"Evidence passages:\n{_format_evidence(passages)}\n\n"
        "Write the answer now. Cite with [cite:CHUNK_ID]. "
        "If evidence is insufficient, reply ABSTAIN."
    )


def _llm_failure(
    exc: Exception,
    question: str,
    passages: list[RetrievedPassage],
    max_score: float | None,
) -> AnswerResult:
    if isinstance(exc, LLMNotConfiguredError):
        logger.warning("LLM unavailable: %s", exc)
        error = "llm_not_configured"
    else:
        # Details go to the server log only; users get a generic message.
        logger.error("LLM generation failed", exc_info=exc)
        error = type(exc).__name__
    return AnswerResult(
        status="error",
        answer=LLM_UNAVAILABLE_MESSAGE,
        passages=passages,
        sources=format_source_block(passages, question=question),
        max_score=max_score,
        error=error,
    )


def _finalize_answer(
    raw: str,
    question: str,
    passages: list[RetrievedPassage],
    max_score: float | None,
) -> AnswerResult:
    """Abstain check, citation validation, and [cite:ID] → [n] numbering."""
    if _is_abstain_response(raw):
        return AnswerResult(
            status="abstained",
            answer=ABSTAIN_MESSAGE,
            passages=passages,
            sources=format_source_block(passages, question=question),
            max_score=max_score,
        )

    check = validate_citations(raw, passages)
    # If the model cited nothing valid, treat as abstention rather than uncited claims.
    if not check.valid_ids:
        return AnswerResult(
            status="abstained",
            answer=ABSTAIN_MESSAGE,
            passages=passages,
            sources=format_source_block(passages, question=question),
            cited_ids=[],
            invalid_citation_ids=check.invalid_ids,
            max_score=max_score,
        )

    sources = format_source_block(passages, check.valid_ids, question=question)
    return AnswerResult(
        status="answered",
        # [cite:ID] → [n], matching each source card's "ref" for inline markers.
        answer=number_citations(raw, sources),
        passages=passages,
        sources=sources,
        cited_ids=check.valid_ids,
        invalid_citation_ids=check.invalid_ids,
        max_score=max_score,
    )


def _prefer_program_title_passages(
    passages: list[RetrievedPassage],
    program_label: str,
) -> list[RetrievedPassage]:
    """Surface Grad Catalog program-title pages ahead of course-list noise."""
    if not passages or not program_label:
        return passages
    label = program_label.lower()
    title_cues = (
        f"{label}, ms",
        f"master of science degree in {label}",
        f"{label} ms (",
    )

    def rank_key(item: RetrievedPassage) -> tuple[int, int, float]:
        text = (item.chunk.text or "").lower()
        title_hit = any(cue in text for cue in title_cues)
        hours_hit = "30 hours" in text or "(53-976)" in text
        # Prefer title pages, then high dense/phrase scores.
        return (0 if title_hit else 1, 0 if hours_hit else 1, -float(item.score))

    return sorted(passages, key=rank_key)


@dataclass
class _Prepared:
    """Everything needed to generate an answer, or an early result that skips the LLM."""

    early: AnswerResult | None = None
    settings: Settings | None = None
    passages: list[RetrievedPassage] = field(default_factory=list)
    llm_question: str = ""
    academic_level: str = "all"
    min_score: float | None = None
    suggested: str | None = None
    overview: Any = None


def answer_question(
    question: str,
    *,
    store: VectorStore | None = None,
    settings: Settings | None = None,
    top_k: int | None = None,
    academic_level: str = "all",
    year: str | None = None,
    first_name: str | None = None,
) -> AnswerResult:
    """Conversational short-circuit, else retrieve → evidence gate → generate."""
    prep = _prepare(
        question,
        store=store,
        settings=settings,
        top_k=top_k,
        academic_level=academic_level,
        year=year,
        first_name=first_name,
    )
    if prep.early is not None:
        return prep.early
    result = generate_answer(
        prep.llm_question,
        prep.passages,
        settings=prep.settings,
        academic_level=prep.academic_level,
        min_score=prep.min_score,
    )
    return _apply_overview(result, prep)


class LLMRefusalError(RuntimeError):
    """Claude (and any fallback model) declined the request."""


class _CitationStreamer:
    """Turn streamed ``[cite:ID]`` text into ``[n]`` markers as it arrives.

    Refs are assigned per document page in first-cited order, which is the same
    order ``format_source_block`` uses for the final answer, so numbers shown
    while streaming match the final source list. Text that might still become a
    marker (an unclosed ``[``) is held back, as is a reply that is only ABSTAIN.
    """

    _MAX_HELD = 300

    def __init__(self, passages: list[RetrievedPassage]):
        self._key_by_id = {
            p.chunk_id: _source_key(p) for p in passages if not is_dev_fixture(p.chunk)
        }
        self._ref_by_key: dict[tuple[str, str], int] = {}
        self._buf = ""
        self._started = False

    def _refs(self, raw_ids: str) -> str:
        refs: list[int] = []
        for cid in _split_ids(raw_ids):
            key = self._key_by_id.get(cid)
            if key is None:
                continue
            ref = self._ref_by_key.setdefault(key, len(self._ref_by_key) + 1)
            if ref not in refs:
                refs.append(ref)
        return "".join(f"[{r}]" for r in refs)

    def feed(self, text: str) -> str:
        self._buf += text
        if not self._started:
            head = self._buf.lstrip().upper()
            if "ABSTAIN".startswith(head) or head.startswith("ABSTAIN"):
                return ""
            self._started = True
        out: list[str] = []
        while self._buf:
            start = self._buf.find("[")
            if start == -1:
                out.append(self._buf)
                self._buf = ""
                break
            out.append(self._buf[:start])
            self._buf = self._buf[start:]
            end = self._buf.find("]")
            if end == -1:
                if len(self._buf) > self._MAX_HELD:  # not a citation marker after all
                    out.append(self._buf)
                    self._buf = ""
                break
            token, self._buf = self._buf[: end + 1], self._buf[end + 1 :]
            match = CITATION_PATTERN.fullmatch(token)
            out.append(self._refs(match.group(1)) if match else token)
        return "".join(out)

    def flush(self) -> str:
        rest = self._buf if self._started else ""
        self._buf = ""
        return rest


def _stream_llm(prompt: str, settings: Settings) -> Iterator[str]:
    """Yield answer text as the model writes it."""
    if settings.llm_provider != "anthropic":
        # OpenAI-compatible path: one chunk (keeps behavior identical to /v1/ask).
        yield _call_openai_compatible(prompt, settings)
        return
    if not settings.anthropic_api_key:
        raise LLMNotConfiguredError(
            "ANTHROPIC_API_KEY is not set. Add it to .env (LLM_PROVIDER=anthropic)."
        )
    client = _anthropic_client(settings.anthropic_api_key, settings.llm_timeout_seconds)
    with client.beta.messages.stream(
        model=settings.llm_model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": settings.llm_effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        for text in stream.text_stream:
            yield text
        final = stream.get_final_message()
    if final.stop_reason == "refusal":
        raise LLMRefusalError(f"request_id={final._request_id}")
    if final.stop_reason == "max_tokens":
        logger.warning("Claude hit max_tokens (request_id=%s)", final._request_id)


def stream_answer_events(
    question: str,
    *,
    store: VectorStore | None = None,
    settings: Settings | None = None,
    top_k: int | None = None,
    academic_level: str = "all",
    year: str | None = None,
    first_name: str | None = None,
) -> Iterator[tuple[str, Any]]:
    """Streaming variant of ``answer_question``.

    Yields ``("status", {...})`` once retrieval is done, ``("delta", {"text": ...})``
    while the answer is written, and finally ``("done", AnswerResult)``. The final
    result is computed exactly as in ``answer_question`` (citation validation,
    abstention) and is authoritative: clients replace the streamed text with it.
    """
    prep = _prepare(
        question,
        store=store,
        settings=settings,
        top_k=top_k,
        academic_level=academic_level,
        year=year,
        first_name=first_name,
    )
    if prep.early is not None:
        yield "done", prep.early
        return

    assert prep.settings is not None
    passages = prep.passages
    max_score = max((p.score for p in passages), default=None)
    gate = prep.settings.evidence_min_score if prep.min_score is None else prep.min_score
    if not evidence_is_sufficient(passages, min_score=gate):
        # No LLM call: generate_answer returns the abstention immediately.
        result = generate_answer(
            prep.llm_question,
            passages,
            settings=prep.settings,
            academic_level=prep.academic_level,
            min_score=prep.min_score,
        )
        yield "done", _apply_overview(result, prep)
        return

    yield "status", {"stage": "writing", "passages": len(passages)}

    prompt = _build_prompt(prep.llm_question, passages, prep.academic_level)
    streamer = _CitationStreamer(passages)
    parts: list[str] = []
    try:
        for chunk in _stream_llm(prompt, prep.settings):
            parts.append(chunk)
            visible = streamer.feed(chunk)
            if visible:
                yield "delta", {"text": visible}
        tail = streamer.flush()
        if tail:
            yield "delta", {"text": tail}
    except LLMRefusalError as exc:
        logger.warning("Claude declined the request (%s)", exc)
        parts = ["ABSTAIN"]
    except Exception as exc:  # noqa: BLE001
        yield "done", _apply_overview(_llm_failure(exc, prep.llm_question, passages, max_score), prep)
        return

    result = _finalize_answer("".join(parts), prep.llm_question, passages, max_score)
    yield "done", _apply_overview(result, prep)


def _apply_overview(result: AnswerResult, prep: _Prepared) -> AnswerResult:
    """Program-overview post-processing shared by the blocking and streaming paths."""
    overview = prep.overview
    result.suggested_academic_level = prep.suggested
    if overview is None:
        return result
    result.intent = "program_overview"
    # Never abstain on “I am a masters student in <known program>”.
    if result.status in {"abstained", "error"}:
        return AnswerResult(
            status="conversational",
            answer=overview.soft_prompt,
            intent="student_context",
            suggested_academic_level=prep.suggested,
            passages=prep.passages,
            max_score=max((p.score for p in prep.passages), default=None),
            error=result.error,
        )
    return result


def _prepare(
    question: str,
    *,
    store: VectorStore | None,
    settings: Settings | None,
    top_k: int | None,
    academic_level: str,
    year: str | None,
    first_name: str | None,
) -> _Prepared:
    """Conversational short-circuit, level resolution, and retrieval."""
    chat = conversational_reply(question, first_name=first_name)
    if chat is not None:
        return _Prepared(
            early=AnswerResult(
                status="conversational",
                answer=chat.answer,
                intent=chat.intent,
                suggested_academic_level=chat.suggested_academic_level,
            )
        )

    settings = settings or get_settings()
    store = store or load_default_store(settings)
    level = None if academic_level.lower() == "all" else academic_level.lower()
    level_hint = detect_session_level_hint(question)
    overview = rewrite_program_overview_query(question)
    retrieval_query = overview.retrieval_query if overview else question
    llm_question = overview.llm_question if overview else question
    suggested = (
        overview.suggested_academic_level
        if overview
        else level_hint
    )

    # Masters / known-program identity overrides a mismatched Undergraduate UI filter
    # so CS MS (53-976) Grad Catalog pages are not blocked by BS chunks.
    wants_graduate = level_hint == "graduate" or (
        overview is not None and overview.suggested_academic_level == "graduate"
    )
    if wants_graduate:
        level = "graduate"
        suggested = "graduate"
    elif not level and level_hint:
        level = level_hint
    elif overview and overview.suggested_academic_level and not level:
        level = overview.suggested_academic_level

    passages = retrieve(
        retrieval_query,
        store,
        settings=settings,
        top_k=top_k,
        level=level,
        year=year,
        # Also scan the raw student utterance for program-title phrase rules.
        phrase_query=question if overview is not None else None,
    )

    if overview is not None:
        passages = _prefer_program_title_passages(passages, overview.program_label)

    # Program identity with weak evidence → soft prompt, never a dry abstention.
    if overview is not None and not evidence_is_sufficient(
        passages, min_score=settings.evidence_min_score
    ):
        return _Prepared(
            early=AnswerResult(
                status="conversational",
                answer=overview.soft_prompt,
                intent="student_context",
                suggested_academic_level=suggested,
                passages=passages,
                max_score=max((p.score for p in passages), default=None),
            )
        )

    # Known program overview: slightly lower gate so title-page phrase hits can answer.
    overview_gate = None
    if overview is not None:
        overview_gate = max(0.28, float(settings.evidence_min_score) - 0.08)

    return _Prepared(
        settings=settings,
        passages=passages,
        llm_question=llm_question,
        academic_level=level or academic_level,
        min_score=overview_gate,
        suggested=suggested,
        overview=overview,
    )


def answer_question_json(question: str, **kwargs: Any) -> str:
    return json.dumps(answer_question(question, **kwargs).to_dict(), indent=2)
