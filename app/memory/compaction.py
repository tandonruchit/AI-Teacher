"""ADK History Compaction & Sliding Window Token Management.

Prevents context window bloat over extended interactive tutoring sessions by:
1. Estimating token consumption across system prompt, episodic summary, and turns.
2. Enforcing a configurable sliding window (`sliding_window_turns`) of verbatim recent
   messages while compacting older turns into a dense pedagogical memory digest.
3. Exposing ADK-compatible compaction hooks and Gemini Context Caching metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from app.config import settings
from app.observability.logging import get_logger
from app.observability.pii_scrubber import scrub_pii
from app.observability.tracing import trace_agent_span

logger = get_logger("memory.compaction")


def estimate_token_count(text: str) -> int:
    """Estimates LLM token count using a conservative 4-chars-per-token heuristic."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass
class TurnMessage:
    """Represents a single conversational turn in an ADK session."""

    role: str
    content: str
    concept_id: str = ""
    mastery_level: str = "Novice"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    token_count: int = 0

    def __post_init__(self) -> None:
        self.content = scrub_pii(self.content)
        if self.token_count <= 0:
            self.token_count = estimate_token_count(self.content)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "concept_id": self.concept_id,
            "mastery_level": self.mastery_level,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TurnMessage:
        return cls(
            role=str(data.get("role", "user")),
            content=str(data.get("content", "")),
            concept_id=str(data.get("concept_id", "")),
            mastery_level=str(data.get("mastery_level", "Novice")),
            timestamp=str(
                data.get("timestamp", datetime.now(timezone.utc).isoformat())
            ),
            token_count=int(data.get("token_count", 0)),
        )


@dataclass
class CompactionResult:
    """Structured telemetry returned after running history compaction."""

    retained_turns: List[TurnMessage]
    compacted_summary: str
    original_turn_count: int
    retained_turn_count: int
    original_tokens: int
    compacted_tokens: int
    was_compacted: bool
    context_cache_key: str


class ConversationCompactor:
    """Sliding-window and token-budget history compactor for ADK sessions."""

    def __init__(
        self,
        max_context_tokens: int | None = None,
        sliding_window_turns: int | None = None,
    ) -> None:
        self.max_context_tokens = max_context_tokens or settings.max_context_tokens
        self.sliding_window_turns = (
            sliding_window_turns or settings.sliding_window_turns
        )

    def summarize_evicted_turns(
        self,
        existing_summary: str,
        evicted_turns: List[TurnMessage],
    ) -> str:
        """Synthesizes evicted conversational turns into a compact episodic digest."""
        if not evicted_turns:
            return existing_summary

        concepts_covered: List[str] = []
        key_points: List[str] = []

        for turn in evicted_turns:
            if turn.concept_id and turn.concept_id not in concepts_covered:
                concepts_covered.append(turn.concept_id)
            snippet = " ".join(turn.content.strip().split())
            if len(snippet) > 110:
                snippet = snippet[:107] + "..."
            key_points.append(f"[{turn.role.upper()}@{turn.mastery_level}] {snippet}")

        digest_header = (
            f"Concepts explored: {', '.join(concepts_covered)}. "
            if concepts_covered
            else ""
        )
        new_digest = digest_header + " | ".join(key_points[-6:])

        if existing_summary:
            merged = f"{existing_summary.strip()} || {new_digest}"
        else:
            merged = new_digest

        # Cap episodic summary length to prevent unbounded summary growth
        if len(merged) > 900:
            merged = "... " + merged[-896:]
        return scrub_pii(merged)

    def compact_history(
        self,
        turns: List[TurnMessage | Dict[str, Any]],
        existing_summary: str = "",
        session_id: str = "default",
    ) -> CompactionResult:
        """Compacts conversational history when turn count or token budget is exceeded.

        Args:
            turns: Chronological list of TurnMessage objects or dicts.
            existing_summary: Previously compacted episodic summary string.
            session_id: Active learner session identifier.

        Returns:
            CompactionResult containing retained sliding-window turns, updated
            episodic summary, and token reduction metrics.
        """
        with trace_agent_span(
            "ConversationCompactor.compact_history",
            agent_name="CompactionEngine",
            stage="memory",
            attributes={"argon.session_id": session_id},
        ) as span:
            normalized: List[TurnMessage] = [
                t if isinstance(t, TurnMessage) else TurnMessage.from_dict(t)
                for t in turns
            ]
            original_count = len(normalized)
            summary_tokens = estimate_token_count(existing_summary)
            turn_tokens = sum(t.token_count for t in normalized)
            original_tokens = summary_tokens + turn_tokens

            needs_window_compaction = original_count > self.sliding_window_turns
            needs_token_compaction = original_tokens > self.max_context_tokens

            if not needs_window_compaction and not needs_token_compaction:
                span.set_attribute("argon.compaction.triggered", False)
                return CompactionResult(
                    retained_turns=normalized,
                    compacted_summary=existing_summary,
                    original_turn_count=original_count,
                    retained_turn_count=original_count,
                    original_tokens=original_tokens,
                    compacted_tokens=original_tokens,
                    was_compacted=False,
                    context_cache_key=f"adk-cache-{session_id}-v0",
                )

            # First apply sliding window cutoff
            keep_count = min(original_count, self.sliding_window_turns)
            evicted = normalized[:-keep_count] if keep_count > 0 else normalized
            retained = normalized[-keep_count:] if keep_count > 0 else []

            # If still above token budget, evict oldest turns until under target budget
            target_budget = int(self.max_context_tokens * 0.75)
            while (
                len(retained) > 2
                and sum(t.token_count for t in retained) > target_budget
            ):
                evicted.append(retained.pop(0))

            updated_summary = self.summarize_evicted_turns(existing_summary, evicted)
            compacted_tokens = estimate_token_count(updated_summary) + sum(
                t.token_count for t in retained
            )

            span.set_attribute("argon.compaction.triggered", True)
            span.set_attribute("argon.compaction.original_tokens", original_tokens)
            span.set_attribute("argon.compaction.compacted_tokens", compacted_tokens)

            logger.info(
                "session_history_compacted",
                session_id=session_id,
                original_turns=original_count,
                retained_turns=len(retained),
                original_tokens=original_tokens,
                compacted_tokens=compacted_tokens,
            )

            return CompactionResult(
                retained_turns=retained,
                compacted_summary=updated_summary,
                original_turn_count=original_count,
                retained_turn_count=len(retained),
                original_tokens=original_tokens,
                compacted_tokens=compacted_tokens,
                was_compacted=True,
                context_cache_key=f"adk-cache-{session_id}-{len(evicted)}",
            )

    def build_adk_context_window(
        self,
        session_id: str,
        turns: List[TurnMessage | Dict[str, Any]],
        existing_summary: str = "",
    ) -> Tuple[str, List[Dict[str, Any]], CompactionResult]:
        """Builds a compacted context window ready for ADK LlmAgent invocation."""
        result = self.compact_history(
            turns=turns,
            existing_summary=existing_summary,
            session_id=session_id,
        )
        formatted_messages = [t.to_dict() for t in result.retained_turns]
        return result.compacted_summary, formatted_messages, result
