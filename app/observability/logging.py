"""Structured JSON Logging with Intent vs. Outcome Capture using structlog.

Configures machine-readable JSON logs with automatic PII redaction, OpenTelemetry
trace correlation, and explicit `agent_intent` (pre-execution) and `tool_outcome`
(post-execution) audit records for every reasoning cycle.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any, Deque, Dict, List, Optional
import uuid

from app.observability.pii_scrubber import scrub_pii

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    structlog = None  # type: ignore[assignment]
    HAS_STRUCTLOG = False


RECENT_AUDIT_LOGS: Deque[Dict[str, Any]] = deque(maxlen=250)


def _pii_redaction_processor(
    logger: Any, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Structlog processor that scrubs sensitive PII from all log event fields."""
    return scrub_pii(event_dict)


def configure_structured_logging() -> None:
    """Initializes structlog with JSONRenderer and ISO-8601 UTC timestamps."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    if HAS_STRUCTLOG and structlog is not None:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                _pii_redaction_processor,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )


configure_structured_logging()


class StructuredJSONLogger:
    """Unified structured JSON logger wrapping structlog with fallback JSON emitter."""

    def __init__(self, component: str = "argon-learn") -> None:
        self.component = component
        self._logger = (
            structlog.get_logger(component=component)
            if HAS_STRUCTLOG and structlog is not None
            else logging.getLogger(component)
        )

    def _emit(self, level: str, event: str, **kwargs: Any) -> Dict[str, Any]:
        sanitized_kwargs = scrub_pii(kwargs)
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": level.upper(),
            "component": self.component,
            "event": event,
            **sanitized_kwargs,
        }
        RECENT_AUDIT_LOGS.append(record)

        if HAS_STRUCTLOG and structlog is not None:
            log_fn = getattr(self._logger, level.lower(), self._logger.info)
            log_fn(event, **sanitized_kwargs)
        else:
            sys.stdout.write(json.dumps(record) + "\n")
            sys.stdout.flush()
        return record

    def info(self, event: str, **kwargs: Any) -> Dict[str, Any]:
        return self._emit("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> Dict[str, Any]:
        return self._emit("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> Dict[str, Any]:
        return self._emit("ERROR", event, **kwargs)


def get_logger(component: str = "argon-learn") -> StructuredJSONLogger:
    """Returns a configured StructuredJSONLogger instance for the given component."""
    return StructuredJSONLogger(component=component)


def log_agent_intent(
    *,
    agent_name: str,
    model_id: str,
    intended_action: str,
    target_tool: str,
    rationale: str,
    input_parameters: Dict[str, Any],
    session_id: str = "default",
    trace_id: Optional[str] = None,
) -> str:
    """Logs the agent's explicit intent BEFORE executing a tool or sub-agent call.

    Args:
        agent_name: Name of the ADK agent initiating the reasoning step.
        model_id: Underlying Gemini model identifier selected by the router.
        intended_action: High-level description of what the agent intends to achieve.
        target_tool: Specific tool or sub-agent scheduled for invocation.
        rationale: Pedagogical or architectural reason for this action.
        input_parameters: Sanitized input arguments passed to the tool.
        session_id: Active learner session identifier.
        trace_id: OpenTelemetry correlation trace ID.

    Returns:
        A unique `intent_id` used to correlate with the subsequent `log_tool_outcome`.
    """
    intent_id = f"intent-{uuid.uuid4().hex[:12]}"
    logger = get_logger(agent_name)
    logger.info(
        "agent_intent",
        audit_stage="PRE_EXECUTION",
        intent_id=intent_id,
        agent_name=agent_name,
        model_id=model_id,
        agent_intent=intended_action,
        target_tool=target_tool,
        rationale=rationale,
        input_parameters=scrub_pii(input_parameters),
        session_id=session_id,
        trace_id=trace_id or "local-trace",
    )
    return intent_id


def log_tool_outcome(
    *,
    intent_id: str,
    agent_name: str,
    tool_name: str,
    status: str,
    latency_ms: float,
    outcome_summary: str,
    output_metadata: Optional[Dict[str, Any]] = None,
    error_payload: Optional[Dict[str, Any]] = None,
    session_id: str = "default",
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Logs the actual outcome AFTER tool or sub-agent execution completes.

    Args:
        intent_id: Correlation identifier returned by `log_agent_intent`.
        agent_name: Name of the invoking ADK agent.
        tool_name: Executed tool or sub-agent name.
        status: Execution result status ('SUCCESS', 'RECOVERABLE_ERROR', 'BLOCKED_HITL').
        latency_ms: Wall-clock execution duration in milliseconds.
        outcome_summary: Concise description of the actual result vs. intent.
        output_metadata: Key metrics or artifact summaries produced by the tool.
        error_payload: Structured remediation payload if an exception occurred.
        session_id: Active learner session identifier.
        trace_id: OpenTelemetry correlation trace ID.

    Returns:
        The emitted structured JSON log dictionary.
    """
    logger = get_logger(agent_name)
    level = "INFO" if status == "SUCCESS" else "WARNING"
    return logger._emit(
        level,
        "tool_outcome",
        audit_stage="POST_EXECUTION",
        intent_id=intent_id,
        agent_name=agent_name,
        tool_name=tool_name,
        tool_outcome=outcome_summary,
        status=status,
        latency_ms=round(latency_ms, 2),
        output_metadata=scrub_pii(output_metadata or {}),
        error_payload=scrub_pii(error_payload) if error_payload else None,
        session_id=session_id,
        trace_id=trace_id or "local-trace",
    )


def get_recent_audit_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Returns the most recent structured JSON logs for UI & test inspection."""
    items = list(RECENT_AUDIT_LOGS)
    return items[-limit:]
