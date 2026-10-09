"""Distributed Tracing using OpenTelemetry & Google Cloud Trace Exporter.

Links parent and child spans across the entire multi-agent execution pipeline
(Coordinator -> Triage -> Pedagogy -> Visualizer -> Tool Calls) and exports
spans to Google Cloud Trace (`opentelemetry-exporter-gcp-trace`) while keeping
an inspectable in-memory span buffer for the live Cloud Run / Trace UI HUD.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
import time
from typing import Any, Deque, Dict, Iterator, List, Optional
import uuid

from app.config import settings

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    HAS_OTEL = True
except ImportError:
    trace = None  # type: ignore[assignment]
    HAS_OTEL = False

try:
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
    HAS_GCP_TRACE_EXPORTER = True
except ImportError:
    CloudTraceSpanExporter = None  # type: ignore[assignment]
    HAS_GCP_TRACE_EXPORTER = False


_current_trace_id: ContextVar[str] = ContextVar("current_trace_id", default="")
_current_span_id: ContextVar[str] = ContextVar("current_span_id", default="")

RECENT_SPANS: Deque[Dict[str, Any]] = deque(maxlen=200)


def _setup_tracer_provider() -> Any:
    """Configures OpenTelemetry TracerProvider and attaches CloudTraceSpanExporter."""
    if not HAS_OTEL or trace is None:
        return None

    resource = Resource.create(
        {
            "service.name": settings.app_name,
            "service.version": "1.0.0",
            "deployment.environment": settings.environment,
            "cloud.provider": "gcp",
            "cloud.platform": "gcp_cloud_run",
        }
    )
    provider = TracerProvider(resource=resource)

    if (
        HAS_GCP_TRACE_EXPORTER
        and CloudTraceSpanExporter is not None
        and settings.enable_cloud_trace
        and settings.gcp_project != "argon-learn-local"
    ):
        try:
            gcp_exporter = CloudTraceSpanExporter(project_id=settings.gcp_project)
            provider.add_span_processor(BatchSpanProcessor(gcp_exporter))
        except Exception:
            pass

    trace.set_tracer_provider(provider)
    return trace.get_tracer("argon-learn.multi-agent")


_TRACER = _setup_tracer_provider()


def get_tracer() -> Any:
    """Returns the configured OpenTelemetry tracer instance."""
    return _TRACER


class SpanContextHandle:
    """Lightweight handle exposing span metadata and attribute recording."""

    def __init__(
        self,
        name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.attributes: Dict[str, Any] = dict(attributes or {})
        self.status: str = "OK"
        self.start_time_ns: int = time.time_ns()
        self.end_time_ns: int = self.start_time_ns
        self.duration_ms: float = 0.0

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def record_exception(self, exc: BaseException) -> None:
        self.status = "ERROR"
        self.attributes["error.type"] = type(exc).__name__
        self.attributes["error.message"] = str(exc)


@contextmanager
def trace_agent_span(
    span_name: str,
    *,
    agent_name: str = "ArgonCoordinator",
    model_id: str = "gemini-2.5-pro",
    stage: str = "orchestration",
    attributes: Optional[Dict[str, Any]] = None,
) -> Iterator[SpanContextHandle]:
    """Context manager creating a linked OpenTelemetry span and recording metrics.

    Args:
        span_name: Operation name (e.g. 'TriageIngestAgent.parse_breakthrough').
        agent_name: Name of the active ADK agent or tool.
        model_id: Gemini model identifier assigned to this span.
        stage: Pipeline stage ('triage', 'pedagogy', 'visualizer', 'tool', 'memory').
        attributes: Additional key-value span attributes.

    Yields:
        SpanContextHandle with `trace_id`, `span_id`, and `set_attribute()`.
    """
    parent_trace_id = _current_trace_id.get()
    parent_span_id = _current_span_id.get() or None

    trace_id = parent_trace_id or uuid.uuid4().hex
    span_id = uuid.uuid4().hex[:16]

    token_trace = _current_trace_id.set(trace_id)
    token_span = _current_span_id.set(span_id)

    merged_attrs: Dict[str, Any] = {
        "gen_ai.system": "google_adk",
        "gen_ai.agent.name": agent_name,
        "gen_ai.request.model": model_id,
        "argon.pipeline.stage": stage,
        **(attributes or {}),
    }

    handle = SpanContextHandle(
        name=span_name,
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        attributes=merged_attrs,
    )

    otel_cm = (
        _TRACER.start_as_current_span(span_name)
        if _TRACER is not None
        else None
    )
    otel_span = otel_cm.__enter__() if otel_cm is not None else None

    start_perf = time.perf_counter()
    try:
        yield handle
    except Exception as exc:
        handle.record_exception(exc)
        if otel_span is not None:
            try:
                otel_span.record_exception(exc)
            except Exception:
                pass
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        handle.end_time_ns = time.time_ns()
        handle.duration_ms = round(elapsed_ms, 2)

        if otel_span is not None:
            try:
                for k, v in handle.attributes.items():
                    otel_span.set_attribute(k, str(v))
            except Exception:
                pass
            if otel_cm is not None:
                otel_cm.__exit__(None, None, None)

        RECENT_SPANS.append(
            {
                "trace_id": handle.trace_id,
                "span_id": handle.span_id,
                "parent_span_id": handle.parent_span_id,
                "name": handle.name,
                "agent_name": agent_name,
                "model_id": model_id,
                "stage": stage,
                "status": handle.status,
                "duration_ms": handle.duration_ms,
                "start_time_ns": handle.start_time_ns,
                "end_time_ns": handle.end_time_ns,
                "attributes": handle.attributes,
            }
        )

        _current_span_id.reset(token_span)
        if not parent_trace_id:
            _current_trace_id.reset(token_trace)


def get_recent_traces(trace_id: Optional[str] = None, limit: int = 40) -> List[Dict[str, Any]]:
    """Retrieves recorded OpenTelemetry spans filtered by optional `trace_id`."""
    spans = list(RECENT_SPANS)
    if trace_id:
        spans = [s for s in spans if s["trace_id"] == trace_id]
    return spans[-limit:]
