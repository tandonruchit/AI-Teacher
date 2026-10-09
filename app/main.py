"""FastAPI Backend & ASGI Entrypoint for ArgonLearn (`argon-learn`).

Exposes REST endpoints for the ADK multi-agent learning pipeline, Firestore
Knowledge Graph DAG queries, Human-in-the-Loop (HITL) execution approval,
OpenTelemetry + structlog observability telemetry, and serves the interactive
HTML5/Canvas/Manim frontend.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from app.agents.coordinator import get_coordinator
from app.config import BaseModel, ConfigDict, Field, settings
from app.memory.firestore_client import get_firestore_store
from app.observability.logging import (
    get_logger,
    get_recent_audit_logs,
)
from app.observability.pii_scrubber import PIIScrubbingMiddleware
from app.observability.tracing import get_recent_traces
from app.tools.knowledge_tools import (
    query_learner_knowledge_graph,
    update_learner_mastery_state,
)
from app.tools.simulation_tools import compile_manim_animation

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    HAS_OTEL_FASTAPI = True
except ImportError:
    FastAPIInstrumentor = None  # type: ignore[assignment]
    HAS_OTEL_FASTAPI = False


logger = get_logger("app.main")
STATIC_DIR = Path(__file__).resolve().parent / "static"


class LearnRequestSchema(BaseModel):
    """Strict Pydantic v2 request schema for `POST /api/learn`."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="AI announcement headline, paper snippet, or question.",
    )
    learner_id: str = Field(
        default="learner_demo_01",
        min_length=2,
        max_length=120,
        description="Persistent learner identifier in Firestore.",
    )
    session_id: str = Field(
        default="session_demo_01",
        min_length=2,
        max_length=120,
        description="Conversational session identifier.",
    )
    mastery_mode: str = Field(
        default="Novice",
        pattern=r"^(ELI5|Novice|Practitioner|Hero)$",
        description="Target mastery explanation tier.",
    )
    masking_ratio: float = Field(
        default=0.80,
        ge=0.05,
        le=0.95,
        description="Initial token masking ratio in [0.05, 0.95].",
    )
    execute_manim_render: bool = Field(
        default=False,
        description="Whether to request high-stakes Manim compilation.",
    )
    publish_to_shared_repo: bool = Field(
        default=False,
        description="Whether to publish generated artifacts to team repository.",
    )
    human_approved: bool = Field(
        default=False,
        description="Explicit Human-in-the-Loop confirmation flag.",
    )


class HITLApprovalRequestSchema(BaseModel):
    """Strict Pydantic v2 schema for `POST /api/hitl/approve`."""

    model_config = ConfigDict(extra="forbid", strict=True)

    concept_id: str = Field(
        default="discrete_parallel_diffusion",
        min_length=3,
        max_length=100,
        description="Concept identifier for the Manim scene.",
    )
    scene_class_name: str = Field(
        default="CosineSimilarityClusteringScene",
        pattern=r"^[A-Z][A-Za-z0-9_]*Scene$",
        description="PascalCase Manim Scene class name.",
    )
    execute_render: bool = Field(
        default=True,
        description="Execute high-stakes Manim render.",
    )
    publish_to_shared_repo: bool = Field(
        default=True,
        description="Publish verified Manim scene to team repository.",
    )
    human_approved: bool = Field(
        default=True,
        description="Explicit human confirmation boolean.",
    )
    session_id: str = Field(
        default="session_demo_01",
        description="Session identifier for audit logging.",
    )


def handle_healthz() -> Dict[str, Any]:
    """Returns Cloud Run readiness/liveness and subsystem telemetry."""
    store = get_firestore_store()
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": "1.0.0",
        "environment": settings.environment,
        "gcp_project": settings.gcp_project,
        "gcp_region": settings.gcp_region,
        "firestore_backend": store.backend_mode,
        "async_memory_ops_completed": store.async_ops_completed,
        "models": {
            "triage": settings.triage_model,
            "pedagogy": settings.pedagogy_model,
            "visualizer": settings.visualizer_model,
            "frontier_hero": settings.frontier_model,
        },
    }


def handle_learn_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and executes a full ADK multi-agent learning cycle."""
    req = LearnRequestSchema(
        prompt=str(payload.get("prompt", "Meta just released Muse: Masked Generative Video Transformer.")),
        learner_id=str(payload.get("learner_id", "learner_demo_01")),
        session_id=str(payload.get("session_id", "session_demo_01")),
        mastery_mode=str(payload.get("mastery_mode", "Novice")),
        masking_ratio=float(payload.get("masking_ratio", 0.80)),
        execute_manim_render=bool(payload.get("execute_manim_render", False)),
        publish_to_shared_repo=bool(payload.get("publish_to_shared_repo", False)),
        human_approved=bool(payload.get("human_approved", False)),
    )
    coordinator = get_coordinator()
    return coordinator.process_learning_request(
        prompt=req.prompt,
        learner_id=req.learner_id,
        session_id=req.session_id,
        mastery_mode=req.mastery_mode,
        masking_ratio=req.masking_ratio,
        execute_manim_render=req.execute_manim_render,
        publish_to_shared_repo=req.publish_to_shared_repo,
        human_approved=req.human_approved,
    )


def handle_hitl_approval(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Processes explicit Human-in-the-Loop approval for Manim execution."""
    req = HITLApprovalRequestSchema(
        concept_id=str(payload.get("concept_id", "discrete_parallel_diffusion")),
        scene_class_name=str(
            payload.get("scene_class_name", "CosineSimilarityClusteringScene")
        ),
        execute_render=bool(payload.get("execute_render", True)),
        publish_to_shared_repo=bool(payload.get("publish_to_shared_repo", True)),
        human_approved=bool(payload.get("human_approved", True)),
        session_id=str(payload.get("session_id", "session_demo_01")),
    )
    return compile_manim_animation(
        concept_id=req.concept_id,
        scene_class_name=req.scene_class_name,
        mastery_level="Hero",
        execute_render=req.execute_render,
        publish_to_shared_repo=req.publish_to_shared_repo,
        human_approved=req.human_approved,
        session_id=req.session_id,
    )


def handle_observability_telemetry(trace_id: Optional[str] = None) -> Dict[str, Any]:
    """Returns live OpenTelemetry spans, structlog Intent-vs-Outcome logs, and Cloud Run status."""
    store = get_firestore_store()
    return {
        "status": "success",
        "cloud_run_metadata": {
            "service_name": settings.app_name,
            "revision": "argon-learn-00042-x9k",
            "region": settings.gcp_region,
            "project_id": settings.gcp_project,
            "service_account": f"{settings.app_name}-runtime-sa@{settings.gcp_project}.iam.gserviceaccount.com",
            "firestore_backend": store.backend_mode,
            "secret_manager_key_id": settings.gemini_api_key_secret_id,
            "async_memory_tasks_completed": store.async_ops_completed,
        },
        "trace_spans": get_recent_traces(trace_id=trace_id, limit=30),
        "audit_logs": get_recent_audit_logs(limit=25),
    }


if HAS_FASTAPI:
    app = FastAPI(
        title="ArgonLearn — ELI5-to-Hero Multi-Agent AI Visualizer",
        description=(
            "Production multi-agent AI application built with Google Agent Development Kit "
            "(ADK-Python), Firestore DAG memory, OpenTelemetry tracing, and Cloud Run."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(PIIScrubbingMiddleware)

    if HAS_OTEL_FASTAPI and FastAPIInstrumentor is not None:
        try:
            FastAPIInstrumentor.instrument_app(app)
        except Exception:
            pass

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_index() -> Any:
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return HTMLResponse("<h1>ArgonLearn UI</h1>")

    @app.get("/healthz")
    async def healthz_endpoint() -> Dict[str, Any]:
        return handle_healthz()

    @app.post("/api/learn")
    async def learn_endpoint(request: Request) -> JSONResponse:
        body = await request.json()
        result = handle_learn_request(body)
        return JSONResponse(content=result)

    @app.get("/api/knowledge-graph/{learner_id}")
    async def knowledge_graph_endpoint(
        learner_id: str, target_concept_id: Optional[str] = None
    ) -> JSONResponse:
        result = query_learner_knowledge_graph(
            learner_id=learner_id,
            target_concept_id=target_concept_id,
        )
        return JSONResponse(content=result)

    @app.post("/api/knowledge-graph/{learner_id}/mastery")
    async def update_mastery_endpoint(
        learner_id: str, request: Request
    ) -> JSONResponse:
        body = await request.json()
        result = update_learner_mastery_state(
            learner_id=learner_id,
            concept_id=str(body.get("concept_id", "discrete_parallel_diffusion")),
            concept_title=str(body.get("concept_title", "Discrete Parallel Diffusion")),
            mastery_delta=float(body.get("mastery_delta", 0.25)),
            absolute_mastery=body.get("absolute_mastery"),
            prerequisites=body.get("prerequisites"),
            key_equation=str(
                body.get("key_equation", "s(u, v) = (u · v) / (||u||_2 ||v||_2)")
            ),
        )
        return JSONResponse(content=result)

    @app.post("/api/hitl/approve")
    async def hitl_approve_endpoint(request: Request) -> JSONResponse:
        body = await request.json()
        result = handle_hitl_approval(body)
        return JSONResponse(content=result)

    @app.get("/api/observability/telemetry")
    async def observability_endpoint(trace_id: Optional[str] = None) -> JSONResponse:
        return JSONResponse(content=handle_observability_telemetry(trace_id=trace_id))

else:
    app = None  # type: ignore[assignment]


class _FallbackArgonHTTPRequestHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP server handler when uvicorn/fastapi are not installed locally."""

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def _serve_static_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._send_json(404, {"status": "error", "message": "File not found"})
            return
        mime_type, _ = mimetypes.guess_type(str(file_path))
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._serve_static_file(STATIC_DIR / "index.html")
            return
        if path.startswith("/static/"):
            rel_name = path.replace("/static/", "", 1)
            target = (STATIC_DIR / rel_name).resolve()
            if str(target).startswith(str(STATIC_DIR.resolve())):
                self._serve_static_file(target)
                return
        if path in {"/app.js", "/style.css"}:
            self._serve_static_file(STATIC_DIR / path.lstrip("/"))
            return
        if path == "/healthz":
            self._send_json(200, handle_healthz())
            return
        if path.startswith("/api/knowledge-graph/"):
            learner_id = path.split("/api/knowledge-graph/")[1].split("/")[0] or "learner_demo_01"
            target_cid = query.get("target_concept_id", [None])[0]
            self._send_json(
                200,
                query_learner_knowledge_graph(
                    learner_id=learner_id,
                    target_concept_id=target_cid,
                ),
            )
            return
        if path == "/api/observability/telemetry":
            trace_id = query.get("trace_id", [None])[0]
            self._send_json(200, handle_observability_telemetry(trace_id=trace_id))
            return

        self._send_json(404, {"status": "error", "path": path})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        content_len = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            body = json.loads(raw_body) if raw_body else {}
        except Exception:
            body = {}

        try:
            if path == "/api/learn":
                self._send_json(200, handle_learn_request(body))
                return
            if path == "/api/hitl/approve":
                self._send_json(200, handle_hitl_approval(body))
                return
            if path.startswith("/api/knowledge-graph/") and path.endswith("/mastery"):
                parts = path.split("/")
                learner_id = parts[3] if len(parts) > 3 else "learner_demo_01"
                res = update_learner_mastery_state(
                    learner_id=learner_id,
                    concept_id=str(body.get("concept_id", "discrete_parallel_diffusion")),
                    concept_title=str(body.get("concept_title", "Discrete Parallel Diffusion")),
                    mastery_delta=float(body.get("mastery_delta", 0.25)),
                    absolute_mastery=body.get("absolute_mastery"),
                    prerequisites=body.get("prerequisites"),
                    key_equation=str(
                        body.get("key_equation", "s(u, v) = (u · v) / (||u||_2 ||v||_2)")
                    ),
                )
                self._send_json(200, res)
                return
        except Exception as exc:
            self._send_json(
                400,
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "raw_message": str(exc),
                },
            )
            return

        self._send_json(404, {"status": "error", "path": path})


def run_server(host: str = "0.0.0.0", port: Optional[int] = None) -> None:
    """Starts the ArgonLearn HTTP server using Uvicorn (if available) or ThreadingHTTPServer."""
    resolved_port = port or int(os.environ.get("PORT", "8080"))
    try:
        import uvicorn  # type: ignore[import-not-found]
        if app is not None:
            uvicorn.run(app, host=host, port=resolved_port)
            return
    except ImportError:
        pass

    server = ThreadingHTTPServer((host, resolved_port), _FallbackArgonHTTPRequestHandler)
    logger.info(
        "argon_learn_server_started",
        host=host,
        port=resolved_port,
        mode="stdlib_http_adapter",
    )
    server.serve_forever()


if __name__ == "__main__":
    run_server()
