"""ADK Multi-Agent Coordinator (`ArgonMultiAgentCoordinator`) & Strategic Model Router.

Implements the Google Agent Development Kit (ADK) Coordinator & Sequential multi-agent
pattern connecting:
1. `TriageIngestAgent` (`gemini-2.5-flash`) — fast headline ingestion & concept extraction.
2. `PedagogyCoordinatorAgent` (`gemini-2.5-pro` / `gemini-4-argon`) — ELI5-to-Hero synthesis
   and Firestore DAG updates.
3. `InteractiveVisualizerAgent` (`gemini-2.5-pro` / `gemini-4-argon`) — sandboxed HTML5/JS
   interactive simulation widget and AST-validated Manim 3D vector script generation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from app.agents.pedagogy import PedagogyCoordinatorAgent, build_pedagogy_adk_agent
from app.agents.triage import LlmAgent, TriageIngestAgent, build_triage_adk_agent
from app.agents.visualizer import (
    InteractiveVisualizerAgent,
    build_visualizer_adk_agent,
)
from app.config import SYSTEM_CONSTITUTION, settings
from app.memory.compaction import ConversationCompactor, TurnMessage
from app.memory.firestore_client import FirestoreSessionStore, get_firestore_store
from app.observability.logging import (
    get_logger,
    get_recent_audit_logs,
    log_agent_intent,
    log_tool_outcome,
)
from app.observability.pii_scrubber import PIIScrubbingMiddleware
from app.observability.tracing import get_recent_traces, trace_agent_span

try:
    from google.adk.agents import SequentialAgent
    HAS_SEQUENTIAL_AGENT = True
except ImportError:
    HAS_SEQUENTIAL_AGENT = False

    class SequentialAgent:  # type: ignore[no-redef]
        """ADK-compatible SequentialAgent coordinator fallback."""

        def __init__(
            self,
            *,
            name: str,
            description: str,
            sub_agents: List[Any],
        ) -> None:
            self.name = name
            self.description = description
            self.sub_agents = sub_agents


logger = get_logger("agents.coordinator")


class ModelRouter:
    """Strategic model router selecting between Flash, Pro, and Gemini 4 Argon."""

    @staticmethod
    def select_model(task_type: str, mastery_level: str = "Novice") -> Dict[str, str]:
        """Selects the optimal Gemini model tier based on task complexity and latency budget.

        Args:
            task_type: Pipeline stage ('triage', 'compaction', 'pedagogy', 'visualizer').
            mastery_level: Requested depth ('ELI5', 'Novice', 'Practitioner', 'Hero').

        Returns:
            Dict containing `model_id`, `tier`, and `routing_rationale`.
        """
        if task_type in {"triage", "compaction", "extraction"}:
            return {
                "model_id": settings.triage_model,
                "tier": "fast_flash",
                "routing_rationale": (
                    "Low-latency O(1) classification, headline parsing, and token compaction "
                    "routed to gemini-2.5-flash."
                ),
            }
        if mastery_level == "Hero":
            return {
                "model_id": settings.frontier_model,
                "tier": "frontier_argon",
                "routing_rationale": (
                    "Hero-mode mathematical derivations, 3D cosine-similarity vector manifolds, "
                    "and Manim AST synthesis routed to gemini-4-argon."
                ),
            }
        return {
            "model_id": settings.pedagogy_model,
            "tier": "reasoning_pro",
            "routing_rationale": (
                "Multi-level pedagogical decomposition and sandboxed HTML5/Canvas simulation "
                "synthesis routed to gemini-2.5-pro."
            ),
        }


def build_root_adk_coordinator() -> LlmAgent:
    """Constructs the root Google ADK Coordinator `LlmAgent` with registered sub-agents."""
    triage_agent = build_triage_adk_agent()
    pedagogy_agent = build_pedagogy_adk_agent()
    visualizer_agent = build_visualizer_adk_agent()

    sequential_pipeline = SequentialAgent(
        name="ArgonSequentialLearningPipeline",
        description="Sequential execution of Triage -> Pedagogy -> Interactive Visualizer.",
        sub_agents=[triage_agent, pedagogy_agent, visualizer_agent],
    )

    return LlmAgent(
        name="ArgonRootCoordinatorAgent",
        model=settings.pedagogy_model,
        description=(
            "Root ADK Multi-Agent Coordinator orchestrating TriageIngestAgent, "
            "PedagogyCoordinatorAgent, and InteractiveVisualizerAgent."
        ),
        instruction=SYSTEM_CONSTITUTION,
        sub_agents=[sequential_pipeline],
    )


class ArgonMultiAgentCoordinator:
    """End-to-end production orchestrator for the ArgonLearn ADK multi-agent system."""

    def __init__(
        self,
        store: Optional[FirestoreSessionStore] = None,
        compactor: Optional[ConversationCompactor] = None,
    ) -> None:
        self.store = store or get_firestore_store()
        self.compactor = compactor or ConversationCompactor()
        self.pii_middleware = PIIScrubbingMiddleware()
        self.router = ModelRouter()

        self.triage_agent = TriageIngestAgent()
        self.pedagogy_agent = PedagogyCoordinatorAgent()
        self.visualizer_agent = InteractiveVisualizerAgent()
        self.root_adk_agent = build_root_adk_coordinator()

    def process_learning_request(
        self,
        *,
        prompt: str,
        learner_id: str = "learner_demo_01",
        session_id: str = "session_demo_01",
        mastery_mode: str = "Novice",
        masking_ratio: float = 0.80,
        execute_manim_render: bool = False,
        publish_to_shared_repo: bool = False,
        human_approved: bool = False,
    ) -> Dict[str, Any]:
        """Executes a full multi-agent learning cycle from headline to interactive simulation.

        Pipeline steps:
        1. Inbound PII scrubbing (`PIIScrubbingMiddleware`).
        2. Session history retrieval from Firestore & sliding-window `ConversationCompactor`.
        3. Strategic model routing (`ModelRouter`).
        4. `TriageIngestAgent` (`gemini-2.5-flash`) -> `fetch_recent_ai_breakthroughs`.
        5. `PedagogyCoordinatorAgent` (`gemini-2.5-pro` / `gemini-4-argon`) ->
           `query_learner_knowledge_graph`, ELI5-to-Hero explanations, and async
           `update_learner_mastery_state`.
        6. `InteractiveVisualizerAgent` (`gemini-2.5-pro` / `gemini-4-argon`) ->
           `generate_interactive_sim_code` (HTML5/JS) + `compile_manim_animation` (Manim).
        7. Persistent session update and OpenTelemetry trace assembly.
        """
        start_time = time.perf_counter()

        # Step 1: Inbound PII Scrubbing
        sanitized_req, pii_report = self.pii_middleware.sanitize_inbound_request(
            {"prompt": prompt, "learner_id": learner_id, "session_id": session_id}
        )
        clean_prompt = sanitized_req["prompt"]

        with trace_agent_span(
            "ArgonMultiAgentCoordinator.process_learning_request",
            agent_name="ArgonRootCoordinatorAgent",
            model_id=settings.pedagogy_model,
            stage="orchestration",
            attributes={
                "argon.learner_id": learner_id,
                "argon.session_id": session_id,
                "argon.mastery_mode": mastery_mode,
                "argon.pii.redactions": pii_report.redaction_count,
            },
        ) as root_span:
            intent_id = log_agent_intent(
                agent_name="ArgonRootCoordinatorAgent",
                model_id=settings.pedagogy_model,
                intended_action="Orchestrate multi-agent ELI5-to-Hero pipeline",
                target_tool="ArgonSequentialLearningPipeline",
                rationale="Transform raw AI announcement into pedagogy, HTML5 sim, Manim scene, and DAG update",
                input_parameters={
                    "prompt": clean_prompt,
                    "learner_id": learner_id,
                    "mastery_mode": mastery_mode,
                    "masking_ratio": masking_ratio,
                },
                session_id=session_id,
                trace_id=root_span.trace_id,
            )

            # Step 2: Load & compact session history
            session_state = self.store.get_session(session_id)
            raw_turns = list(session_state.get("turns", []))
            raw_turns.append(
                TurnMessage(
                    role="user",
                    content=clean_prompt,
                    mastery_level=mastery_mode,
                ).to_dict()
            )
            compacted_summary, retained_turns, compaction_metrics = (
                self.compactor.build_adk_context_window(
                    session_id=session_id,
                    turns=raw_turns,
                    existing_summary=str(session_state.get("compacted_summary", "")),
                )
            )

            # Step 3: Route & execute TriageIngestAgent (gemini-2.5-flash)
            triage_route = self.router.select_model("triage", mastery_mode)
            triage_output = self.triage_agent.run(
                raw_query=clean_prompt,
                session_id=session_id,
            )
            if triage_output.get("status") == "error":
                return triage_output

            primary_concept = triage_output["primary_concept"]

            # Step 4: Route & execute PedagogyCoordinatorAgent (gemini-2.5-pro / gemini-4-argon)
            pedagogy_route = self.router.select_model("pedagogy", mastery_mode)
            pedagogy_output = self.pedagogy_agent.run(
                learner_id=learner_id,
                concept=primary_concept,
                target_mode=mastery_mode,
                session_id=session_id,
            )

            # Step 5: Route & execute InteractiveVisualizerAgent (gemini-2.5-pro / gemini-4-argon)
            visualizer_route = self.router.select_model("visualizer", mastery_mode)
            visualizer_output = self.visualizer_agent.run(
                concept_id=primary_concept["concept_id"],
                concept_title=primary_concept["concept_title"],
                mastery_level=mastery_mode,
                initial_masking_ratio=masking_ratio,
                execute_render=execute_manim_render,
                publish_to_shared_repo=publish_to_shared_repo,
                human_approved=human_approved,
                session_id=session_id,
            )

            # Step 6: Append assistant response turn & persist compacted session
            assistant_summary = (
                f"[{pedagogy_output['hud_banner']}] "
                f"ELI5: {pedagogy_output['explanations']['ELI5'][:140]}..."
            )
            retained_turns.append(
                TurnMessage(
                    role="assistant",
                    content=assistant_summary,
                    concept_id=primary_concept["concept_id"],
                    mastery_level=mastery_mode,
                ).to_dict()
            )
            self.store.save_session(
                session_id,
                {
                    "turns": retained_turns,
                    "compacted_summary": compacted_summary,
                    "compaction_count": int(session_state.get("compaction_count", 0))
                    + (1 if compaction_metrics.was_compacted else 0),
                },
            )

            total_latency_ms = (time.perf_counter() - start_time) * 1000.0
            log_tool_outcome(
                intent_id=intent_id,
                agent_name="ArgonRootCoordinatorAgent",
                tool_name="ArgonSequentialLearningPipeline",
                status="SUCCESS",
                latency_ms=total_latency_ms,
                outcome_summary=(
                    f"Completed full ELI5-to-Hero synthesis for '{primary_concept['concept_id']}' "
                    f"at tier '{mastery_mode}'."
                ),
                output_metadata={
                    "concept_id": primary_concept["concept_id"],
                    "mastery_mode": mastery_mode,
                    "compacted": compaction_metrics.was_compacted,
                },
                session_id=session_id,
                trace_id=root_span.trace_id,
            )

        trace_spans = get_recent_traces(trace_id=root_span.trace_id)

        return {
            "status": "success",
            "trace_id": root_span.trace_id,
            "session_id": session_id,
            "learner_id": learner_id,
            "latency_ms": round(total_latency_ms, 2),
            "pii_sanitization": {
                "sanitized_prompt": clean_prompt,
                "redaction_count": pii_report.redaction_count,
                "matched_categories": pii_report.matched_categories,
            },
            "model_routing": {
                "triage": triage_route,
                "pedagogy": pedagogy_route,
                "visualizer": visualizer_route,
            },
            "compaction_telemetry": {
                "was_compacted": compaction_metrics.was_compacted,
                "original_turn_count": compaction_metrics.original_turn_count,
                "retained_turn_count": compaction_metrics.retained_turn_count,
                "original_tokens": compaction_metrics.original_tokens,
                "compacted_tokens": compaction_metrics.compacted_tokens,
                "compacted_summary": compacted_summary,
                "context_cache_key": compaction_metrics.context_cache_key,
            },
            "triage": triage_output,
            "pedagogy": pedagogy_output,
            "visualizer": visualizer_output,
            "observability": {
                "trace_spans": trace_spans,
                "recent_audit_logs": get_recent_audit_logs(limit=12),
                "firestore_backend": self.store.backend_mode,
            },
        }


_COORDINATOR_SINGLETON = ArgonMultiAgentCoordinator()


def get_coordinator() -> ArgonMultiAgentCoordinator:
    """Returns the singleton `ArgonMultiAgentCoordinator` instance."""
    return _COORDINATOR_SINGLETON
