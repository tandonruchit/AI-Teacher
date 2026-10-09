"""Learner Knowledge Graph Tools (`query_learner_knowledge_graph`, `update_learner_mastery_state`).

Provides strict Pydantic v2 input/output schemas, Google-style docstrings,
Guided Error Handling with actionable `remediation_instruction` payloads, and
asynchronous Firestore DAG persistence.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from app.config import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)
from app.memory.knowledge_graph import get_knowledge_graph
from app.observability.logging import log_agent_intent, log_tool_outcome
from app.observability.pii_scrubber import scrub_pii
from app.observability.tracing import trace_agent_span
from app.tools.ingestion_tools import ToolErrorPayload


class QueryKnowledgeGraphInput(BaseModel):
    """Strict input schema for `query_learner_knowledge_graph`."""

    model_config = ConfigDict(extra="forbid", strict=True)

    learner_id: str = Field(
        ...,
        min_length=2,
        max_length=120,
        pattern=r"^[a-zA-Z0-9_\-\.]+$",
        description="Unique learner identifier querying the Firestore knowledge graph DAG.",
    )
    target_concept_id: Optional[str] = Field(
        default=None,
        description="Optional concept_id to inspect prerequisite readiness and mastery tier.",
    )


class QueryKnowledgeGraphOutput(BaseModel):
    """Strict output schema for `query_learner_knowledge_graph`."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: str = Field(default="success", description="Tool execution status.")
    learner_id: str = Field(..., description="Sanitized learner identifier.")
    is_valid_dag: bool = Field(
        ..., description="True if the learner knowledge graph has zero cycles."
    )
    recommended_starting_tier: str = Field(
        ..., description="Recommended pedagogical starting tier ('Novice', 'Practitioner', 'Hero')."
    )
    prerequisite_readiness_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized readiness score across prerequisites."
    )
    topological_order: List[str] = Field(
        ..., description="Topological sort of mastered and active concept nodes."
    )
    nodes: Dict[str, Any] = Field(
        ..., description="Map of concept_id to ConceptNode metadata and 3D embeddings."
    )
    edges: List[Dict[str, Any]] = Field(
        ..., description="Directed prerequisite edges connecting DAG vertices."
    )


class UpdateMasteryStateInput(BaseModel):
    """Strict input schema for `update_learner_mastery_state`."""

    model_config = ConfigDict(extra="forbid", strict=True)

    learner_id: str = Field(
        ...,
        min_length=2,
        max_length=120,
        pattern=r"^[a-zA-Z0-9_\-\.]+$",
        description="Unique learner identifier whose mastery DAG is being updated.",
    )
    concept_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
        pattern=r"^[a-z0-9_]+$",
        description="Snake-case canonical identifier of the concept being mastered.",
    )
    concept_title: str = Field(
        ...,
        min_length=3,
        max_length=160,
        description="Human-readable concept title displayed in the skill-tree HUD.",
    )
    mastery_delta: float = Field(
        default=0.25,
        ge=-0.5,
        le=1.0,
        description="Incremental mastery change in [-0.5, 1.0] applied to the node.",
    )
    absolute_mastery: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional explicit mastery score override in [0.0, 1.0].",
    )
    prerequisites: List[str] = Field(
        default_factory=lambda: ["transformer_architecture"],
        description="Prerequisite concept_ids connected via directed edges in the DAG.",
    )
    key_equation: str = Field(
        default="s(u, v) = (u · v) / (||u||_2 ||v||_2)",
        max_length=300,
        description="Governing mathematical equation associated with the mastered concept.",
    )

    @field_validator("concept_title")
    def sanitize_title(cls, value: str) -> str:
        return scrub_pii(value.strip())


class UpdateMasteryStateOutput(BaseModel):
    """Strict output schema for `update_learner_mastery_state`."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: str = Field(default="success", description="Tool execution status.")
    learner_id: str = Field(..., description="Learner identifier.")
    updated_node: Dict[str, Any] = Field(
        ..., description="Updated ConceptNode including new mastery_score and mastery_tier."
    )
    hud_badge: str = Field(
        ..., description="Skill-tree HUD notification banner (e.g. '[+1 Mastered: Discrete Diffusion]')."
    )
    connected_prerequisites: List[str] = Field(
        ..., description="Prerequisite nodes linked to this concept in the skill tree."
    )
    async_persistence_scheduled: bool = Field(
        ..., description="True when background Firestore & embedding consolidation was scheduled."
    )
    dag_snapshot: Dict[str, Any] = Field(
        ..., description="Updated Directed Acyclic Graph state."
    )


def query_learner_knowledge_graph(
    learner_id: str,
    target_concept_id: Optional[str] = None,
    session_id: str = "default",
) -> Dict[str, Any]:
    """Queries the learner's persistent Directed Acyclic Graph (DAG) of AI concepts in Firestore.

    Retrieves all mastered and in-progress concept vertices, verifies topological
    acyclicity, computes prerequisite readiness for `target_concept_id`, and recommends
    whether the pedagogy agent should begin at `Novice`, `Practitioner`, or `Hero` depth.

    Args:
        learner_id: Alphanumeric learner identifier (`^[a-zA-Z0-9_\\-\\.]+$`, length 2..120).
        target_concept_id: Optional canonical snake_case concept ID to evaluate readiness for.
        session_id: Active conversational session ID for trace correlation.

    Returns:
        Dict[str, Any]: Serialized `QueryKnowledgeGraphOutput` with nodes, edges, and
        topological ordering, or `ToolErrorPayload` with `remediation_instruction` on error.
    """
    start_time = time.perf_counter()
    intent_id = log_agent_intent(
        agent_name="PedagogyCoordinatorAgent",
        model_id="gemini-2.5-pro",
        intended_action="Query learner knowledge graph DAG for prerequisite mastery and readiness",
        target_tool="query_learner_knowledge_graph",
        rationale="Calibrate explanation depth (ELI5 vs Hero) to learner's prior Transformer mastery",
        input_parameters={
            "learner_id": learner_id,
            "target_concept_id": target_concept_id,
        },
        session_id=session_id,
    )

    with trace_agent_span(
        "tool.query_learner_knowledge_graph",
        agent_name="PedagogyCoordinatorAgent",
        model_id="gemini-2.5-pro",
        stage="tool",
        attributes={"argon.tool.name": "query_learner_knowledge_graph"},
    ) as span:
        try:
            validated = QueryKnowledgeGraphInput(
                learner_id=learner_id,
                target_concept_id=target_concept_id,
            )
            kg = get_knowledge_graph()
            graph_data = kg.query_graph(
                learner_id=validated.learner_id,
                target_concept_id=validated.target_concept_id,
            )

            output_model = QueryKnowledgeGraphOutput(
                status="success",
                learner_id=validated.learner_id,
                is_valid_dag=bool(graph_data["is_valid_dag"]),
                recommended_starting_tier=str(graph_data["recommended_starting_tier"]),
                prerequisite_readiness_score=float(
                    graph_data["prerequisite_readiness_score"]
                ),
                topological_order=list(graph_data["topological_order"]),
                nodes=dict(graph_data["nodes"]),
                edges=list(graph_data["edges"]),
            )
            result = output_model.model_dump()

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            log_tool_outcome(
                intent_id=intent_id,
                agent_name="PedagogyCoordinatorAgent",
                tool_name="query_learner_knowledge_graph",
                status="SUCCESS",
                latency_ms=latency_ms,
                outcome_summary=(
                    f"Retrieved {len(result['nodes'])} DAG nodes for '{validated.learner_id}' "
                    f"(recommended tier: {result['recommended_starting_tier']})."
                ),
                output_metadata={
                    "node_count": len(result["nodes"]),
                    "recommended_tier": result["recommended_starting_tier"],
                },
                session_id=session_id,
                trace_id=span.trace_id,
            )
            return result

        except ValidationError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            err = ToolErrorPayload(
                tool_name="query_learner_knowledge_graph",
                error_type="ValidationError",
                raw_message=str(exc),
                remediation_instruction=(
                    "Provide a valid alphanumeric `learner_id` (2..120 chars matching "
                    "`^[a-zA-Z0-9_\\-\\.]+$`, e.g., 'learner_demo_01')."
                ),
            ).model_dump()
            log_tool_outcome(
                intent_id=intent_id,
                agent_name="PedagogyCoordinatorAgent",
                tool_name="query_learner_knowledge_graph",
                status="RECOVERABLE_ERROR",
                latency_ms=latency_ms,
                outcome_summary="Invalid learner_id passed to query_learner_knowledge_graph",
                error_payload=err,
                session_id=session_id,
                trace_id=span.trace_id,
            )
            return err

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            err = ToolErrorPayload(
                tool_name="query_learner_knowledge_graph",
                error_type=type(exc).__name__,
                raw_message=str(exc),
                remediation_instruction=(
                    "Verify Firestore connectivity or retry with `learner_id='learner_default'`."
                ),
            ).model_dump()
            log_tool_outcome(
                intent_id=intent_id,
                agent_name="PedagogyCoordinatorAgent",
                tool_name="query_learner_knowledge_graph",
                status="RECOVERABLE_ERROR",
                latency_ms=latency_ms,
                outcome_summary=f"Knowledge graph query failed: {type(exc).__name__}",
                error_payload=err,
                session_id=session_id,
                trace_id=span.trace_id,
            )
            return err


def update_learner_mastery_state(
    learner_id: str,
    concept_id: str,
    concept_title: str,
    mastery_delta: float = 0.25,
    absolute_mastery: Optional[float] = None,
    prerequisites: Optional[List[str]] = None,
    key_equation: str = "s(u, v) = (u · v) / (||u||_2 ||v||_2)",
    session_id: str = "default",
) -> Dict[str, Any]:
    """Updates a learner's concept mastery node in the Firestore DAG asynchronously.

    Validates DAG acyclicity (rejecting self-dependencies or cycles), computes updated
    3D cosine-similarity embeddings, schedules non-blocking background persistence via
    `asyncio.create_task`, and emits a skill-tree HUD badge connecting the newly mastered
    concept to prior foundational nodes (such as `transformer_architecture`).

    Args:
        learner_id: Alphanumeric learner identifier (`^[a-zA-Z0-9_\\-\\.]+$`).
        concept_id: Snake-case concept identifier (`^[a-z0-9_]+$`, length 3..100).
        concept_title: Display title for the concept node (length 3..160).
        mastery_delta: Incremental mastery score adjustment in [-0.5, 1.0]. Defaults to 0.25.
        absolute_mastery: Optional explicit mastery score override in [0.0, 1.0].
        prerequisites: Prerequisite concept IDs to connect with directed DAG edges.
        key_equation: Governing equation stored on the concept vertex.
        session_id: Active conversational session ID for trace correlation.

    Returns:
        Dict[str, Any]: Serialized `UpdateMasteryStateOutput` containing `hud_badge`,
        `updated_node`, and `dag_snapshot`, or `ToolErrorPayload` on cycle/validation error.
    """
    start_time = time.perf_counter()
    intent_id = log_agent_intent(
        agent_name="PedagogyCoordinatorAgent",
        model_id="gemini-2.5-pro",
        intended_action=f"Update learner mastery DAG for concept '{concept_id}'",
        target_tool="update_learner_mastery_state",
        rationale="Persist learner progression and connect new concept node to Transformer prerequisites",
        input_parameters={
            "learner_id": learner_id,
            "concept_id": concept_id,
            "concept_title": concept_title,
            "mastery_delta": mastery_delta,
            "prerequisites": prerequisites,
        },
        session_id=session_id,
    )

    with trace_agent_span(
        "tool.update_learner_mastery_state",
        agent_name="PedagogyCoordinatorAgent",
        model_id="gemini-2.5-pro",
        stage="tool",
        attributes={
            "argon.tool.name": "update_learner_mastery_state",
            "argon.concept_id": concept_id,
        },
    ) as span:
        try:
            validated = UpdateMasteryStateInput(
                learner_id=learner_id,
                concept_id=concept_id,
                concept_title=concept_title,
                mastery_delta=mastery_delta,
                absolute_mastery=absolute_mastery,
                prerequisites=prerequisites
                if prerequisites is not None
                else ["transformer_architecture"],
                key_equation=key_equation,
            )

            kg = get_knowledge_graph()
            mutation = kg.upsert_concept_mastery(
                learner_id=validated.learner_id,
                concept_id=validated.concept_id,
                title=validated.concept_title,
                mastery_delta=validated.mastery_delta,
                absolute_mastery=validated.absolute_mastery,
                prerequisites=validated.prerequisites,
                key_equation=validated.key_equation,
            )

            updated_node = mutation["updated_node"]
            short_title = validated.concept_title.split("(")[0].strip()
            hud_badge = f"[+1 Mastered: {short_title}]"

            output_model = UpdateMasteryStateOutput(
                status="success",
                learner_id=validated.learner_id,
                updated_node=updated_node,
                hud_badge=hud_badge,
                connected_prerequisites=list(updated_node.get("prerequisites", [])),
                async_persistence_scheduled=bool(
                    mutation["async_persistence_scheduled"]
                ),
                dag_snapshot=mutation["dag"],
            )
            result = output_model.model_dump()

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            log_tool_outcome(
                intent_id=intent_id,
                agent_name="PedagogyCoordinatorAgent",
                tool_name="update_learner_mastery_state",
                status="SUCCESS",
                latency_ms=latency_ms,
                outcome_summary=(
                    f"Updated '{validated.concept_id}' to score={updated_node['mastery_score']} "
                    f"({updated_node['mastery_tier']}) with async Firestore consolidation."
                ),
                output_metadata={
                    "concept_id": validated.concept_id,
                    "mastery_score": updated_node["mastery_score"],
                    "hud_badge": hud_badge,
                },
                session_id=session_id,
                trace_id=span.trace_id,
            )
            return result

        except (ValidationError, ValueError) as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            err = ToolErrorPayload(
                tool_name="update_learner_mastery_state",
                error_type=type(exc).__name__,
                raw_message=str(exc),
                remediation_instruction=(
                    "Ensure `concept_id` is lowercase snake_case (`^[a-z0-9_]+$`), "
                    "`mastery_delta` is within [-0.5, 1.0], and `prerequisites` does not "
                    "contain `concept_id` or create a dependency cycle in the DAG."
                ),
            ).model_dump()
            log_tool_outcome(
                intent_id=intent_id,
                agent_name="PedagogyCoordinatorAgent",
                tool_name="update_learner_mastery_state",
                status="RECOVERABLE_ERROR",
                latency_ms=latency_ms,
                outcome_summary=f"DAG mastery update rejected: {exc}",
                error_payload=err,
                session_id=session_id,
                trace_id=span.trace_id,
            )
            return err
