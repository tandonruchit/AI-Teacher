"""Pydantic-validated ADK Tools with Guided Error Handling and Guardrail Policies."""

from app.tools.ingestion_tools import (
    BreakthroughArticleSchema,
    FetchBreakthroughsInput,
    FetchBreakthroughsOutput,
    ToolErrorPayload,
    fetch_recent_ai_breakthroughs,
)
from app.tools.knowledge_tools import (
    QueryKnowledgeGraphInput,
    QueryKnowledgeGraphOutput,
    UpdateMasteryStateInput,
    UpdateMasteryStateOutput,
    query_learner_knowledge_graph,
    update_learner_mastery_state,
)
from app.tools.simulation_tools import (
    CompileManimInput,
    CompileManimOutput,
    GenerateInteractiveSimInput,
    GenerateInteractiveSimOutput,
    HumanInTheLoopGate,
    SimulationSafetyPolicyPlugin,
    compile_manim_animation,
    generate_interactive_sim_code,
)

__all__ = [
    "BreakthroughArticleSchema",
    "CompileManimInput",
    "CompileManimOutput",
    "FetchBreakthroughsInput",
    "FetchBreakthroughsOutput",
    "GenerateInteractiveSimInput",
    "GenerateInteractiveSimOutput",
    "HumanInTheLoopGate",
    "QueryKnowledgeGraphInput",
    "QueryKnowledgeGraphOutput",
    "SimulationSafetyPolicyPlugin",
    "ToolErrorPayload",
    "UpdateMasteryStateInput",
    "UpdateMasteryStateOutput",
    "compile_manim_animation",
    "fetch_recent_ai_breakthroughs",
    "generate_interactive_sim_code",
    "query_learner_knowledge_graph",
    "update_learner_mastery_state",
]
