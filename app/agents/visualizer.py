"""InteractiveVisualizerAgent (`gemini-2.5-pro` / `gemini-4-argon`) using Google ADK.

Synthesizes standalone, sandboxed HTML5/Canvas/Three.js interactive widgets and
AST-verified Python Manim animations while enforcing `SimulationSafetyPolicyPlugin`
and `HumanInTheLoopGate`.
"""

from __future__ import annotations

from typing import Any, Dict

from app.agents.triage import FunctionTool, LlmAgent
from app.config import SYSTEM_CONSTITUTION, settings
from app.observability.logging import get_logger
from app.observability.tracing import trace_agent_span
from app.tools.simulation_tools import (
    compile_manim_animation,
    generate_interactive_sim_code,
)

logger = get_logger("agents.visualizer")

VISUALIZER_AGENT_INSTRUCTION = f"""{SYSTEM_CONSTITUTION}

## ACTIVE AGENT ROLE: InteractiveVisualizerAgent
- Assigned Model Tier: `{settings.visualizer_model}` / `{settings.frontier_model}`.
- Primary Responsibility:
  1. Call `generate_interactive_sim_code` to build a live interactive HTML5/Canvas
     simulation widget (e.g., Token Grid with 'Masking Ratio: 80% -> 20%' slider
     and dynamic attention heatmaps).
  2. Call `compile_manim_animation` to synthesize a 3D cosine-similarity vector
     clustering Manim `ThreeDScene` script.
- Enforce `SimulationSafetyPolicyPlugin` and `HumanInTheLoopGate` on all outputs.
"""


def build_visualizer_adk_agent() -> LlmAgent:
    """Constructs the ADK `LlmAgent` definition for `InteractiveVisualizerAgent`."""
    return LlmAgent(
        name="InteractiveVisualizerAgent",
        model=settings.visualizer_model,
        description=(
            "Interactive simulation and mathematical animation agent on gemini-2.5-pro / "
            "gemini-4-argon that produces sandboxed HTML5/JS widgets and Manim 3D scenes."
        ),
        instruction=VISUALIZER_AGENT_INSTRUCTION,
        tools=[
            FunctionTool(generate_interactive_sim_code),
            FunctionTool(compile_manim_animation),
        ],
    )


class InteractiveVisualizerAgent:
    """Runtime executor for `InteractiveVisualizerAgent`."""

    def __init__(self, model_id: str | None = None) -> None:
        self.name = "InteractiveVisualizerAgent"
        self.model_id = model_id or settings.visualizer_model
        self.adk_agent = build_visualizer_adk_agent()

    def run(
        self,
        *,
        concept_id: str,
        concept_title: str,
        mastery_level: str = "Novice",
        initial_masking_ratio: float = 0.80,
        execute_render: bool = False,
        publish_to_shared_repo: bool = False,
        human_approved: bool = False,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """Generates both the interactive HTML5 toybox widget and Manim 3D scene."""
        active_model = (
            settings.frontier_model
            if mastery_level == "Hero"
            else self.model_id
        )

        with trace_agent_span(
            "InteractiveVisualizerAgent.run",
            agent_name=self.name,
            model_id=active_model,
            stage="visualizer",
            attributes={
                "argon.concept_id": concept_id,
                "argon.mastery_level": mastery_level,
            },
        ):
            sim_result = generate_interactive_sim_code(
                concept_id=concept_id,
                concept_title=concept_title,
                mastery_level=mastery_level,
                initial_masking_ratio=initial_masking_ratio,
                session_id=session_id,
            )

            manim_result = compile_manim_animation(
                concept_id=concept_id,
                scene_class_name="CosineSimilarityClusteringScene",
                mastery_level="Hero" if mastery_level == "Hero" else "Practitioner",
                execute_render=execute_render,
                publish_to_shared_repo=publish_to_shared_repo,
                human_approved=human_approved,
                session_id=session_id,
            )

            return {
                "status": "success",
                "agent_name": self.name,
                "model_id": active_model,
                "interactive_simulation": sim_result,
                "manim_animation": manim_result,
            }
