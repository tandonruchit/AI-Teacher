"""TriageIngestAgent (`gemini-2.5-flash`) implemented with Google ADK.

Routes fast announcement ingestion, concept extraction, and novelty detection
to `gemini-2.5-flash` while enforcing the System Constitution and PII scrubbing.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from app.config import SYSTEM_CONSTITUTION, settings
from app.observability.logging import get_logger
from app.observability.pii_scrubber import scrub_pii
from app.observability.tracing import trace_agent_span
from app.tools.ingestion_tools import fetch_recent_ai_breakthroughs

try:
    from google.adk.agents import LlmAgent
    from google.adk.tools import FunctionTool
    HAS_GOOGLE_ADK = True
except ImportError:
    HAS_GOOGLE_ADK = False

    class FunctionTool:  # type: ignore[no-redef]
        """ADK-compatible FunctionTool wrapper when google-adk is not installed."""

        def __init__(self, func: Callable[..., Any]) -> None:
            self.func = func
            self.name = getattr(func, "__name__", "tool")
            self.description = (getattr(func, "__doc__", "") or "").strip()

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.func(*args, **kwargs)

    class LlmAgent:  # type: ignore[no-redef]
        """ADK-compatible LlmAgent descriptor when google-adk is not installed."""

        def __init__(
            self,
            *,
            name: str,
            model: str,
            description: str,
            instruction: str,
            tools: List[Any] | None = None,
            sub_agents: List[Any] | None = None,
            before_tool_callback: Callable[..., Any] | None = None,
            after_model_callback: Callable[..., Any] | None = None,
        ) -> None:
            self.name = name
            self.model = model
            self.description = description
            self.instruction = instruction
            self.tools = tools or []
            self.sub_agents = sub_agents or []
            self.before_tool_callback = before_tool_callback
            self.after_model_callback = after_model_callback


logger = get_logger("agents.triage")

TRIAGE_AGENT_INSTRUCTION = f"""{SYSTEM_CONSTITUTION}

## ACTIVE AGENT ROLE: TriageIngestAgent
- Assigned Model Tier: `{settings.triage_model}` (optimized for low-latency extraction).
- Primary Responsibility: Parse raw AI announcements, research paper abstracts, or
  chat queries using `fetch_recent_ai_breakthroughs`.
- Output Contract: Always identify the canonical `concept_id`, `concept_title`,
  `governing_equation`, and `prerequisites` for downstream pedagogical synthesis.
"""


def build_triage_adk_agent() -> LlmAgent:
    """Constructs the ADK `LlmAgent` definition for `TriageIngestAgent`."""
    return LlmAgent(
        name="TriageIngestAgent",
        model=settings.triage_model,
        description=(
            "Fast low-latency triage agent on gemini-2.5-flash that parses AI research "
            "headlines and extracts canonical concepts and prerequisites."
        ),
        instruction=TRIAGE_AGENT_INSTRUCTION,
        tools=[FunctionTool(fetch_recent_ai_breakthroughs)],
    )


class TriageIngestAgent:
    """Runtime executor for `TriageIngestAgent` backed by `gemini-2.5-flash`."""

    def __init__(self, model_id: str | None = None) -> None:
        self.name = "TriageIngestAgent"
        self.model_id = model_id or settings.triage_model
        self.adk_agent = build_triage_adk_agent()

    def run(
        self,
        raw_query: str,
        session_id: str = "default",
        difficulty_filter: str = "all",
    ) -> Dict[str, Any]:
        """Executes fast triage and concept extraction on the inbound user prompt."""
        sanitized_query = scrub_pii(raw_query)
        with trace_agent_span(
            "TriageIngestAgent.run",
            agent_name=self.name,
            model_id=self.model_id,
            stage="triage",
            attributes={"argon.session_id": session_id},
        ) as span:
            ingestion_result = fetch_recent_ai_breakthroughs(
                topic_query=sanitized_query,
                max_results=3,
                difficulty_filter=difficulty_filter,
                session_id=session_id,
            )

            if ingestion_result.get("status") == "error":
                span.set_attribute("argon.triage.status", "error")
                return {
                    "status": "error",
                    "agent_name": self.name,
                    "model_id": self.model_id,
                    "error": ingestion_result,
                }

            primary = ingestion_result["breakthroughs"][0]
            span.set_attribute("argon.concept_id", primary["concept_id"])
            return {
                "status": "success",
                "agent_name": self.name,
                "model_id": self.model_id,
                "hud_header": f"Level: Novice | Concept: {primary['concept_title'].split('(')[0].strip()}",
                "primary_concept": primary,
                "all_breakthroughs": ingestion_result["breakthroughs"],
            }
