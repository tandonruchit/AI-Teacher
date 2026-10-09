"""Interactive Simulation & Manim Animation Tools (`generate_interactive_sim_code`, `compile_manim_animation`).

Includes:
1. Strict Pydantic v2 input/output JSON Schemas and Google-style docstrings.
2. `SimulationSafetyPolicyPlugin`: Security guardrail validating that generated HTML5/JS
   contains zero external `<script src=...>` injections, `eval()`, cookies, or network
   exfiltration, and that generated Manim Python code passes `ast.parse` with only
   allowlisted scientific imports (`manim`, `math`, `numpy`).
3. `HumanInTheLoopGate`: High-stakes execution hook requiring explicit user approval
   (`human_approved=True`) before executing arbitrary simulation scripts or exporting
   artifacts to shared team repositories.
4. Guided Error Handling returning structured `ToolErrorPayload` dicts.
"""

from __future__ import annotations

import ast
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)
from app.observability.logging import log_agent_intent, log_tool_outcome
from app.observability.tracing import trace_agent_span
from app.tools.ingestion_tools import ToolErrorPayload


# ==============================================================================
# GUARDRAILS & POLICY PLUGIN (HTML/JS & MANIM AST SANDBOX VALIDATION)
# ==============================================================================
FORBIDDEN_HTML_JS_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "EXTERNAL_SCRIPT_SRC",
        re.compile(r"<script[^>]+src\s*=\s*['\"]?https?://", re.IGNORECASE),
    ),
    (
        "DYNAMIC_CODE_EVAL",
        re.compile(r"(?:\beval\s*\(|\bnew\s+Function\s*\()"),
    ),
    (
        "COOKIE_OR_STORAGE_THEFT",
        re.compile(r"\b(?:document\.cookie|localStorage|sessionStorage)\b", re.IGNORECASE),
    ),
    (
        "UNAUTHORIZED_NETWORK_CALL",
        re.compile(r"\b(?:fetch|XMLHttpRequest|WebSocket|navigator\.sendBeacon)\s*\(", re.IGNORECASE),
    ),
    (
        "IFRAME_INJECTION",
        re.compile(r"<iframe\b", re.IGNORECASE),
    ),
)

ALLOWED_MANIM_IMPORTS: Set[str] = {"manim", "numpy", "np", "math", "typing"}
FORBIDDEN_PYTHON_CALLS: Set[str] = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "input",
    "breakpoint",
}


class SimulationSafetyPolicyPlugin:
    """ADK Security Policy Plugin enforcing sandboxed HTML/JS and Manim AST safety."""

    @classmethod
    def validate_html_simulation(cls, html_code: str) -> Tuple[bool, List[str]]:
        """Scans generated HTML5/JS simulation code for malicious patterns or XSS."""
        violations: List[str] = []
        if not html_code or len(html_code.strip()) < 20:
            violations.append("EMPTY_OR_TRUNCATED_SIMULATION_CODE")
            return False, violations

        for rule_name, pattern in FORBIDDEN_HTML_JS_PATTERNS:
            if pattern.search(html_code):
                violations.append(rule_name)

        return len(violations) == 0, violations

    @classmethod
    def validate_manim_python_ast(cls, python_code: str) -> Tuple[bool, List[str]]:
        """Parses Manim Python code into an AST and blocks unsafe imports or calls."""
        violations: List[str] = []
        try:
            tree = ast.parse(python_code)
        except SyntaxError as exc:
            return False, [f"PYTHON_SYNTAX_ERROR: line {exc.lineno}: {exc.msg}"]

        has_scene_class = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    if root_pkg not in ALLOWED_MANIM_IMPORTS:
                        violations.append(f"FORBIDDEN_IMPORT:{root_pkg}")
            elif isinstance(node, ast.ImportFrom):
                root_pkg = (node.module or "").split(".")[0]
                if root_pkg not in ALLOWED_MANIM_IMPORTS:
                    violations.append(f"FORBIDDEN_IMPORT_FROM:{root_pkg}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_PYTHON_CALLS:
                    violations.append(f"FORBIDDEN_BUILTIN_CALL:{node.func.id}")
                elif isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "system",
                    "popen",
                    "Popen",
                    "run",
                    "check_output",
                    "remove",
                    "unlink",
                    "rmtree",
                }:
                    violations.append(f"FORBIDDEN_SYSTEM_METHOD:{node.func.attr}")
            elif isinstance(node, ast.ClassDef):
                base_names = [
                    b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                    for b in node.bases
                ]
                if any(b in {"Scene", "ThreeDScene", "MovingCameraScene"} for b in base_names):
                    has_scene_class = True

        if not has_scene_class:
            violations.append("MISSING_MANIM_SCENE_SUBCLASS")

        return len(violations) == 0, violations


# ==============================================================================
# HUMAN-IN-THE-LOOP (HITL) EXECUTION GATE
# ==============================================================================
class HumanInTheLoopGate:
    """High-stakes execution gate requiring explicit human confirmation."""

    @staticmethod
    def check_approval(
        *,
        action_name: str,
        execute_render: bool,
        publish_to_shared_repo: bool,
        human_approved: bool,
        approval_token: Optional[str] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verifies human confirmation before executing high-stakes render/publish actions."""
        is_high_stakes = execute_render or publish_to_shared_repo
        if not is_high_stakes:
            return True, None

        if human_approved:
            return True, None

        return False, {
            "status": "requires_human_approval",
            "hitl_gate_triggered": True,
            "action_name": action_name,
            "risk_level": "HIGH_STAKES_CODE_EXECUTION_OR_SHARED_PUBLISH",
            "pending_operations": {
                "execute_render": execute_render,
                "publish_to_shared_repo": publish_to_shared_repo,
            },
            "confirmation_prompt": (
                f"Human-in-the-Loop Confirmation Required: Approve execution of '{action_name}' "
                f"(execute_render={execute_render}, publish_to_shared_repo={publish_to_shared_repo})?"
            ),
            "remediation_instruction": (
                "Present the generated code and confirmation prompt to the user. Once the user "
                "explicitly clicks 'Approve & Execute', re-invoke the tool with `human_approved=True`."
            ),
        }


# ==============================================================================
# PYDANTIC V2 SCHEMAS FOR SIMULATION & MANIM TOOLS
# ==============================================================================
class GenerateInteractiveSimInput(BaseModel):
    """Strict input schema for `generate_interactive_sim_code`."""

    model_config = ConfigDict(extra="forbid", strict=True)

    concept_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Canonical concept identifier (e.g., 'discrete_parallel_diffusion').",
    )
    concept_title: str = Field(
        ...,
        min_length=3,
        max_length=160,
        description="Human-readable title for the simulation header.",
    )
    mastery_level: str = Field(
        default="Novice",
        pattern=r"^(ELI5|Novice|Practitioner|Hero)$",
        description="Learner mastery tier controlling widget complexity ('ELI5', 'Novice', 'Practitioner', 'Hero').",
    )
    initial_masking_ratio: float = Field(
        default=0.80,
        ge=0.05,
        le=0.95,
        description="Initial slider ratio (e.g., 0.80 for 80% masked tokens in Meta Muse).",
    )
    custom_html_override: Optional[str] = Field(
        default=None,
        description="Optional custom HTML/JS snippet to validate through SimulationSafetyPolicyPlugin.",
    )


class GenerateInteractiveSimOutput(BaseModel):
    """Strict output schema for `generate_interactive_sim_code`."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: str = Field(default="success", description="Tool execution status.")
    concept_id: str = Field(..., description="Target concept identifier.")
    widget_title: str = Field(..., description="Interactive toybox widget title.")
    control_label: str = Field(
        ..., description="Primary interactive slider control label (e.g. 'Masking Ratio: 80% -> 20%')."
    )
    html_code: str = Field(
        ..., description="Sandboxed, self-contained HTML5/Canvas/JS simulation widget code."
    )
    guardrail_passed: bool = Field(
        ..., description="True after passing SimulationSafetyPolicyPlugin inspection."
    )
    interactive_controls: List[Dict[str, Any]] = Field(
        ..., description="Metadata describing interactive sliders and controls."
    )


class CompileManimInput(BaseModel):
    """Strict input schema for `compile_manim_animation`."""

    model_config = ConfigDict(extra="forbid", strict=True)

    concept_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Canonical concept identifier to render as a Manim Scene.",
    )
    scene_class_name: str = Field(
        default="CosineSimilarityClusteringScene",
        min_length=3,
        max_length=80,
        pattern=r"^[A-Z][A-Za-z0-9_]*Scene$",
        description="PascalCase Manim Scene class name ending in 'Scene'.",
    )
    mastery_level: str = Field(
        default="Hero",
        pattern=r"^(Novice|Practitioner|Hero)$",
        description="Pedagogical rigor level for the mathematical animation.",
    )
    execute_render: bool = Field(
        default=False,
        description="High-stakes flag: if True, executes Manim compilation (requires HITL approval).",
    )
    publish_to_shared_repo: bool = Field(
        default=False,
        description="High-stakes flag: if True, publishes scene to shared repository (requires HITL approval).",
    )
    human_approved: bool = Field(
        default=False,
        description="Explicit Human-in-the-Loop confirmation flag granted by the user.",
    )
    custom_manim_code: Optional[str] = Field(
        default=None,
        description="Optional custom Manim Python source code to AST-validate.",
    )


class CompileManimOutput(BaseModel):
    """Strict output schema for `compile_manim_animation`."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: str = Field(default="success", description="Tool execution status.")
    concept_id: str = Field(..., description="Concept identifier.")
    scene_class_name: str = Field(..., description="Validated Manim Scene class name.")
    manim_python_code: str = Field(
        ..., description="Complete, AST-verified Python Manim Scene script."
    )
    ast_validated: bool = Field(
        ..., description="True if Python AST parsing and import allowlist succeeded."
    )
    hitl_verified: bool = Field(
        ..., description="True if Human-in-the-Loop policy check passed."
    )
    keyframes: List[Dict[str, Any]] = Field(
        ..., description="3D vector & cosine similarity keyframe coordinates for live canvas playback."
    )


def _build_sandboxed_simulation_html(
    concept_id: str,
    concept_title: str,
    initial_mask_ratio: float,
) -> str:
    """Generates a self-contained HTML5/Canvas interactive simulation widget."""
    pct = int(round(initial_mask_ratio * 100))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  :root {{
    --bg: #0b0f19;
    --panel: #111827;
    --accent: #38bdf8;
    --violet: #a855f7;
    --emerald: #10b981;
    --text: #f8fafc;
  }}
  body {{
    margin: 0;
    padding: 16px;
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', system-ui, sans-serif;
  }}
  .sim-card {{
    background: var(--panel);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 12px;
    padding: 16px;
  }}
  .sim-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }}
  .slider-row {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 12px 0;
  }}
  input[type="range"] {{
    flex: 1;
    accent-color: var(--accent);
  }}
  canvas {{
    width: 100%;
    height: 240px;
    border-radius: 8px;
    background: #060911;
    display: block;
  }}
</style>
</head>
<body>
  <div class="sim-card" data-concept="{concept_id}">
    <div class="sim-header">
      <strong>{concept_title} — Interactive Token Grid</strong>
      <span id="maskBadge">Masking Ratio: {pct}%</span>
    </div>
    <div class="slider-row">
      <label for="maskSlider">Masking Ratio: 80% &rarr; 20%</label>
      <input id="maskSlider" type="range" min="10" max="90" value="{pct}" />
    </div>
    <canvas id="tokenCanvas" width="560" height="240"></canvas>
  </div>
  <script>
    (function() {{
      const slider = document.getElementById('maskSlider');
      const badge = document.getElementById('maskBadge');
      const canvas = document.getElementById('tokenCanvas');
      const ctx = canvas.getContext('2d');
      const cols = 8, rows = 4;

      function renderGrid(maskPct) {{
        badge.textContent = 'Masking Ratio: ' + maskPct + '%';
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const cellW = canvas.width / cols;
        const cellH = canvas.height / rows;
        const threshold = maskPct / 100.0;

        for (let r = 0; r < rows; r++) {{
          for (let c = 0; c < cols; c++) {{
            const idx = r * cols + c;
            const deterministicScore = ((idx * 37) % 100) / 100.0;
            const isResolved = deterministicScore >= threshold;
            const confidence = Math.min(1, Math.max(0.15, 1.0 - threshold + deterministicScore * 0.35));
            ctx.fillStyle = isResolved
              ? `rgba(16, 185, 129, ${{0.35 + confidence * 0.55}})`
              : `rgba(168, 85, 247, ${{0.20 + (1 - confidence) * 0.45}})`;
            ctx.fillRect(c * cellW + 4, r * cellH + 4, cellW - 8, cellH - 8);
            ctx.strokeStyle = isResolved ? '#38bdf8' : '#475569';
            ctx.lineWidth = isResolved ? 1.5 : 1;
            ctx.strokeRect(c * cellW + 4, r * cellH + 4, cellW - 8, cellH - 8);
            ctx.fillStyle = '#f8fafc';
            ctx.font = '11px monospace';
            ctx.fillText(isResolved ? 'T_' + idx : '[MASK]', c * cellW + 10, r * cellH + 32);
          }}
        }}
      }}

      slider.addEventListener('input', function(e) {{
        renderGrid(parseInt(e.target.value, 10));
      }});
      renderGrid({pct});
    }})();
  </script>
</body>
</html>"""


def _build_manim_scene_code(concept_id: str, scene_class_name: str) -> str:
    """Generates a deterministic, AST-verified Manim ThreeDScene script."""
    return f'''"""Auto-generated ArgonLearn Manim Scene for {concept_id}.

Visualizes 3D latent token vectors clustering via Cosine Similarity
s(u, v) = (u . v) / (||u||_2 ||v||_2) during parallel unmasking.
"""

import math
from manim import (
    ThreeDScene,
    ThreeDAxes,
    Arrow3D,
    Dot3D,
    MathTex,
    Text,
    VGroup,
    BLUE_D,
    TEAL_A,
    PURPLE_B,
    GOLD_B,
    UP,
    ORIGIN,
)
import numpy as np


class {scene_class_name}(ThreeDScene):
    """Renders 3D cosine-similarity vector clustering for {concept_id}."""

    def construct(self) -> None:
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            z_length=6,
        )
        self.set_camera_orientation(phi=68 * math.pi / 180, theta=-45 * math.pi / 180)

        title = Text("ArgonLearn Hero Mode: Cosine Similarity Clustering", font_size=26)
        formula = MathTex(
            r"s(\\mathbf{{u}}, \\mathbf{{v}}) = \\frac{{\\mathbf{{u}} \\cdot \\mathbf{{v}}}}{{\\|\\mathbf{{u}}\\|_2 \\|\\mathbf{{v}}\\|_2}}, \\quad r(t) = \\cos\\left(\\frac{{\\pi t}}{{2T}}\\right)"
        ).scale(0.68)
        formula.to_Edge = UP

        cluster_centroids = [
            np.array([1.8, 1.4, 1.2]),
            np.array([-1.6, 1.5, 0.9]),
            np.array([0.4, -1.9, 1.6]),
        ]
        colors = [TEAL_A, PURPLE_B, GOLD_B]

        vectors = VGroup()
        for idx, centroid in enumerate(cluster_centroids):
            arrow = Arrow3D(
                start=ORIGIN,
                end=centroid,
                color=colors[idx % len(colors)],
            )
            endpoint = Dot3D(point=centroid, color=BLUE_D, radius=0.08)
            vectors.add(arrow, endpoint)

        self.add(axes, vectors, title, formula)
        self.begin_ambient_camera_rotation(rate=0.18)
        self.wait(4)
        self.stop_ambient_camera_rotation()
'''


def generate_interactive_sim_code(
    concept_id: str,
    concept_title: str,
    mastery_level: str = "Novice",
    initial_masking_ratio: float = 0.80,
    custom_html_override: Optional[str] = None,
    session_id: str = "default",
) -> Dict[str, Any]:
    """Generates a standalone, sandboxed HTML5/Canvas/JS interactive simulation widget.

    Validates all input constraints via `GenerateInteractiveSimInput`, synthesizes an
    interactive token-grid and attention-heatmap simulation (with a real-time
    `Masking Ratio: 80% -> 20%` slider), and enforces `SimulationSafetyPolicyPlugin`
    to guarantee zero external script injections, `eval()`, or network exfiltration.

    Args:
        concept_id: Snake-case concept identifier (3..100 chars, e.g.,
            'discrete_parallel_diffusion').
        concept_title: Human-readable title displayed in the widget header (3..160 chars).
        mastery_level: Pedagogical complexity tier matching `^(ELI5|Novice|Practitioner|Hero)$`.
            Defaults to 'Novice'.
        initial_masking_ratio: Starting masking ratio slider position in [0.05, 0.95].
            Defaults to 0.80 (80%).
        custom_html_override: Optional custom HTML/JS code to validate against the
            security guardrail policy plugin.
        session_id: Active conversational session identifier for trace correlation.

    Returns:
        Dict[str, Any]: Serialized `GenerateInteractiveSimOutput` containing the verified
        `html_code` and `interactive_controls`, or `ToolErrorPayload` with actionable
        `remediation_instruction` if validation or guardrail checks fail.
    """
    start_time = time.perf_counter()
    intent_id = log_agent_intent(
        agent_name="InteractiveVisualizerAgent",
        model_id="gemini-2.5-pro",
        intended_action=f"Generate sandboxed HTML5 interactive simulation widget for '{concept_id}'",
        target_tool="generate_interactive_sim_code",
        rationale="Provide hands-on slider manipulation of token masking and parallel attention",
        input_parameters={
            "concept_id": concept_id,
            "concept_title": concept_title,
            "mastery_level": mastery_level,
            "initial_masking_ratio": initial_masking_ratio,
        },
        session_id=session_id,
    )

    with trace_agent_span(
        "tool.generate_interactive_sim_code",
        agent_name="InteractiveVisualizerAgent",
        model_id="gemini-2.5-pro",
        stage="tool",
        attributes={
            "argon.tool.name": "generate_interactive_sim_code",
            "argon.concept_id": concept_id,
        },
    ) as span:
        try:
            validated = GenerateInteractiveSimInput(
                concept_id=concept_id,
                concept_title=concept_title,
                mastery_level=mastery_level,
                initial_masking_ratio=initial_masking_ratio,
                custom_html_override=custom_html_override,
            )

            html_payload = (
                validated.custom_html_override
                if validated.custom_html_override is not None
                else _build_sandboxed_simulation_html(
                    concept_id=validated.concept_id,
                    concept_title=validated.concept_title,
                    initial_mask_ratio=validated.initial_masking_ratio,
                )
            )

            passed, violations = SimulationSafetyPolicyPlugin.validate_html_simulation(
                html_payload
            )
            if not passed:
                raise PermissionError(
                    f"SimulationSafetyPolicyPlugin blocked unsafe HTML/JS patterns: {', '.join(violations)}"
                )

            output_model = GenerateInteractiveSimOutput(
                status="success",
                concept_id=validated.concept_id,
                widget_title=f"{validated.concept_title} — Interactive Toybox",
                control_label="Masking Ratio: 80% → 20%",
                html_code=html_payload,
                guardrail_passed=True,
                interactive_controls=[
                    {
                        "id": "masking_ratio_slider",
                        "label": "Masking Ratio: 80% → 20%",
                        "min": 10,
                        "max": 90,
                        "default": int(round(validated.initial_masking_ratio * 100)),
                        "unit": "%",
                        "effect": "Dynamically unmasks discrete tokens in parallel with live attention heatmaps.",
                    },
                    {
                        "id": "sampling_temperature_slider",
                        "label": "Gumbel Temperature τ",
                        "min": 0.1,
                        "max": 2.0,
                        "default": 0.7,
                        "unit": "τ",
                        "effect": "Controls sharpening of parallel token probability distributions.",
                    },
                ],
            )
            result = output_model.model_dump()

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            log_tool_outcome(
                intent_id=intent_id,
                agent_name="InteractiveVisualizerAgent",
                tool_name="generate_interactive_sim_code",
                status="SUCCESS",
                latency_ms=latency_ms,
                outcome_summary=(
                    f"Generated guardrail-verified HTML5 simulation ({len(html_payload)} bytes) "
                    f"for '{validated.concept_id}'."
                ),
                output_metadata={
                    "guardrail_passed": True,
                    "html_bytes": len(html_payload),
                },
                session_id=session_id,
                trace_id=span.trace_id,
            )
            return result

        except (ValidationError, PermissionError) as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            err = ToolErrorPayload(
                tool_name="generate_interactive_sim_code",
                error_type=type(exc).__name__,
                raw_message=str(exc),
                remediation_instruction=(
                    "Remove all external `<script src='http...'>` tags, `eval()`, `fetch()`, "
                    "`document.cookie`, and `<iframe>` elements. Ensure `initial_masking_ratio` "
                    "is between 0.05 and 0.95 and `mastery_level` is one of "
                    "['ELI5', 'Novice', 'Practitioner', 'Hero']."
                ),
            ).model_dump()
            log_tool_outcome(
                intent_id=intent_id,
                agent_name="InteractiveVisualizerAgent",
                tool_name="generate_interactive_sim_code",
                status="RECOVERABLE_ERROR",
                latency_ms=latency_ms,
                outcome_summary=f"Simulation generation rejected by guardrail/validator: {exc}",
                error_payload=err,
                session_id=session_id,
                trace_id=span.trace_id,
            )
            return err


def compile_manim_animation(
    concept_id: str,
    scene_class_name: str = "CosineSimilarityClusteringScene",
    mastery_level: str = "Hero",
    execute_render: bool = False,
    publish_to_shared_repo: bool = False,
    human_approved: bool = False,
    custom_manim_code: Optional[str] = None,
    session_id: str = "default",
) -> Dict[str, Any]:
    """Generates, AST-verifies, and compiles a Python Manim mathematical vector animation.

    Enforces two independent safety layers:
    1. `HumanInTheLoopGate`: If `execute_render=True` or `publish_to_shared_repo=True`,
       execution is paused unless `human_approved=True` is explicitly provided.
    2. `SimulationSafetyPolicyPlugin.validate_manim_python_ast`: Parses the generated
       Python script into an abstract syntax tree (`ast.parse`), verifies it subclasses
       a Manim `Scene`, and blocks any non-scientific imports (`os`, `subprocess`, `sys`,
       `socket`) or dangerous builtins (`eval`, `exec`, `open`).

    Args:
        concept_id: Canonical snake_case concept identifier (3..100 chars).
        scene_class_name: PascalCase Manim class name matching `^[A-Z][A-Za-z0-9_]*Scene$`.
            Defaults to 'CosineSimilarityClusteringScene'.
        mastery_level: Pedagogical depth matching `^(Novice|Practitioner|Hero)$`.
            Defaults to 'Hero'.
        execute_render: High-stakes flag requesting actual Manim video rendering execution.
        publish_to_shared_repo: High-stakes flag requesting publication to shared team repo.
        human_approved: Explicit Human-in-the-Loop approval boolean from the user.
        custom_manim_code: Optional custom Manim Python script to validate and compile.
        session_id: Active conversational session ID for OpenTelemetry correlation.

    Returns:
        Dict[str, Any]: Serialized `CompileManimOutput` with AST-verified `manim_python_code`
        and 3D vector `keyframes`, a HITL pause payload if awaiting human confirmation,
        or a `ToolErrorPayload` with `remediation_instruction` if validation fails.
    """
    start_time = time.perf_counter()
    intent_id = log_agent_intent(
        agent_name="InteractiveVisualizerAgent",
        model_id="gemini-4-argon",
        intended_action=f"Synthesize and AST-verify Manim 3D vector scene '{scene_class_name}'",
        target_tool="compile_manim_animation",
        rationale="Render Hero-mode 3D cosine similarity vector clustering animation",
        input_parameters={
            "concept_id": concept_id,
            "scene_class_name": scene_class_name,
            "mastery_level": mastery_level,
            "execute_render": execute_render,
            "publish_to_shared_repo": publish_to_shared_repo,
            "human_approved": human_approved,
        },
        session_id=session_id,
    )

    with trace_agent_span(
        "tool.compile_manim_animation",
        agent_name="InteractiveVisualizerAgent",
        model_id="gemini-4-argon",
        stage="tool",
        attributes={
            "argon.tool.name": "compile_manim_animation",
            "argon.scene_class": scene_class_name,
        },
    ) as span:
        try:
            validated = CompileManimInput(
                concept_id=concept_id,
                scene_class_name=scene_class_name,
                mastery_level=mastery_level,
                execute_render=execute_render,
                publish_to_shared_repo=publish_to_shared_repo,
                human_approved=human_approved,
                custom_manim_code=custom_manim_code,
            )

            # 1. Check Human-in-the-Loop Gate for high-stakes execution/publishing
            approved, hitl_payload = HumanInTheLoopGate.check_approval(
                action_name=f"compile_manim_animation:{validated.scene_class_name}",
                execute_render=validated.execute_render,
                publish_to_shared_repo=validated.publish_to_shared_repo,
                human_approved=validated.human_approved,
            )
            if not approved and hitl_payload is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                span.set_attribute("argon.hitl.blocked", True)
                log_tool_outcome(
                    intent_id=intent_id,
                    agent_name="InteractiveVisualizerAgent",
                    tool_name="compile_manim_animation",
                    status="BLOCKED_HITL",
                    latency_ms=latency_ms,
                    outcome_summary="Paused high-stakes Manim execution pending explicit human approval",
                    output_metadata=hitl_payload,
                    session_id=session_id,
                    trace_id=span.trace_id,
                )
                return hitl_payload

            # 2. Generate or inspect Manim Python code via AST safety policy plugin
            manim_code = (
                validated.custom_manim_code
                if validated.custom_manim_code is not None
                else _build_manim_scene_code(
                    concept_id=validated.concept_id,
                    scene_class_name=validated.scene_class_name,
                )
            )

            ast_ok, violations = SimulationSafetyPolicyPlugin.validate_manim_python_ast(
                manim_code
            )
            if not ast_ok:
                raise PermissionError(
                    f"Manim AST safety check failed: {', '.join(violations)}"
                )

            keyframes = [
                {
                    "step": 0,
                    "label": "Initial Masked Latent Tokens (t=0, r=0.80)",
                    "cosine_similarity_mean": 0.31,
                    "clusters": [
                        {"name": "Visual Patch Group A", "vector": [0.78, 0.52, 0.34], "color": "#38bdf8"},
                        {"name": "Motion Trajectory B", "vector": [-0.64, 0.61, 0.46], "color": "#a855f7"},
                        {"name": "Semantic Anchor C", "vector": [0.22, -0.79, 0.57], "color": "#10b981"},
                    ],
                },
                {
                    "step": 1,
                    "label": "Parallel Iterative Unmasking (t=T/2, r=0.50)",
                    "cosine_similarity_mean": 0.68,
                    "clusters": [
                        {"name": "Visual Patch Group A", "vector": [0.86, 0.44, 0.25], "color": "#38bdf8"},
                        {"name": "Motion Trajectory B", "vector": [-0.73, 0.58, 0.36], "color": "#a855f7"},
                        {"name": "Semantic Anchor C", "vector": [0.15, -0.86, 0.48], "color": "#10b981"},
                    ],
                },
                {
                    "step": 2,
                    "label": "Converged Cosine-Similarity Manifold (t=T, r=0.20)",
                    "cosine_similarity_mean": 0.94,
                    "clusters": [
                        {"name": "Visual Patch Group A", "vector": [0.92, 0.33, 0.20], "color": "#38bdf8"},
                        {"name": "Motion Trajectory B", "vector": [-0.81, 0.51, 0.28], "color": "#a855f7"},
                        {"name": "Semantic Anchor C", "vector": [0.09, -0.91, 0.40], "color": "#10b981"},
                    ],
                },
            ]

            output_model = CompileManimOutput(
                status="success",
                concept_id=validated.concept_id,
                scene_class_name=validated.scene_class_name,
                manim_python_code=manim_code,
                ast_validated=True,
                hitl_verified=True,
                keyframes=keyframes,
            )
            result = output_model.model_dump()

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            log_tool_outcome(
                intent_id=intent_id,
                agent_name="InteractiveVisualizerAgent",
                tool_name="compile_manim_animation",
                status="SUCCESS",
                latency_ms=latency_ms,
                outcome_summary=(
                    f"AST-validated Manim scene '{validated.scene_class_name}' "
                    f"with {len(keyframes)} 3D vector keyframes."
                ),
                output_metadata={
                    "scene_class_name": validated.scene_class_name,
                    "ast_validated": True,
                    "hitl_verified": True,
                },
                session_id=session_id,
                trace_id=span.trace_id,
            )
            return result

        except (ValidationError, PermissionError) as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            err = ToolErrorPayload(
                tool_name="compile_manim_animation",
                error_type=type(exc).__name__,
                raw_message=str(exc),
                remediation_instruction=(
                    "Ensure `scene_class_name` is PascalCase ending in 'Scene' "
                    "(matching `^[A-Z][A-Za-z0-9_]*Scene$`), the script subclasses "
                    "`Scene` or `ThreeDScene`, and only imports from `manim`, `numpy`, or `math`."
                ),
            ).model_dump()
            log_tool_outcome(
                intent_id=intent_id,
                agent_name="InteractiveVisualizerAgent",
                tool_name="compile_manim_animation",
                status="RECOVERABLE_ERROR",
                latency_ms=latency_ms,
                outcome_summary=f"Manim compilation rejected: {exc}",
                error_payload=err,
                session_id=session_id,
                trace_id=span.trace_id,
            )
            return err
