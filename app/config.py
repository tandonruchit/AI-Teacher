"""Configuration, Secret Manager injection, and System Constitution for ArgonLearn.

Implements strict Secret Manager resolution (zero hardcoded keys), model routing
tiers, context compaction thresholds, and the multi-agent System Constitution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Type, get_origin, get_args

try:
    from google.cloud import secretmanager
    HAS_SECRET_MANAGER = True
except ImportError:
    secretmanager = None  # type: ignore[assignment]
    HAS_SECRET_MANAGER = False

try:
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        ValidationError,
        field_validator,
    )
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

    class ValidationError(ValueError):
        """Fallback validation error matching Pydantic v2 ValidationError interface."""

        def __init__(self, errors: List[Dict[str, Any]], model_name: str = "BaseModel") -> None:
            self._errors = errors
            self.model_name = model_name
            msg = "; ".join(f"{e.get('loc')}: {e.get('msg')}" for e in errors)
            super().__init__(f"{len(errors)} validation error(s) for {model_name}: {msg}")

        def errors(self) -> List[Dict[str, Any]]:
            return self._errors

    class _FieldInfo:
        def __init__(
            self,
            default: Any = ...,
            *,
            default_factory: Optional[Callable[[], Any]] = None,
            description: str = "",
            min_length: Optional[int] = None,
            max_length: Optional[int] = None,
            ge: Optional[float] = None,
            le: Optional[float] = None,
            gt: Optional[float] = None,
            lt: Optional[float] = None,
            pattern: Optional[str] = None,
            examples: Optional[List[Any]] = None,
        ) -> None:
            self.default = default
            self.default_factory = default_factory
            self.description = description
            self.min_length = min_length
            self.max_length = max_length
            self.ge = ge
            self.le = le
            self.gt = gt
            self.lt = lt
            self.pattern = pattern
            self.examples = examples or []

    def Field(  # type: ignore[misc]
        default: Any = ...,
        *,
        default_factory: Optional[Callable[[], Any]] = None,
        description: str = "",
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        ge: Optional[float] = None,
        le: Optional[float] = None,
        gt: Optional[float] = None,
        lt: Optional[float] = None,
        pattern: Optional[str] = None,
        examples: Optional[List[Any]] = None,
    ) -> Any:
        return _FieldInfo(
            default=default,
            default_factory=default_factory,
            description=description,
            min_length=min_length,
            max_length=max_length,
            ge=ge,
            le=le,
            gt=gt,
            lt=lt,
            pattern=pattern,
            examples=examples,
        )

    def ConfigDict(**kwargs: Any) -> Dict[str, Any]:  # type: ignore[misc]
        return dict(kwargs)

    def field_validator(*fields: str, mode: str = "after") -> Callable[[Any], Any]:  # type: ignore[misc]
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            target = fn.__func__ if isinstance(fn, classmethod) else fn
            setattr(target, "__validator_fields__", fields)
            setattr(target, "__validator_mode__", mode)
            return classmethod(target)  # type: ignore[return-value]
        return decorator

    class BaseModel:  # type: ignore[no-redef]
        """Strict JSON-schema-validated BaseModel fallback when pydantic is absent."""

        model_config: Dict[str, Any] = {"extra": "forbid", "strict": True}

        def __init__(self, **data: Any) -> None:
            cls = self.__class__
            annotations = getattr(cls, "__annotations__", {})
            errors: List[Dict[str, Any]] = []
            extra_policy = cls.model_config.get("extra", "forbid")

            if extra_policy == "forbid":
                for key in data:
                    if key not in annotations:
                        errors.append({
                            "loc": (key,),
                            "msg": f"Extra inputs are not permitted: '{key}'",
                            "type": "extra_forbidden",
                        })

            for name, _ in annotations.items():
                field_info = getattr(cls, name, None)
                if name in data:
                    val = data[name]
                elif isinstance(field_info, _FieldInfo):
                    if field_info.default is not ...:
                        val = field_info.default
                    elif field_info.default_factory is not None:
                        val = field_info.default_factory()
                    else:
                        errors.append({
                            "loc": (name,),
                            "msg": f"Field '{name}' is required",
                            "type": "missing",
                        })
                        continue
                elif field_info is not None:
                    val = field_info
                else:
                    errors.append({
                        "loc": (name,),
                        "msg": f"Field '{name}' is required",
                        "type": "missing",
                    })
                    continue

                if isinstance(field_info, _FieldInfo) and val is not None:
                    if field_info.min_length is not None and hasattr(val, "__len__"):
                        if len(val) < field_info.min_length:
                            errors.append({
                                "loc": (name,),
                                "msg": f"Length must be >= {field_info.min_length}",
                                "type": "too_short",
                            })
                    if field_info.max_length is not None and hasattr(val, "__len__"):
                        if len(val) > field_info.max_length:
                            errors.append({
                                "loc": (name,),
                                "msg": f"Length must be <= {field_info.max_length}",
                                "type": "too_long",
                            })
                    if field_info.ge is not None and isinstance(val, (int, float)):
                        if val < field_info.ge:
                            errors.append({
                                "loc": (name,),
                                "msg": f"Input should be greater than or equal to {field_info.ge}",
                                "type": "greater_than_equal",
                            })
                    if field_info.le is not None and isinstance(val, (int, float)):
                        if val > field_info.le:
                            errors.append({
                                "loc": (name,),
                                "msg": f"Input should be less than or equal to {field_info.le}",
                                "type": "less_than_equal",
                            })
                    if field_info.gt is not None and isinstance(val, (int, float)):
                        if val <= field_info.gt:
                            errors.append({
                                "loc": (name,),
                                "msg": f"Input should be greater than {field_info.gt}",
                                "type": "greater_than",
                            })
                    if field_info.lt is not None and isinstance(val, (int, float)):
                        if val >= field_info.lt:
                            errors.append({
                                "loc": (name,),
                                "msg": f"Input should be less than {field_info.lt}",
                                "type": "less_than",
                            })
                    if field_info.pattern is not None and isinstance(val, str):
                        if not re.match(field_info.pattern, val):
                            errors.append({
                                "loc": (name,),
                                "msg": f"String should match pattern '{field_info.pattern}'",
                                "type": "string_pattern_mismatch",
                            })

                for attr_name in dir(cls):
                    attr = inspect.getattr_static(cls, attr_name)
                    raw_fn = attr.__func__ if isinstance(attr, classmethod) else attr
                    v_fields = getattr(raw_fn, "__validator_fields__", None)
                    if v_fields and (name in v_fields or "*" in v_fields):
                        try:
                            val = raw_fn(cls, val)
                        except Exception as exc:
                            errors.append({
                                "loc": (name,),
                                "msg": str(exc),
                                "type": "value_error",
                            })

                setattr(self, name, val)

            if errors:
                raise ValidationError(errors, model_name=cls.__name__)

        def model_dump(self) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for name in getattr(self.__class__, "__annotations__", {}):
                val = getattr(self, name, None)
                if isinstance(val, BaseModel):
                    out[name] = val.model_dump()
                elif isinstance(val, list):
                    out[name] = [
                        item.model_dump() if isinstance(item, BaseModel) else item
                        for item in val
                    ]
                elif isinstance(val, dict):
                    out[name] = {
                        k: (v.model_dump() if isinstance(v, BaseModel) else v)
                        for k, v in val.items()
                    }
                else:
                    out[name] = val
            return out

        def model_dump_json(self, indent: Optional[int] = None) -> str:
            return json.dumps(self.model_dump(), indent=indent)

        @classmethod
        def model_json_schema(cls) -> Dict[str, Any]:
            properties: Dict[str, Any] = {}
            required: List[str] = []
            annotations = getattr(cls, "__annotations__", {})
            for name, type_hint in annotations.items():
                field_info = getattr(cls, name, None)
                prop: Dict[str, Any] = {"title": name.replace("_", " ").title()}
                hint_str = str(type_hint)
                if "int" in hint_str:
                    prop["type"] = "integer"
                elif "float" in hint_str:
                    prop["type"] = "number"
                elif "bool" in hint_str:
                    prop["type"] = "boolean"
                elif "List" in hint_str or "list" in hint_str:
                    prop["type"] = "array"
                elif "Dict" in hint_str or "dict" in hint_str:
                    prop["type"] = "object"
                else:
                    prop["type"] = "string"

                if isinstance(field_info, _FieldInfo):
                    if field_info.description:
                        prop["description"] = field_info.description
                    if field_info.min_length is not None:
                        prop["minLength"] = field_info.min_length
                    if field_info.max_length is not None:
                        prop["maxLength"] = field_info.max_length
                    if field_info.ge is not None:
                        prop["minimum"] = field_info.ge
                    if field_info.le is not None:
                        prop["maximum"] = field_info.le
                    if field_info.pattern is not None:
                        prop["pattern"] = field_info.pattern
                    if field_info.default is ... and field_info.default_factory is None:
                        required.append(name)
                    elif field_info.default is not ...:
                        prop["default"] = field_info.default
                elif field_info is None:
                    required.append(name)

                properties[name] = prop

            return {
                "title": cls.__name__,
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": cls.model_config.get("extra") != "forbid",
            }


# ==============================================================================
# SYSTEM PROMPT CONSTITUTION (STRICT ROLE BOUNDARIES & MATHEMATICAL RIGOR)
# ==============================================================================
SYSTEM_CONSTITUTION = """
# ARGONLEARN SYSTEM CONSTITUTION (v1.0)

You are operating within **ArgonLearn**, a production multi-agent AI educational
visualizer built with the Google Agent Development Kit (ADK). Every agent in this
mesh MUST strictly adhere to the following constitutional invariants:

## 1. ROLE BOUNDARIES & SEPARATION OF CONCERNS
- **TriageIngestAgent (`gemini-2.5-flash`)**: Exclusively parses inbound research
  headlines, arXiv abstracts, or learner queries. Extracts canonical AI concepts,
  identifies prerequisites, and queries the learner's knowledge graph. Never
  generates executable HTML/JS or Manim scripts directly.
- **PedagogyCoordinatorAgent (`gemini-2.5-pro` / `gemini-4-argon`)**: Exclusively
  synthesizes progressive explanations across four mastery tiers:
  1. `ELI5` (Intuitive physical/everyday analogy with zero jargon)
  2. `Novice` (High-level architectural flow and token/vector intuition)
  3. `Practitioner` (Algorithmic mechanics, attention masks, routing equations)
  4. `Hero` (Rigorous mathematical formulation, loss objectives, complexity bounds)
- **InteractiveVisualizerAgent (`gemini-2.5-pro` / `gemini-4-argon`)**: Exclusively
  generates sandboxed HTML5/Canvas/Three.js interactive simulations and deterministic
  Python Manim `Scene` classes.

## 2. ZERO-HALLUCINATION MATHEMATICAL EXPLANATIONS
- Every mathematical formula (e.g., Masked Generative Transformer cross-entropy over
  masked token subsets, Top-K Mixture-of-Experts softmax gating with load-balancing
  auxiliary loss, Speculative Decoding acceptance probability min(1, p(x)/q(x)),
  and Cosine Similarity s(u, v) = (u · v) / (||u|| ||v||)) MUST be stated with exact
  variable definitions, tensor dimensions, and boundary conditions.
- Never fabricate citations, benchmark scores, or non-existent tensor operations.
  If a paper's exact hyperparameter is unspecified in the input context, state the
  canonical formulation explicitly and label empirical defaults.

## 3. CODE EXECUTION & SANDBOX SAFETY POLICY
- **HTML5/JS Simulation Sandbox**: Generated interactive widgets MUST be 100%
  self-contained. External `<script src="http...">` tags, `eval()`, `Function()`,
  `document.cookie`, `localStorage`, and outbound `fetch()`/`XMLHttpRequest` calls
  are strictly prohibited and blocked by `SimulationSafetyPolicyPlugin`.
- **Manim Animation Sandbox**: Generated Manim code MUST subclass `manim.Scene`
  (or `manim.ThreeDScene`) and import solely from `manim`, `math`, and `numpy`.
  System calls (`os`, `subprocess`, `sys`, `socket`, `shutil`, `open`, `exec`,
  `eval`, `__import__`) are strictly forbidden.
- **Human-in-the-Loop (HITL) Gate**: Any request to execute arbitrary user-supplied
  simulation scripts or publish artifacts to shared repositories MUST pause and
  require explicit human approval (`human_approved=True`).

## 4. PRIVACY & OBSERVABILITY COMPLIANCE
- Never repeat or store raw PII (emails, API keys, employee IDs, phone numbers).
- Always respect structured tool error `remediation_instruction` payloads on failure.
""".strip()


# ==============================================================================
# SECRET MANAGER & ENVIRONMENT CONFIGURATION
# ==============================================================================
def resolve_secret(
    project_id: str,
    secret_id: str,
    env_var_fallback: str,
    version_id: str = "latest",
) -> str:
    """Resolves a sensitive credential from Google Cloud Secret Manager or environment.

    Prioritizes explicit environment variables injected by Cloud Run Secret Manager
    mounts, then queries the Secret Manager API directly when running in GCP, and
    returns an empty string (triggering deterministic local mode) if unavailable.

    Args:
        project_id: The Google Cloud project ID hosting Secret Manager.
        secret_id: The Secret Manager secret identifier.
        env_var_fallback: Environment variable name to check first.
        version_id: Secret version to access (defaults to 'latest').

    Returns:
        The resolved secret string without trailing newlines, or empty string if
        running in local offline/emulator mode.
    """
    env_value = os.environ.get(env_var_fallback, "").strip()
    if env_value:
        return env_value

    if HAS_SECRET_MANAGER and project_id and project_id != "argon-learn-local":
        try:
            client = secretmanager.SecretManagerServiceClient()
            resource_name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
            response = client.access_secret_version(request={"name": resource_name})
            return response.payload.data.decode("utf-8").strip()
        except Exception:
            return ""

    return ""


@dataclass(frozen=True)
class AppSettings:
    """Immutable application settings resolved from environment and Secret Manager."""

    app_name: str = "argon-learn"
    environment: str = field(default_factory=lambda: os.environ.get("APP_ENV", "development"))
    gcp_project: str = field(
        default_factory=lambda: os.environ.get("GCP_PROJECT", "argon-learn-local")
    )
    gcp_region: str = field(
        default_factory=lambda: os.environ.get("GCP_REGION", "us-central1")
    )
    firestore_database: str = field(
        default_factory=lambda: os.environ.get("FIRESTORE_DATABASE", "(default)")
    )
    gemini_api_key_secret_id: str = field(
        default_factory=lambda: os.environ.get(
            "GEMINI_API_KEY_SECRET_ID", "argon-learn-gemini-api-key"
        )
    )
    gemini_api_key: str = ""

    # Strategic Model Routing Tiers
    triage_model: str = field(
        default_factory=lambda: os.environ.get("TRIAGE_MODEL", "gemini-2.5-flash")
    )
    pedagogy_model: str = field(
        default_factory=lambda: os.environ.get("PEDAGOGY_MODEL", "gemini-2.5-pro")
    )
    visualizer_model: str = field(
        default_factory=lambda: os.environ.get("VISUALIZER_MODEL", "gemini-2.5-pro")
    )
    frontier_model: str = field(
        default_factory=lambda: os.environ.get("FRONTIER_MODEL", "gemini-4-argon")
    )

    # Context Compaction & Sliding Window Memory Parameters
    max_context_tokens: int = field(
        default_factory=lambda: int(os.environ.get("MAX_CONTEXT_TOKENS", "4096"))
    )
    sliding_window_turns: int = field(
        default_factory=lambda: int(os.environ.get("SLIDING_WINDOW_TURNS", "8"))
    )
    compaction_target_ratio: float = 0.60

    # Observability & Safety Flags
    enable_cloud_trace: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_CLOUD_TRACE", "false").lower() == "true"
    )
    require_hitl_for_execution: bool = True


def load_settings() -> AppSettings:
    """Loads application settings and injects secrets from Secret Manager / env."""
    gcp_project = os.environ.get("GCP_PROJECT", "argon-learn-local")
    secret_id = os.environ.get("GEMINI_API_KEY_SECRET_ID", "argon-learn-gemini-api-key")
    api_key = resolve_secret(
        project_id=gcp_project,
        secret_id=secret_id,
        env_var_fallback="GEMINI_API_KEY",
    )
    return AppSettings(
        gcp_project=gcp_project,
        gemini_api_key_secret_id=secret_id,
        gemini_api_key=api_key,
    )


settings = load_settings()
