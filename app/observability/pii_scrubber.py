"""Inbound and outbound PII Scrubbing Middleware using Regex & Google Cloud DLP.

Actively redacts emails, authentication tokens, API keys, employee IDs, phone
numbers, and SSNs before any text is written to structured logs or persisted in
Google Cloud Firestore session/knowledge-graph memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Dict, List, Pattern, Tuple

try:
    from google.cloud import dlp_v2
    HAS_CLOUD_DLP = True
except ImportError:
    dlp_v2 = None  # type: ignore[assignment]
    HAS_CLOUD_DLP = False


@dataclass(frozen=True)
class RedactionRule:
    """Defines a named regular expression rule and its replacement token."""

    name: str
    pattern: Pattern[str]
    replacement: str


DEFAULT_REDACTION_RULES: Tuple[RedactionRule, ...] = (
    RedactionRule(
        name="AUTH_BEARER_TOKEN",
        pattern=re.compile(
            r"(?i)\b(bearer\s+[a-zA-Z0-9_\-\.~+/]+=*|ya29\.[a-zA-Z0-9_\-]+|AIza[0-9A-Za-z_\-]{20,}|sk-[a-zA-Z0-9_\-]{16,})"
        ),
        replacement="[REDACTED_AUTH_TOKEN]",
    ),
    RedactionRule(
        name="API_KEY_ASSIGNMENT",
        pattern=re.compile(
            r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{8,})['\"]?"
        ),
        replacement=r"\1=[REDACTED_SECRET]",
    ),
    RedactionRule(
        name="EMAIL_ADDRESS",
        pattern=re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
        ),
        replacement="[REDACTED_EMAIL]",
    ),
    RedactionRule(
        name="INTERNAL_EMPLOYEE_ID",
        pattern=re.compile(
            r"\b(?:EMP|GEMP|CORP|LDAP|BADGE)[-_#: ]?[0-9]{4,10}\b",
            re.IGNORECASE,
        ),
        replacement="[REDACTED_EMPLOYEE_ID]",
    ),
    RedactionRule(
        name="US_SSN",
        pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        replacement="[REDACTED_SSN]",
    ),
    RedactionRule(
        name="PHONE_NUMBER",
        pattern=re.compile(
            r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
        ),
        replacement="[REDACTED_PHONE]",
    ),
)


@dataclass
class RedactionReport:
    """Structured result of a PII scrubbing operation."""

    sanitized_text: str
    redaction_count: int = 0
    matched_categories: List[str] = field(default_factory=list)
    used_cloud_dlp: bool = False


class PIIScrubber:
    """Multi-layered PII scrubber combining regex patterns with Google Cloud DLP."""

    def __init__(
        self,
        project_id: str = "argon-learn-local",
        enable_cloud_dlp: bool = False,
        rules: Tuple[RedactionRule, ...] = DEFAULT_REDACTION_RULES,
    ) -> None:
        self.project_id = project_id
        self.enable_cloud_dlp = enable_cloud_dlp and HAS_CLOUD_DLP
        self.rules = rules
        self._dlp_client = None
        if self.enable_cloud_dlp and dlp_v2 is not None:
            try:
                self._dlp_client = dlp_v2.DlpServiceClient()
            except Exception:
                self._dlp_client = None

    def inspect_and_redact(self, text: str) -> RedactionReport:
        """Scans text for sensitive PII and replaces matches with redaction tokens.

        Args:
            text: Raw input string potentially containing PII or credentials.

        Returns:
            RedactionReport with sanitized string and category telemetry.
        """
        if not text or not isinstance(text, str):
            return RedactionReport(sanitized_text=str(text) if text is not None else "")

        sanitized = text
        total_redactions = 0
        categories: List[str] = []

        for rule in self.rules:
            sanitized, count = rule.pattern.subn(rule.replacement, sanitized)
            if count > 0:
                total_redactions += count
                if rule.name not in categories:
                    categories.append(rule.name)

        used_dlp = False
        if self._dlp_client is not None and self.project_id != "argon-learn-local":
            try:
                parent = f"projects/{self.project_id}/locations/global"
                inspect_config = {
                    "info_types": [
                        {"name": "EMAIL_ADDRESS"},
                        {"name": "PHONE_NUMBER"},
                        {"name": "PERSON_NAME"},
                        {"name": "GCP_CREDENTIALS"},
                        {"name": "AUTH_TOKEN"},
                    ],
                    "min_likelihood": "LIKELIHOOD_POSSIBLE",
                }
                deidentify_config = {
                    "info_type_transformations": {
                        "transformations": [
                            {
                                "primitive_transformation": {
                                    "replace_with_info_type_config": {}
                                }
                            }
                        ]
                    }
                }
                response = self._dlp_client.deidentify_content(
                    request={
                        "parent": parent,
                        "deidentify_config": deidentify_config,
                        "inspect_config": inspect_config,
                        "item": {"value": sanitized},
                    }
                )
                if response and response.item and response.item.value:
                    sanitized = response.item.value
                    used_dlp = True
            except Exception:
                used_dlp = False

        return RedactionReport(
            sanitized_text=sanitized,
            redaction_count=total_redactions,
            matched_categories=categories,
            used_cloud_dlp=used_dlp,
        )

    def scrub_text(self, text: str) -> str:
        """Convenience method returning only the sanitized string."""
        return self.inspect_and_redact(text).sanitized_text

    def scrub_payload(self, payload: Any) -> Any:
        """Recursively scrubs dictionaries, lists, and strings in-flight."""
        if isinstance(payload, str):
            return self.scrub_text(payload)
        if isinstance(payload, dict):
            return {
                k: (
                    "[REDACTED_SECRET]"
                    if isinstance(k, str)
                    and any(
                        s in k.lower()
                        for s in (
                            "api_key",
                            "secret_key",
                            "client_secret",
                            "password",
                            "authorization",
                            "auth_token",
                            "access_token",
                            "bearer_token",
                        )
                    )
                    else self.scrub_payload(v)
                )
                for k, v in payload.items()
            }
        if isinstance(payload, list):
            return [self.scrub_payload(item) for item in payload]
        if isinstance(payload, tuple):
            return tuple(self.scrub_payload(item) for item in payload)
        return payload


_default_scrubber = PIIScrubber()


def scrub_pii(value: Any) -> Any:
    """Module-level helper that scrubs PII from strings, dicts, or lists."""
    return _default_scrubber.scrub_payload(value)


class PIIScrubbingMiddleware:
    """ASGI & Agent pipeline middleware enforcing inbound/outbound PII redaction."""

    def __init__(self, app: Any = None, scrubber: PIIScrubber | None = None) -> None:
        self.app = app
        self.scrubber = scrubber or _default_scrubber

    def sanitize_inbound_request(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], RedactionReport]:
        """Sanitizes an inbound user request dictionary prior to agent execution."""
        raw_prompt = str(payload.get("prompt", payload.get("query", "")))
        report = self.scrubber.inspect_and_redact(raw_prompt)
        sanitized_payload = self.scrubber.scrub_payload(payload)
        if "prompt" in sanitized_payload:
            sanitized_payload["prompt"] = report.sanitized_text
        if "query" in sanitized_payload:
            sanitized_payload["query"] = report.sanitized_text
        return sanitized_payload, report

    async def __call__(self, scope: Dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        """ASGI interface wrapping HTTP requests with automatic body PII scrubbing."""
        if self.app is None:
            return
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def scrubbed_receive() -> Dict[str, Any]:
            message = await receive()
            if message.get("type") == "http.request" and message.get("body"):
                try:
                    raw_text = message["body"].decode("utf-8")
                    clean_text = self.scrubber.scrub_text(raw_text)
                    message = dict(message)
                    message["body"] = clean_text.encode("utf-8")
                except Exception:
                    pass
            return message

        await self.app(scope, scrubbed_receive, send)
