"""Unit Tests for ArgonLearn Tools, Schemas, Guardrails, Memory, and Observability."""

from __future__ import annotations

import ast
import unittest

from app.memory.compaction import ConversationCompactor, TurnMessage
from app.memory.knowledge_graph import get_knowledge_graph
from app.observability.logging import get_recent_audit_logs
from app.observability.pii_scrubber import PIIScrubber
from app.tools.ingestion_tools import (
    FetchBreakthroughsInput,
    fetch_recent_ai_breakthroughs,
)
from app.tools.knowledge_tools import (
    QueryKnowledgeGraphInput,
    UpdateMasteryStateInput,
    query_learner_knowledge_graph,
    update_learner_mastery_state,
)
from app.tools.simulation_tools import (
    CompileManimInput,
    GenerateInteractiveSimInput,
    SimulationSafetyPolicyPlugin,
    compile_manim_animation,
    generate_interactive_sim_code,
)


class TestToolInterfacesAndSchemas(unittest.TestCase):
    """Verifies Rubric Section 1: Pydantic v2 Schemas, Docstrings, and Guided Error Handling."""

    def test_all_five_tools_have_google_style_docstrings_and_json_schemas(self) -> None:
        tools = [
            fetch_recent_ai_breakthroughs,
            query_learner_knowledge_graph,
            update_learner_mastery_state,
            generate_interactive_sim_code,
            compile_manim_animation,
        ]
        for tool_fn in tools:
            doc = tool_fn.__doc__ or ""
            self.assertIn("Args:", doc, f"Missing Args section in {tool_fn.__name__}")
            self.assertIn("Returns:", doc, f"Missing Returns section in {tool_fn.__name__}")

        schema_models = [
            FetchBreakthroughsInput,
            QueryKnowledgeGraphInput,
            UpdateMasteryStateInput,
            GenerateInteractiveSimInput,
            CompileManimInput,
        ]
        for model_cls in schema_models:
            schema = model_cls.model_json_schema()
            self.assertEqual(schema.get("type"), "object")
            self.assertIn("properties", schema)
            self.assertTrue(len(schema["properties"]) > 0)

    def test_fetch_recent_ai_breakthroughs_success_and_guided_error(self) -> None:
        res = fetch_recent_ai_breakthroughs(
            topic_query="Meta just released Muse: Masked Generative Video Transformer.",
            max_results=2,
            difficulty_filter="Novice",
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["primary_concept_id"], "discrete_parallel_diffusion")
        self.assertGreaterEqual(len(res["breakthroughs"]), 1)

        # Trigger guided error with invalid parameters
        err = fetch_recent_ai_breakthroughs(
            topic_query="x",
            max_results=99,
            difficulty_filter="InvalidTier",
        )
        self.assertEqual(err["status"], "error")
        self.assertIn("error_type", err)
        self.assertIn("raw_message", err)
        self.assertIn("remediation_instruction", err)
        self.assertIn("topic_query", err["remediation_instruction"])

    def test_knowledge_graph_query_and_cycle_guard_error_handling(self) -> None:
        learner_id = "test_learner_unit"
        graph_res = query_learner_knowledge_graph(
            learner_id=learner_id,
            target_concept_id="transformer_architecture",
        )
        self.assertEqual(graph_res["status"], "success")
        self.assertTrue(graph_res["is_valid_dag"])
        self.assertIn("transformer_architecture", graph_res["nodes"])

        # Valid mastery update
        upd = update_learner_mastery_state(
            learner_id=learner_id,
            concept_id="discrete_parallel_diffusion",
            concept_title="Discrete Parallel Diffusion",
            mastery_delta=0.35,
            prerequisites=["transformer_architecture"],
        )
        self.assertEqual(upd["status"], "success")
        self.assertIn("[+1 Mastered:", upd["hud_badge"])

        # Self-cycle must be caught and return structured remediation_instruction
        cycle_err = update_learner_mastery_state(
            learner_id=learner_id,
            concept_id="discrete_parallel_diffusion",
            concept_title="Discrete Parallel Diffusion",
            prerequisites=["discrete_parallel_diffusion"],
        )
        self.assertEqual(cycle_err["status"], "error")
        self.assertIn("remediation_instruction", cycle_err)

    def test_simulation_safety_guardrail_and_hitl_gate(self) -> None:
        # 1. Valid sandboxed simulation succeeds
        sim_ok = generate_interactive_sim_code(
            concept_id="discrete_parallel_diffusion",
            concept_title="Discrete Parallel Diffusion",
            mastery_level="Novice",
            initial_masking_ratio=0.80,
        )
        self.assertEqual(sim_ok["status"], "success")
        self.assertTrue(sim_ok["guardrail_passed"])
        self.assertIn("Masking Ratio: 80%", sim_ok["html_code"])

        # 2. Malicious external script injection is blocked with remediation_instruction
        malicious_html = (
            "<div>Sim</div><script src='https://evil.example.com/xss.js'></script>"
        )
        sim_blocked = generate_interactive_sim_code(
            concept_id="discrete_parallel_diffusion",
            concept_title="Discrete Parallel Diffusion",
            custom_html_override=malicious_html,
        )
        self.assertEqual(sim_blocked["status"], "error")
        self.assertIn("remediation_instruction", sim_blocked)

        # 3. High-stakes Manim execution without human approval triggers HITL gate
        hitl_pause = compile_manim_animation(
            concept_id="discrete_parallel_diffusion",
            scene_class_name="CosineSimilarityClusteringScene",
            execute_render=True,
            human_approved=False,
        )
        self.assertEqual(hitl_pause["status"], "requires_human_approval")
        self.assertTrue(hitl_pause["hitl_gate_triggered"])

        # 4. With human_approved=True, Manim compilation succeeds and passes AST check
        manim_ok = compile_manim_animation(
            concept_id="discrete_parallel_diffusion",
            scene_class_name="CosineSimilarityClusteringScene",
            execute_render=True,
            human_approved=True,
        )
        self.assertEqual(manim_ok["status"], "success")
        self.assertTrue(manim_ok["ast_validated"])
        ast.parse(manim_ok["manim_python_code"])

        # 5. Forbidden Python import in Manim code is rejected by AST policy plugin
        unsafe_manim = "import os\nfrom manim import Scene\nclass BadScene(Scene):\n    pass\n"
        manim_blocked = compile_manim_animation(
            concept_id="discrete_parallel_diffusion",
            scene_class_name="BadScene",
            custom_manim_code=unsafe_manim,
        )
        self.assertEqual(manim_blocked["status"], "error")
        self.assertIn("FORBIDDEN_IMPORT:os", manim_blocked["raw_message"])

    def test_pii_scrubber_and_history_compaction(self) -> None:
        scrubber = PIIScrubber()
        dirty = (
            "Contact researcher@google.com (Badge EMP-884920) with token "
            "Bearer mock_session_cred_12345 or 415-555-0199."
        )
        report = scrubber.inspect_and_redact(dirty)
        self.assertNotIn("researcher@google.com", report.sanitized_text)
        self.assertNotIn("EMP-884920", report.sanitized_text)
        self.assertNotIn("mock_session_cred_12345", report.sanitized_text)
        self.assertNotIn("415-555-0199", report.sanitized_text)
        self.assertGreaterEqual(report.redaction_count, 4)

        compactor = ConversationCompactor(max_context_tokens=200, sliding_window_turns=4)
        turns = [
            TurnMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Turn {i}: Detailed discussion of masked token prediction and cosine schedule " * 3,
                concept_id="discrete_parallel_diffusion",
                mastery_level="Novice",
            )
            for i in range(10)
        ]
        result = compactor.compact_history(turns, session_id="test_compact")
        self.assertTrue(result.was_compacted)
        self.assertLessEqual(result.retained_turn_count, 4)
        self.assertLess(result.compacted_tokens, result.original_tokens)
        self.assertTrue(len(result.compacted_summary) > 10)

        # Verify structured Intent vs Outcome logs were captured
        logs = get_recent_audit_logs(limit=30)
        events = {entry["event"] for entry in logs}
        self.assertIn("agent_intent", events)
        self.assertIn("tool_outcome", events)


if __name__ == "__main__":
    unittest.main()
