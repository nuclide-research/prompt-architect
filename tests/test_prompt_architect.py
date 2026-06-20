"""Regression tests for PromptArchitect.

Pure stdlib (`unittest`) so the package stays dependency-free:

    python -m unittest discover -s prompt_architect/tests -v

Covers the public API contract, the three fixes (goal-verb recall, goal-threaded
instruction fallback, format-aware cot variant, prompt-eng-topical grounding),
JSON-serializability, and the RAG layer.
"""

import json
import unittest

from prompt_architect import BookCorpus, PromptArchitect


def _ctx(**over):
    base = {
        "audience": "data engineers",
        "model_context_tokens": 8000,
        "constraints": {"format": "json", "tone": "neutral"},
        "risk_points": ["hallucination", "security", "compliance"],
        "domain": "data_pipeline_security",
        "inputs_type": "mixed",
    }
    base.update(over)
    return base


class TestAnalyze(unittest.TestCase):
    def setUp(self):
        self.a = PromptArchitect()

    def test_analyze_shape(self):
        an = self.a.analyze("do a thing", _ctx())
        self.assertEqual(
            set(an), {"components", "issues", "risk_assessment", "suggested_improvements"}
        )

    def test_detects_all_issues_on_sloppy_prompt(self):
        an = self.a.analyze("look at the json and pull out anything sensitive", _ctx())
        joined = " ".join(an["issues"]).lower()
        for needle in ("role", "format", "instructions", "quality check"):
            self.assertIn(needle, joined)

    def test_security_risk_fires_on_code_request(self):
        an = self.a.analyze("write some code to flag records", _ctx())
        self.assertTrue(any("security" in r.lower() for r in an["risk_assessment"]))

    def test_security_risk_suppressed_when_bounded(self):
        an = self.a.analyze(
            "write read-only code, do not execute anything", _ctx()
        )
        self.assertFalse(any("security risk" in r.lower() for r in an["risk_assessment"]))


class TestGoalExtraction(unittest.TestCase):
    """Fix #1: 'look at ...' and friends must register as the goal."""

    def setUp(self):
        self.a = PromptArchitect()

    def test_look_verb_now_recognized(self):
        an = self.a.analyze("look at the data and find issues", _ctx(domain=""))
        self.assertIsNotNone(an["components"]["goal"])
        self.assertIn("look at the data", an["components"]["goal"].lower())

    def test_pull_verb_recognized(self):
        an = self.a.analyze("pull out the sensitive fields", _ctx(domain=""))
        self.assertIsNotNone(an["components"]["goal"])

    def test_word_boundary_no_prefix_false_match(self):
        # A bare opener fires; a longer word that merely starts with a verb must not.
        ctx = _ctx(domain="")
        self.assertIsNotNone(self.a.analyze("scan the logs", ctx)["components"]["goal"])
        for text in ("Scanning the logs revealed a leak",
                     "Finding nemo is a movie",
                     "Designers love whitespace"):
            self.assertIsNone(
                self.a.analyze(text, ctx)["components"]["goal"], msg=text
            )

    def test_instruction_fallback_uses_goal_not_literal_the_task(self):
        # Fix #1b: with no detectable goal, step 2 must not read "achieve: the task".
        r = self.a.refine("xyzzy plugh", _ctx(), ground=False)
        step2 = r["prompt_spec"]["instructions"][1]
        self.assertNotIn("achieve: the task", step2)

    def test_instruction_fallback_embeds_real_goal(self):
        r = self.a.refine("look at the json and flag secrets", _ctx(), ground=False)
        steps = " ".join(r["prompt_spec"]["instructions"]).lower()
        self.assertIn("look at the json", steps)


class TestRefine(unittest.TestCase):
    def setUp(self):
        self.a = PromptArchitect()

    def test_refine_shape(self):
        r = self.a.refine("summarize findings", _ctx(), ground=False)
        self.assertEqual(
            set(r),
            {
                "prompt_spec",
                "rendered_prompt",
                "analysis_before",
                "analysis_after",
                "change_log",
                "grounding",
            },
        )

    def test_rendered_has_all_sections(self):
        r = self.a.refine("summarize findings", _ctx(), ground=False)
        for section in ("[ROLE]", "[GOAL]", "[INPUTS]", "[INSTRUCTIONS]",
                        "[OUTPUT]", "[CONSTRAINTS]", "[QUALITY CHECK]"):
            self.assertIn(section, r["rendered_prompt"])

    def test_change_log_reports_honest_instruction_count(self):
        # The section-aware extractor must not count constraint/quality bullets.
        r = self.a.refine("summarize findings", _ctx(), ground=False)
        after = r["analysis_after"]["components"]["instructions"]
        self.assertLessEqual(len(after), 7)

    def test_ground_false_yields_empty_grounding(self):
        r = self.a.refine("summarize findings", _ctx(), ground=False)
        self.assertEqual(r["grounding"], [])


class TestVariants(unittest.TestCase):
    def setUp(self):
        self.a = PromptArchitect()

    def test_three_variant_ids(self):
        v = self.a.generate_variants("summarize findings", _ctx())
        self.assertEqual([x["id"] for x in v["variants"]], ["cot", "concise_strict", "fewshot_skeleton"])

    def test_cot_json_has_no_markdown_heading_contradiction(self):
        # Fix #2: JSON format must not get '## Reasoning' markdown headings.
        v = self.a.generate_variants("summarize findings", _ctx(constraints={"format": "json"}))
        cot = next(x for x in v["variants"] if x["id"] == "cot")
        oc = cot["prompt_spec"]["output_contract"]
        self.assertNotIn("##", oc)
        self.assertIn("reasoning", oc.lower())
        self.assertIn("json", oc.lower())

    def test_cot_markdown_keeps_section_headings(self):
        v = self.a.generate_variants("summarize findings", _ctx(constraints={"format": "markdown"}))
        cot = next(x for x in v["variants"] if x["id"] == "cot")
        self.assertIn("## Reasoning", cot["prompt_spec"]["output_contract"])

    def test_variant_does_not_mutate_base(self):
        v = self.a.generate_variants("summarize findings", _ctx())
        base_oc = v["base_spec"]["output_contract"]
        # Mutating a variant spec must not have leaked back into base_spec.
        self.assertNotIn("reasoning", base_oc.lower().replace("reasoning risk", ""))

    def test_n_clamped(self):
        v = self.a.generate_variants("summarize findings", _ctx(), n=99)
        self.assertEqual(len(v["variants"]), 3)
        v0 = self.a.generate_variants("summarize findings", _ctx(), n=0)
        self.assertEqual(len(v0["variants"]), 0)


class TestRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = PromptArchitect()

    def test_ground_returns_relevant_passages(self):
        hits = self.a.ground("how to specify a json output schema", k=3)
        self.assertTrue(hits)
        self.assertTrue(all(h["score"] > 0 for h in hits))
        self.assertEqual(set(hits[0]), {"book", "chunk_id", "score", "text"})

    def test_ground_book_scope(self):
        hits = self.a.ground("context window tokens", k=2, book="farris-how-llms-work")
        self.assertTrue(hits)
        self.assertEqual({h["book"] for h in hits}, {"farris-how-llms-work"})

    def test_empty_query_returns_empty(self):
        self.assertEqual(self.a.ground("   ", k=3), [])

    def test_corpus_chunk_count_stable(self):
        c = BookCorpus().build()
        self.assertGreater(c.num_chunks, 1000)
        self.assertEqual(len(c.books()), 7)

    def test_refine_grounding_is_prompt_eng_topical(self):
        # Fix #3: grounding queries should be framed for prompt engineering.
        r = self.a.refine("look at the json and flag secrets", _ctx())
        self.assertTrue(r["grounding"])
        first_q = r["grounding"][0]["query"].lower()
        self.assertIn("prompt", first_q)


class TestJSONSerializable(unittest.TestCase):
    def setUp(self):
        self.a = PromptArchitect()

    def test_all_returns_json_safe(self):
        ctx = _ctx()
        p = "look at the json and pull out anything sensitive and write code to flag it"
        json.dumps(self.a.analyze(p, ctx))
        json.dumps(self.a.refine(p, ctx))
        json.dumps(self.a.generate_variants(p, ctx))
        json.dumps(self.a.ground("json schema", k=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
