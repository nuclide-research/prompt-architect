"""Regression tests for PromptArchitect.

Pure stdlib (`unittest`) so the package stays dependency-free:

    python -m unittest discover -s prompt_architect/tests -v

Covers the public API contract, the three fixes (goal-verb recall, goal-threaded
instruction fallback, format-aware cot variant, prompt-eng-topical grounding),
JSON-serializability, and the RAG layer.
"""

import json
import tempfile
import unittest

from prompt_architect import BookCorpus, PromptArchitect
from prompt_architect.architect import _PROMPT_ENG_BOOKS


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


class TestOutputContractFix(unittest.TestCase):
    """Fix #2: the base OUTPUT contract must agree with the merged constraints,
    including a format inferred from the prompt text (not just context)."""

    def setUp(self):
        self.a = PromptArchitect()

    def test_base_output_contract_agrees_with_prompt_inferred_format(self):
        # No format in context; the prompt text alone says JSON.
        ctx = _ctx(constraints={})
        r = self.a.refine("Extract the fields and return the result as json.", ctx, ground=False)
        spec = r["prompt_spec"]
        self.assertEqual(spec["constraints"].get("format"), "json")
        oc = spec["output_contract"].lower()
        self.assertIn("json", oc)
        # The markdown default must not leak through when JSON was requested.
        self.assertNotIn("markdown", oc)

    def test_fewshot_inherits_corrected_output_contract(self):
        ctx = _ctx(constraints={})
        v = self.a.generate_variants("Extract the fields and return the result as json.", ctx)
        fs = next(x for x in v["variants"] if x["id"] == "fewshot_skeleton")
        self.assertIn("json", fs["prompt_spec"]["output_contract"].lower())


class TestGroundingQueries(unittest.TestCase):
    """Fix #3 internals: goal query is prompt-eng-scoped; issues stay corpus-wide
    and are capped; the placeholder goal is skipped."""

    def _spec(self, goal):
        return {
            "role": "r", "goal": goal, "inputs_contract": "i", "instructions": [],
            "output_contract": "o", "constraints": {}, "quality_checks": [], "notes": [],
        }

    def _analysis(self, issues):
        return {"components": {}, "issues": issues, "risk_assessment": [], "suggested_improvements": []}

    def test_goal_query_is_prompt_eng_scoped(self):
        qs = PromptArchitect._grounding_queries(self._spec("flag secrets in json"), self._analysis([]))
        self.assertEqual(len(qs), 1)
        q, scope = qs[0]
        self.assertIn("prompt engineering", q.lower())
        self.assertEqual(tuple(scope), _PROMPT_ENG_BOOKS)

    def test_placeholder_goal_query_is_skipped(self):
        qs = PromptArchitect._grounding_queries(
            self._spec("Complete the requested task accurately."),
            self._analysis(["issue one", "issue two"]),
        )
        self.assertTrue(all("prompt engineering" not in q.lower() for q, _ in qs))
        self.assertTrue(all(scope is None for _, scope in qs))

    def test_issue_queries_capped_at_three_and_corpus_wide(self):
        qs = PromptArchitect._grounding_queries(
            self._spec("Complete the requested task accurately."),
            self._analysis([f"issue {i}" for i in range(5)]),
        )
        self.assertEqual(len(qs), 3)  # placeholder goal skipped, 5 issues -> 3
        self.assertTrue(all(scope is None for _, scope in qs))

    def test_refine_goal_grounding_passages_are_prompt_eng_books(self):
        r = self.a.refine("look at the json and flag secrets", _ctx())  # noqa: F841
        first = r["grounding"][0]
        self.assertTrue(first["passages"])
        self.assertTrue(all(p["book"] in _PROMPT_ENG_BOOKS for p in first["passages"]))

    def setUp(self):
        self.a = PromptArchitect()


class TestCorpusScope(unittest.TestCase):
    """BookCorpus.search book scope: single slug, iterable, empty, and BM25 order."""

    @classmethod
    def setUpClass(cls):
        cls.c = BookCorpus().build()

    def test_iterable_scope_limits_to_given_books(self):
        slugs = list(_PROMPT_ENG_BOOKS)
        hits = self.c.search("prompt instruction format", k=8, book=slugs)
        self.assertTrue(hits)
        self.assertTrue(set(h["book"] for h in hits).issubset(set(slugs)))

    def test_empty_scope_returns_empty_not_corpus_wide(self):
        self.assertEqual(self.c.search("prompt", k=5, book=[]), [])

    def test_single_slug_scope_still_works(self):
        hits = self.c.search("context window tokens", k=3, book="farris-how-llms-work")
        self.assertTrue(hits)
        self.assertEqual({h["book"] for h in hits}, {"farris-how-llms-work"})

    def test_stopword_only_query_returns_empty(self):
        self.assertEqual(self.c.search("the and of to with", k=5), [])

    def test_scores_are_descending(self):
        hits = self.c.search("json output schema guardrails", k=8)
        scores = [h["score"] for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_cache_round_trip_is_identical(self):
        # A fresh corpus over the same dir loads the pickle cache and must return
        # byte-identical results to the in-memory build.
        warm = BookCorpus().build()
        q = "few-shot examples improve output"
        a, b = self.c.search(q, k=5), warm.search(q, k=5)
        self.assertEqual(a, b)


class TestDegradation(unittest.TestCase):
    """Missing corpus must degrade, not raise, when grounding is requested."""

    def test_missing_books_dir_yields_empty_grounding(self):
        with tempfile.TemporaryDirectory() as empty:
            a = PromptArchitect(books_dir=empty)
            r = a.refine("summarize findings", _ctx(), ground=True)
            self.assertEqual(r["grounding"], [])  # best-effort, no FileNotFoundError


class TestExactnessAndEdges(unittest.TestCase):
    def setUp(self):
        self.a = PromptArchitect()

    def test_reanalysis_instruction_count_matches_spec(self):
        # Round-trip fidelity: re-analyzing the rendered prompt recovers exactly the
        # instruction steps the spec declared (no over- or under-count).
        r = self.a.refine("summarize findings", _ctx(), ground=False)
        spec_n = len(r["prompt_spec"]["instructions"])
        after_n = len(r["analysis_after"]["components"]["instructions"])
        self.assertEqual(after_n, spec_n)

    def test_concise_strict_defaults_to_markdown(self):
        v = self.a.generate_variants("summarize findings", _ctx(constraints={}))
        cs = next(x for x in v["variants"] if x["id"] == "concise_strict")
        oc = cs["prompt_spec"]["output_contract"].lower()
        self.assertIn("markdown", oc)
        self.assertNotIn("json", oc)

    def test_context_window_overflow_is_flagged(self):
        # A long prompt against a tiny window must raise the context-window issue.
        an = self.a.analyze("word " * 200, _ctx(model_context_tokens=10))
        self.assertTrue(any("context window" in i.lower() for i in an["issues"]))

    def test_context_window_ok_when_within_budget(self):
        an = self.a.analyze("summarize this", _ctx(model_context_tokens=8000))
        self.assertFalse(any("context window" in i.lower() for i in an["issues"]))


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
