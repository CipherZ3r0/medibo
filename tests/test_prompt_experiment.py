"""Phase 9 — two-stage prompt experiment + deterministic rubric unit tests.

Pure-function tests: no Elasticsearch, no LLM, no network.  They pin the
deterministic scoring functions used by ``evaluation/prompt_experiment.py``,
prove that the Phase 9 prompt variants leave the historic baseline prompt
byte-for-byte unchanged, and cover the two-stage design (sample selection,
run caching/migration, token budgets, stage scoring, recommendation).
"""

import unittest

from domain.models import SearchResult
from evaluation.prompt_experiment import (
    BASELINE_ARM,
    GROUNDED_METRICS,
    LIMITS,
    METRIC_RUBRIC,
    STAGE1_CATEGORIES,
    STAGE1_QIDS,
    STAGE1_SELECTION,
    STAGE2_QIDS,
    USER_QUERY,
    VARIANT_ARMS,
    TokenLedger,
    aggregate,
    citation_markers,
    compute_stage1_report,
    decide_recommendation,
    estimate_output_tokens,
    migrate_legacy_runs,
    needs_care_guidance,
    needs_symptom_handling,
    run_key,
    score_answer,
    score_citation_correctness,
    score_completeness,
    score_fallback,
    score_red_flag_handling,
    score_source_adherence,
    score_symptom_vs_diagnosis,
    score_unsupported_claims,
    score_verbosity,
    source_mentions,
    verbatim_leak,
)
from rag.evidence import UNTRUSTED_ANSWER_WARNING
from rag.prompt_builders import BasePromptBuilder, TemplatePromptBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_result(text="Dengue is a viral infection transmitted by Aedes mosquitoes.", **meta):
    metadata = {
        "source": "medlineplus",
        "title": "Dengue",
        "dataset_type": "default",
    }
    metadata.update(meta)
    return SearchResult(
        chunk_id="c1",
        doc_id="d1",
        text=text,
        file_name="medlineplus-dengue.json",
        source_file="/data/medlineplus-dengue.json",
        file_type=".json",
        page_number=4,
        score=1.0,
        metadata=metadata,
    )


class PromptVariantTests(unittest.TestCase):
    def test_baseline_prompt_is_unchanged(self):
        """variant=None → the historic style prompt, byte for byte.

        This is the experiment's baseline arm (the pre-adoption production
        prompt) and must never drift, even though production now defaults to
        variant_structured_brief.
        """
        res = make_result()
        built = TemplatePromptBuilder(variant=None).build_prompt(
            "What is dengue?", [res], style="detailed"
        )

        expected_system = (
            TemplatePromptBuilder.SYSTEM_PROMPTS["detailed"]
            + TemplatePromptBuilder.GROUNDING_RULE
        )
        expected_context = TemplatePromptBuilder.CONTEXT_TEMPLATE.format(
            index=1,
            file_name="medlineplus-dengue.json",
            source_type="Trusted medical source (Medibo knowledge base)",
            page_info="Page: 4",
            text=res.text,
        )
        expected_user = TemplatePromptBuilder.USER_TEMPLATE.format(
            context=expected_context, question="What is dengue?"
        )
        self.assertEqual(
            built,
            [
                {"role": "system", "content": expected_system},
                {"role": "user", "content": expected_user},
            ],
        )

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            TemplatePromptBuilder(variant="does_not_exist")

    def test_three_variants_defined_and_distinct(self):
        self.assertEqual(
            sorted(TemplatePromptBuilder.PROMPT_VARIANTS),
            ["question_first", "structured_brief", "triage_concise"],
        )
        systems = [
            TemplatePromptBuilder(variant=name).build_prompt("q", [])[0]["content"]
            for name in TemplatePromptBuilder.PROMPT_VARIANTS
        ]
        self.assertEqual(len(set(systems)), 3)
        baseline = TemplatePromptBuilder(variant=None).build_prompt("q", [])[0]["content"]
        for system in systems:
            self.assertNotEqual(system, baseline)

    def test_adopted_production_prompt_is_structured_brief(self):
        """Phase 9 adoption: the default builder emits variant_structured_brief."""
        from rag.prompt_builders import ADOPTED_GROUNDED_VARIANT

        self.assertEqual(ADOPTED_GROUNDED_VARIANT, "structured_brief")
        res = make_result()
        default = TemplatePromptBuilder().build_prompt("What is dengue?", [res])
        adopted = TemplatePromptBuilder(variant="structured_brief").build_prompt(
            "What is dengue?", [res]
        )
        self.assertEqual(default, adopted)
        system = default[0]["content"]
        self.assertTrue(
            system.startswith(
                TemplatePromptBuilder.PROMPT_VARIANTS["structured_brief"]["system"]
            )
        )
        self.assertIn("**Seek care**", system)          # adopted format sections
        self.assertIn("**Details**", system)
        self.assertTrue(system.endswith(TemplatePromptBuilder.GROUNDING_RULE))
        # provenance context template + trusted/user source labels preserved
        self.assertIn("Issued by:", default[1]["content"])
        self.assertIn("Trusted medical source (Medibo knowledge base)", default[1]["content"])
        # fallback prompt unchanged (requirement: fallback stays as-is)
        self.assertEqual(
            TemplatePromptBuilder().build_fallback_prompt("q")[0]["content"],
            BasePromptBuilder.FALLBACK_SYSTEM_PROMPT,
        )

    def test_variants_keep_grounding_rule_and_citation_contract(self):
        for name in TemplatePromptBuilder.PROMPT_VARIANTS:
            system = TemplatePromptBuilder(variant=name).build_prompt("q", [])[0]["content"]
            self.assertTrue(system.endswith(TemplatePromptBuilder.GROUNDING_RULE), name)
            self.assertIn("[", system, name)          # citation markers required
            self.assertIn("diagnos", system.lower(), name)  # symptoms ≠ diagnosis

    def test_variant_uses_provenance_context_fields(self):
        res = make_result(
            authority="ICMR",
            region="India",
            source_url="https://mohfw.gov.in/protocol.pdf",
            section="Clinical management",
        )
        user = TemplatePromptBuilder(variant="structured_brief").build_prompt(
            "What is dengue?", [res]
        )[1]["content"]
        self.assertIn("Issued by: ICMR", user)
        self.assertIn("Region: India", user)
        self.assertIn("Source URL: https://mohfw.gov.in/protocol.pdf", user)
        self.assertIn("Section: Clinical management", user)

    def test_fallback_prompt_shared_and_context_free(self):
        q = "Who wrote Pride and Prejudice?"
        for builder in [
            TemplatePromptBuilder(),
            TemplatePromptBuilder(variant="triage_concise"),
        ]:
            messages = builder.build_fallback_prompt(q)
            self.assertEqual(messages[0]["content"], BasePromptBuilder.FALLBACK_SYSTEM_PROMPT)
            self.assertEqual(messages[1]["content"], q)
            self.assertNotIn("Context Document", str(messages))

    def test_stage_arm_config_is_valid(self):
        self.assertEqual(len(VARIANT_ARMS), 3)
        for arm in VARIANT_ARMS:
            self.assertEqual(arm["kind"], "variant")
            self.assertIn(arm["value"], TemplatePromptBuilder.PROMPT_VARIANTS)
        self.assertEqual(
            BASELINE_ARM,
            {"name": "baseline_detailed", "kind": "style", "value": "detailed"},
        )
        self.assertIn("fallback", METRIC_RUBRIC)


# ---------------------------------------------------------------------------
# Rubric helpers
# ---------------------------------------------------------------------------

class CitationAndSourceTests(unittest.TestCase):
    def test_citation_markers_extracted_in_order(self):
        self.assertEqual(citation_markers("Fever [1]. Bleeding [3] and [2]."), [1, 3, 2])
        self.assertEqual(citation_markers("no markers here"), [])
        self.assertEqual(citation_markers("array[0] and [x] ignored"), [0])

    def test_source_mentions_finds_urls_files_and_orgs(self):
        text = "See https://example.org/a.pdf and medlineplus-dengue.json per ICMR and WHO."
        mentions = source_mentions(text)
        self.assertIn("https://example.org/a.pdf", mentions)
        self.assertIn("medlineplus-dengue.json", mentions)
        self.assertIn("icmr", mentions)
        self.assertIn("WHO", mentions)

    def test_adherence_penalises_invented_source(self):
        context = ["Dengue is spread by mosquitoes."]
        good = "Dengue is spread by mosquitoes [1] (https://kb.example/doc1)."
        good_citations = [{"source_url": "https://kb.example/doc1"}]
        self.assertEqual(score_source_adherence(good, good_citations, context), 1.0)
        bad = "See https://invented.example/study."
        self.assertLess(score_source_adherence(bad, good_citations, context), 1.0)

    def test_adherence_flags_verbatim_leak(self):
        context = ["Dengue fever is caused by the dengue virus and is transmitted to humans."]
        answer = "Dengue fever is caused by the dengue virus and is transmitted to humans."
        self.assertTrue(verbatim_leak(answer, context))
        self.assertLessEqual(score_source_adherence(answer, [], context), 0.5)

    def test_adherence_no_sources_named_is_perfect(self):
        self.assertEqual(score_source_adherence("Fluids and rest help.", [], []), 1.0)


class ClaimAndStyleTests(unittest.TestCase):
    def test_echoed_context_is_fully_supported(self):
        context = ["Dengue is a viral infection transmitted by Aedes mosquitoes."]
        self.assertEqual(score_unsupported_claims(context[0], context), 1.0)

    def test_invented_medical_claim_is_unsupported(self):
        context = ["Dengue is a viral infection transmitted by Aedes mosquitoes."]
        answer = "Dengue is caused by drinking cold water at night."
        self.assertLess(score_unsupported_claims(answer, context), 0.5)

    def test_citation_correctness_scores(self):
        self.assertIsNone(score_citation_correctness("answer", 0))     # fallback run
        self.assertEqual(score_citation_correctness("no markers", 2), 0.0)
        self.assertEqual(score_citation_correctness("[1] [4]", 3), 0.4167)
        self.assertEqual(score_citation_correctness("[1] [2] [3]", 3), 1.0)

    def test_verbosity_bands(self):
        short = " ".join(["word"] * 30)
        ideal = " ".join(["word"] * 100)
        long = " ".join(["word"] * 300)
        self.assertEqual(score_verbosity(""), 0.0)
        self.assertEqual(score_verbosity(short), 0.6)
        self.assertEqual(score_verbosity(ideal), 1.0)
        self.assertEqual(score_verbosity(long), 0.6)

    def test_completeness_prefers_answering_the_question(self):
        question = "Dengue symptoms and when to seek care"
        strong = ("Fever, headache and pain behind the eyes are common dengue symptoms. "
                  "Warning signs include bleeding and severe abdominal pain. Seek care "
                  "immediately if these appear, as early monitoring improves outcomes.")
        weak = "Rest and drink plenty of fluids to stay comfortable each day."
        self.assertGreater(score_completeness(question, strong), score_completeness(question, weak))

    def test_red_flag_handling_gating(self):
        q_warn = "What are the warning signs of a heart attack?"
        self.assertTrue(needs_care_guidance(q_warn))
        self.assertEqual(score_red_flag_handling(q_warn, "Seek emergency care now."), 1.0)
        self.assertEqual(score_red_flag_handling(q_warn, "There are several signs."), 0.0)
        q_plain = "What is dengue?"
        self.assertFalse(needs_care_guidance(q_plain))
        self.assertIsNone(score_red_flag_handling(q_plain, "any answer"))

    def test_symptom_vs_diagnosis_scoring(self):
        q = "What are dengue symptoms?"
        self.assertTrue(needs_symptom_handling(q))
        self.assertEqual(score_symptom_vs_diagnosis(q, "You have dengue fever."), 0.0)
        # conditional mentions are not diagnoses
        self.assertGreater(
            score_symptom_vs_diagnosis(q, "If you have a fever, rest and drink fluids."), 0.0
        )
        # caveat lifts 0.5 → 1.0
        self.assertEqual(
            score_symptom_vs_diagnosis(q, "Fever and rash; diagnosis is a clinician's job."), 1.0
        )
        self.assertIsNone(score_symptom_vs_diagnosis("What is rabies?", "answer"))


class FallbackAndAggregateTests(unittest.TestCase):
    def test_fallback_perfect_answer(self):
        model = ("I do not have any documents about that in the knowledge base. "
                 "From general knowledge, the answer is four.")
        final = f"{UNTRUSTED_ANSWER_WARNING}\n\n{model}"
        checks, score = score_fallback(model, final)
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual(score, 1.0)

    def test_fallback_rejects_fabricated_citations_and_sources(self):
        model = "According to the WHO source [1], the answer is four."
        final = f"{UNTRUSTED_ANSWER_WARNING}\n\n{model}"
        checks, score = score_fallback(model, final)
        self.assertTrue(checks["warning_prefixed"])
        self.assertFalse(checks["no_citation_markers"])
        self.assertFalse(checks["no_source_attribution"])
        self.assertLess(score, 1.0)

    def test_fallback_requires_warning_prefix(self):
        checks, _ = score_fallback("From general knowledge: four.", "From general knowledge: four.")
        self.assertFalse(checks["warning_prefixed"])

    def test_aggregate_means(self):
        records = [
            {"metrics": {"source_adherence": 1.0, "unsupported_claims": 1.0,
                         "citation_correctness": 0.5, "answer_completeness": 0.8,
                         "verbosity": 1.0, "red_flag_handling": None,
                         "symptom_vs_diagnosis": None, "fallback": None}},
            {"metrics": {"source_adherence": 0.5, "unsupported_claims": 1.0,
                         "citation_correctness": None, "answer_completeness": 0.6,
                         "verbosity": 1.0, "red_flag_handling": 1.0,
                         "symptom_vs_diagnosis": 0.5, "fallback": None}},
            {"metrics": {"source_adherence": None, "unsupported_claims": None,
                         "citation_correctness": None, "answer_completeness": None,
                         "verbosity": None, "red_flag_handling": None,
                         "symptom_vs_diagnosis": None, "fallback": 0.75}},
        ]
        agg = aggregate(records)
        self.assertEqual(agg["metric_means"]["source_adherence"], 0.75)
        self.assertEqual(agg["metric_applicable_n"]["citation_correctness"], 1)
        self.assertEqual(agg["metric_applicable_n"]["fallback"], 1)
        self.assertIsNotNone(agg["grounded_mean"])
        self.assertEqual(agg["fallback_mean"], 0.75)
        self.assertIsNotNone(agg["overall"])

    def test_score_answer_returns_every_metric(self):
        metrics = score_answer(
            "What are dengue symptoms?",
            "Fever and pain [1]. Diagnosis is a clinician's job.",
            citations=[],
            context_texts=["Dengue causes fever and body pain."],
        )
        self.assertEqual(
            set(metrics),
            {"source_adherence", "unsupported_claims", "citation_correctness",
             "answer_completeness", "verbosity", "red_flag_handling",
             "symptom_vs_diagnosis", "fallback"},
        )
        self.assertIsNone(metrics["fallback"])
        self.assertEqual(metrics["symptom_vs_diagnosis"], 1.0)


class TwoStageDesignTests(unittest.TestCase):
    """Sample selection, call budgets, caching/migration, ledger, decisions."""

    def setUp(self):
        self.queries = [{"query": f"question {i}"} for i in range(1, 29)]

    def test_stage1_sample_covers_required_categories(self):
        self.assertEqual(len(STAGE1_SELECTION), 12)
        required = {
            "symptoms", "disease", "indian_guidance", "basic_care",
            "warning_signs", "ambiguous_symptoms", "user_only_evidence",
            "no_evidence_fallback",
        }
        self.assertEqual(set(STAGE1_CATEGORIES), required)
        int_qids = [q for q in STAGE1_QIDS if isinstance(q, int)]
        self.assertEqual(len(int_qids), 11)
        self.assertTrue(all(1 <= q <= 28 for q in int_qids))
        self.assertIn("u1", STAGE1_QIDS)
        self.assertEqual(USER_QUERY["qid"], "u1")
        self.assertEqual(len(STAGE2_QIDS), 28)

    def test_call_budgets_match_the_brief(self):
        self.assertEqual(len(STAGE1_QIDS) * len(VARIANT_ARMS), 36)
        self.assertEqual(LIMITS["stage1_max_llm_calls"], 36)
        self.assertEqual(LIMITS["stage2_max_llm_calls"], 56)
        self.assertEqual(LIMITS["experiment_target_tokens"], 180000)
        self.assertEqual(LIMITS["daily_token_budget"], 200000)
        self.assertEqual(LIMITS["tpm"], 8000)
        self.assertEqual(LIMITS["tpd"], 200000)

    def test_run_key_formats(self):
        self.assertEqual(run_key("variant_x", 7), "variant_x:q7")
        self.assertEqual(run_key("variant_x", "u1"), "variant_x:u1")

    def test_migration_keeps_standard_queries_and_drops_probes(self):
        legacy = {
            "baseline_detailed:5": {
                "arm": "baseline_detailed", "query_index": 5,
                "prompt_tokens": 100, "answer_words": 10,
            },
            "baseline_detailed:fb1": {
                "arm": "baseline_detailed", "query_index": "fb1",
            },
            "variant_triage_concise:2": {
                "arm": "variant_triage_concise", "query_index": 2,
            },
        }
        migrated = migrate_legacy_runs(legacy, "qwen/qwen3.8-27b", 0.0)
        self.assertIn("baseline_detailed:q5", migrated)
        self.assertIn("variant_triage_concise:q2", migrated)
        self.assertNotIn("baseline_detailed:fb1", migrated)
        self.assertNotIn("baseline_detailed:5", migrated)
        self.assertTrue(migrated["baseline_detailed:q5"]["cached"])
        self.assertEqual(
            migrate_legacy_runs(legacy, "some/other-model", 0.0), {}
        )

    def test_token_ledger_stops_before_experiment_target(self):
        ledger = TokenLedger(
            prior_today=1000, experiment_limit=10000,
            daily_limit=10 ** 9, buffer=0,
        )
        self.assertTrue(ledger.can_call())
        ledger.add_new(4000, 1000)
        ledger.add_new(4000, 1000)
        self.assertFalse(ledger.can_call())
        self.assertEqual(ledger.stopped_reason, "experiment_token_target_exceeded")
        self.assertFalse(ledger.can_call())  # stays stopped

    def test_token_ledger_stops_before_daily_budget(self):
        ledger = TokenLedger(
            prior_today=190000, experiment_limit=10 ** 9,
            daily_limit=200000, buffer=4000,
        )
        self.assertTrue(ledger.can_call())
        ledger.add_new(10000, 0)
        self.assertFalse(ledger.can_call())
        self.assertEqual(ledger.stopped_reason, "daily_token_budget_exceeded")

    def test_token_ledger_accounts_experiment_usage(self):
        ledger = TokenLedger(
            prior_today=0, experiment_limit=10 ** 9,
            daily_limit=10 ** 9, buffer=0,
        )
        ledger.add_new(800, 400)
        ledger.add_cached(900, 300)
        self.assertEqual(ledger.experiment_total(), 2400)
        self.assertEqual(ledger.today_total(), 1200)  # cached spend was earlier
        self.assertEqual(ledger.new_calls, 1)
        self.assertEqual(ledger.cached_calls, 1)
        payload = ledger.as_dict()
        self.assertEqual(payload["experiment_total_tokens"], 2400)
        self.assertIsNone(payload["stopped_reason"])

    def test_estimate_output_tokens(self):
        self.assertEqual(estimate_output_tokens(130), 100)
        self.assertEqual(estimate_output_tokens(0), 0)

    def test_stage1_report_requires_every_run_then_picks_winner(self):
        self.assertIsNone(compute_stage1_report({}, self.queries))
        scores = {
            "variant_structured_brief": 0.7,
            "variant_question_first": 0.9,
            "variant_triage_concise": 0.8,
        }
        runs = {}
        for arm in VARIANT_ARMS:
            for qid in STAGE1_QIDS:
                runs[run_key(arm["name"], qid)] = {
                    "arm": arm["name"],
                    "qid": qid,
                    "grounded": True,
                    "cached": True,
                    "answer_words": 80,
                    "metrics": {
                        m: scores[arm["name"]] for m in GROUNDED_METRICS
                    },
                }
        report = compute_stage1_report(runs, self.queries)
        self.assertEqual(report["n_queries"], 12)
        self.assertEqual(report["winner"], "variant_question_first")
        self.assertEqual(report["winner_overall"], 0.9)
        self.assertEqual(report["margin"], 0.1)  # vs runner-up (0.8)

    def test_recommendation_wording(self):
        stage2 = {
            "winner_arm": "variant_question_first",
            "winner_overall": 0.8,
            "baseline_overall": 0.75,
            "delta": 0.05,
        }
        self.assertIn("adopt variant_question_first", decide_recommendation(stage2))
        small = dict(stage2, winner_overall=0.755, baseline_overall=0.75, delta=0.005)
        self.assertIn("keep current prompt", decide_recommendation(small))
        self.assertIn("incomplete", decide_recommendation(None))
        self.assertIn("incomplete", decide_recommendation({"delta": None}))


if __name__ == "__main__":
    unittest.main()
