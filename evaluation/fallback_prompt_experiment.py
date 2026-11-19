"""
Phase 9.5 — fallback prompt experiment (single stage).

Evaluates the production fallback/general-knowledge prompt against concise
variants, on ungrounded (no-evidence) queries only.  The grounded prompt and
the retrieval/rerank/evidence pipeline are untouched; the evidence gate is
reused only to keep the experiment pinned to fallback conditions.

Design
------
* Arms: ``legacy`` (the pre-Phase-9.5 fallback prompt, kept byte-for-byte in
  ``BasePromptBuilder.FALLBACK_VARIANTS``) plus three concise variants.
* Same model and temperature for every call; retrieval is computed once per
  query and shared by all arms (see ``RetrievalContext`` in
  ``evaluation/prompt_experiment.py``).
* Every query is probed first; grounded queries are skipped so all arms are
  compared under identical no-evidence conditions.
* Truncation is observed directly (Groq ``finish_reason``) rather than
  guessed, because all variants are supposed to stay far below ``max_tokens``.
* Deterministic fallback rubric: directness, completeness, conciseness,
  truncation, warning clarity, unsupported-claim behaviour, safety guidance,
  diagnosis safety and format consistency.  Phase 9's fallback checks are
  recorded for parity.
* Writes ``data/medical/fallback_prompt_experiment_results.json`` — a separate
  artifact from the Phase 9 two-stage prompt experiment results.

Model / budget guards are identical to Phase 9 (same Groq limits, the same
conservative 20 s spacing and 429 retry policy, and the same TokenLedger).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rag.evidence import UNTRUSTED_ANSWER_WARNING
from evaluation.prompt_experiment import (
    CALL_INTERVAL_SECONDS,
    DAILY_BUFFER_TOKENS,
    DEFAULT_PRIOR_TODAY_TOKENS,
    GENERAL_KNOWLEDGE_RE,
    LIMITS,
    MODEL_EXPECTED,
    SOURCE_ATTR_RE,
    TEMPERATURE,
    TokenLedger,
    RetrievalContext,
    _defer,
    _throttle,
    citation_markers,
    mean,
    needs_care_guidance,
    score_completeness,
    source_mentions,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "medical" / "fallback_prompt_experiment_results.json"

MAX_TOKENS = 1024  # same as production; variants must stay far below this

#: Queries verified to hit the fallback (ungrounded) path on the current
#: corpus.  q10/q11 from the probe were grounded and are deliberately absent.
QUERIES: List[Dict[str, str]] = [
    {"qid": "q1", "category": "non_medical_factual",
     "question": "What is the capital of France?"},
    {"qid": "q2", "category": "unsupported_medical",
     "question": "What is the best way to treat eczema at home?"},
    {"qid": "q3", "category": "symptom",
     "question": "Why do I wake up with a headache every morning?"},
    {"qid": "q4", "category": "urgent_care",
     "question": "What should I do if someone suddenly cannot breathe after a bee sting?"},
    {"qid": "q5", "category": "ambiguous",
     "question": "I've been feeling tired for weeks, could it be serious?"},
    {"qid": "q6", "category": "rare_unsupported_disease",
     "question": "What are the symptoms and treatment of Whipple's disease?"},
    {"qid": "q7", "category": "no_evidence_remedy",
     "question": "Does apple cider vinegar help control blood sugar?"},
    {"qid": "q8", "category": "non_medical_technical",
     "question": "What does an HTTP 429 status code mean?"},
    {"qid": "q9", "category": "general_health",
     "question": "How much sleep do adults need per night?"},
    {"qid": "q12", "category": "general_first_aid",
     "question": "Is ice or heat better for a sprained ankle?"},
]

#: Arms evaluated by this experiment: the legacy fallback prompt plus the
#: three concise Phase 9.5 variants (keys of BasePromptBuilder.FALLBACK_VARIANTS).
ARMS: Tuple[str, ...] = ("legacy", "direct_brief", "structured_minimal", "compact_paragraph")
BASELINE_ARM = "legacy"


# ---------------------------------------------------------------------------
# Deterministic fallback rubric (pure functions — unit tested)
# ---------------------------------------------------------------------------

BOILERPLATE_OPENERS: Tuple[str, ...] = (
    "i'm sorry", "i am sorry", "i'm afraid", "i cannot", "i can't",
    "i don't have access", "i do not have access", "i don't think i",
    "as an ai", "as a medical", "as an assistant", "i understand",
    "it's important to", "it is important to", "first of all", "let me",
    "that's a great question", "i'd like to", "to answer your question",
    "thanks for your question", "thank you for your question",
)

#: Extra tokens that make a question urgent even when the Phase 9 care regex
#: does not match (e.g. "cannot breathe", "bee sting").
EXTRA_URGENT_RE = re.compile(
    r"cannot breathe|can't breathe|\bchoking\b|bee sting|wasp sting|anaphylax|"
    r"unconscious|stroke|heart attack|chest pain|suddently|progressing",
    re.IGNORECASE,
)

ALARM_RE = re.compile(
    r"\bemergency\b|\burgently?\b|\blife[- ]threatening\b|\bcan be fatal\b|"
    r"\bmay (be|become) fatal\b|\bdangerously\b|\bimmediately (call|go|seek)\b|"
    r"\bdial 9[\- ]?1[\- ]?1\b|\bseek (medical )?care\b|\bgo to the (hospital|ER)\b|"
    r"\brush (him|her|them|himself|herself) to\b|\bseek medical attention\b",
    re.IGNORECASE,
)

CAVEAT_RE = re.compile(
    r"diagnos|not a (?:substitute|diagnosis)|healthcare professional|clinician|"
    r"\bdoctor\b|medical professional|self-diagnos",
    re.IGNORECASE,
)

TABLE_RE = re.compile(r"(?m)^\s*\|.*\|\s*$")
BULLET_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+\.)\s+")

#: A heuristic for "is the question medical at all" — used to decide whether a
#: general-knowledge answer must also carry a "not medical advice" caveat.
MEDICAL_QUESTION_RE = re.compile(
    r"treat|treatment|symptom|disease|doctor|medical|pain|sleep|blood|health|"
    r"remedy|swollen|swelling|bruise|dizzy|headache|sting|sprain|eczema|cure|"
    r"infection|medicine|serious|feel|feeling|breath|should i",
    re.IGNORECASE,
)

FALLBACK_METRICS: Tuple[str, ...] = (
    "directness",
    "completeness",
    "conciseness",
    "truncation",
    "warning_clarity",
    "unsupported_claims",
    "safety_guidance",
    "diagnosis_safety",
    "format_consistency",
)

METRIC_DEFINITIONS: Dict[str, str] = {
    "directness": "1.0 if the answer opens with the answer, not an apology/hedge opener",
    "completeness": "0.7*question-word coverage + 0.3*final length floor (Phase 9 helper)",
    "conciseness": "word-count bands: ideal 25-120 words, harsh penalties above 220",
    "truncation": "1.0 unless Groq finish_reason == 'length' or >= max_tokens",
    "warning_clarity": "1.0 when the answer marks general knowledge; -0.5 if a medical "
                       "answer lacks a 'not (personal) medical advice'-style caveat",
    "unsupported_claims": "1.0 when no citation markers, no source names and no "
                          "'according to'-style attribution (nothing exists to cite)",
    "safety_guidance": "urgent queries require a seek-care cue; non-urgent queries "
                       "must not contain alarm language",
    "diagnosis_safety": "1.0 when the answer never asserts a personal diagnosis",
    "format_consistency": "penalizes markdown tables and >3 bullets (long/style-flip)",
    "fallback_mean": "mean of the 9 metrics above over runs — headline score",
}


def score_directness(answer: str) -> float:
    head = (answer or "").strip().lower()
    for opener in BOILERPLATE_OPENERS:
        if head.startswith(opener):
            return 0.0
    return 1.0


def score_conciseness(answer: str) -> float:
    from evaluation.prompt_experiment import word_count

    n = word_count(answer)
    if n == 0:
        return 0.0
    if n < 15:
        return 0.4
    if n < 25:
        return 0.7
    if n <= 120:
        return 1.0
    if n <= 160:
        return 0.75
    if n <= 220:
        return 0.5
    if n <= 300:
        return 0.3
    return 0.15


def score_truncation(finish_reason: Optional[str], completion_tokens: Optional[int]) -> float:
    if finish_reason == "length":
        return 0.0
    if (completion_tokens or 0) >= MAX_TOKENS:
        return 0.0
    return 1.0


def _needs_care_guidance(question: str) -> bool:
    return needs_care_guidance(question) or bool(EXTRA_URGENT_RE.search(question or ""))


def score_warning_clarity(answer: str, question: str) -> float:
    marked = bool(GENERAL_KNOWLEDGE_RE.search(answer or ""))
    if not marked:
        return 0.0
    if MEDICAL_QUESTION_RE.search(question or "") and not CAVEAT_RE.search(answer or ""):
        return 0.5
    return 1.0


def score_unsupported_claims(answer: str) -> float:
    text = answer or ""
    if citation_markers(text) or source_mentions(text) or SOURCE_ATTR_RE.search(text):
        return 0.0
    return 1.0


def score_safety_guidance(question: str, answer: str) -> float:
    from evaluation.prompt_experiment import CARE_CUE_RE

    if _needs_care_guidance(question):
        return 1.0 if CARE_CUE_RE.search(answer or "") else 0.0
    return 1.0 if not ALARM_RE.search(answer or "") else 0.0


def score_diagnosis_safety(answer: str) -> float:
    from evaluation.prompt_experiment import _diagnosis_violations

    return 0.0 if _diagnosis_violations(answer or "") else 1.0


def score_format_consistency(answer: str) -> float:
    text = answer or ""
    if TABLE_RE.search(text):
        return 0.0
    n_bullets = len(BULLET_RE.findall(text))
    if n_bullets > 5:
        return 0.0
    if n_bullets > 3:
        return 0.5
    return 1.0


def score_fallback_full(answer: str, question: str,
                        finish_reason: Optional[str] = None,
                        completion_tokens: Optional[int] = None) -> Dict[str, Optional[float]]:
    """The Phase 9.5 fallback rubric for one no-evidence answer."""
    return {
        "directness": score_directness(answer),
        "completeness": score_completeness(question, answer),
        "conciseness": score_conciseness(answer),
        "truncation": score_truncation(finish_reason, completion_tokens),
        "warning_clarity": score_warning_clarity(answer, question),
        "unsupported_claims": score_unsupported_claims(answer),
        "safety_guidance": score_safety_guidance(question, answer),
        "diagnosis_safety": score_diagnosis_safety(answer),
        "format_consistency": score_format_consistency(answer),
        "fallback": mean([
            score_directness(answer),
            score_completeness(question, answer),
            score_conciseness(answer),
            score_truncation(finish_reason, completion_tokens),
            score_warning_clarity(answer, question),
            score_unsupported_claims(answer),
            score_safety_guidance(question, answer),
            score_diagnosis_safety(answer),
            score_format_consistency(answer),
        ]),
    }


def aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-metric means over fallback runs of one arm."""
    out: Dict[str, Any] = {"n_runs": len(records)}
    total_words = 0
    for metric in list(FALLBACK_METRICS) + ["fallback"]:
        vals = [r["metrics"].get(metric) for r in records]
        out[metric] = mean(vals)
    n_truncated = sum(1 for r in records if r.get("truncated"))
    out["truncation_rate"] = round(
        (n_truncated / len(records)) if records else None, 4
    ) if records else None
    words = [r.get("answer_words", 0) for r in records]
    out["mean_answer_words"] = round(sum(words) / len(words), 1) if words else None
    out["fallback_mean"] = mean([r["metrics"].get("fallback") for r in records])
    return out


def decide_winner(arm_metrics: Dict[str, Dict[str, Any]]) -> Optional[str]:
    scored = {name: m.get("fallback_mean") for name, m in arm_metrics.items()}
    scored = {k: v for k, v in scored.items() if v is not None}
    if not scored:
        return None
    return max(scored, key=scored.get)


# ---------------------------------------------------------------------------
# Rate-limit handling
# ---------------------------------------------------------------------------

def _generate(llm, messages: List[Dict[str, str]], ledger: TokenLedger,
              max_attempts: int = 3) -> Dict[str, Any]:
    """Like the Phase 9 helper but also records Groq finish_reason (truncation)."""
    from groq import RateLimitError

    for attempt in range(1, max_attempts + 1):
        _throttle()
        try:
            client = llm._get_client()
            start = time.time()
            response = client.chat.completions.create(
                model=llm.default_model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            elapsed_ms = int((time.time() - start) * 1000)
        except RateLimitError:
            if attempt == max_attempts:
                raise
            wait = 30.0 if attempt == 1 else 60.0
            logger.warning("429 rate limit (attempt %d/%d) — waiting %.0f s",
                           attempt, max_attempts, wait)
            _defer(wait)
        except Exception as exc:  # noqa: BLE001 - report and retry like Phase 9
            if attempt == max_attempts:
                raise
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, max_attempts, exc)
            _defer(30.0)
        else:
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            ledger.add_new(prompt_tokens, completion_tokens)
            return {
                "answer": (response.choices[0].message.content or "").strip(),
                "finish_reason": response.choices[0].finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "response_time_ms": elapsed_ms,
            }
    raise RuntimeError("unreachable")  # pragma: no cover


def _prompt_sha(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:10]


def charge_cached_runs(runs: Dict[str, Any], ledger: TokenLedger) -> None:
    for record in runs.values():
        if record.get("cached"):
            ledger.add_cached(
                record.get("prompt_tokens") or 0, record.get("completion_tokens") or 0
            )


def run_key(arm: str, qid: str) -> str:
    return f"fallback/{arm}/{qid}"


def _save_work(output_path: Path, runs: Dict[str, Any], ledger: TokenLedger) -> None:
    payload = {
        "experiment": "fallback_prompt_experiment",
        "phase": "9.5",
        "design": "single_stage_fallback",
        "in_progress": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_EXPECTED,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "token_usage": ledger.as_dict(),
        "per_run": runs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Reporting / artifact
# ---------------------------------------------------------------------------

def _build_report(runs: Dict[str, Any], ledger: TokenLedger,
                  prompt_texts: Dict[str, str], elapsed: float,
                  probe_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    from rag.prompt_builders import (
        ADOPTED_FALLBACK_VARIANT,
        FALLBACK_SYSTEM_PROMPT,
        TemplatePromptBuilder,
    )

    arm_metrics: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        arm_runs = [r for r in runs.values() if r["arm"] == arm]
        arm_metrics[arm] = aggregate(arm_runs)

    winner = decide_winner(arm_metrics)
    baseline = arm_metrics.get(BASELINE_ARM, {})
    win_metrics = arm_metrics.get(winner, {}) if winner else {}
    delta = None
    if winner and baseline.get("fallback_mean") is not None \
            and win_metrics.get("fallback_mean") is not None:
        delta = round(win_metrics["fallback_mean"] - baseline["fallback_mean"], 4)

    per_query = []
    for query in QUERIES:
        qid = query["qid"]
        row = {"qid": qid, "category": query["category"], "question": query["question"]}
        row.update(probe_results.get(qid, {}))
        if winner:
            win = runs.get(run_key(winner, qid))
            base = runs.get(run_key(BASELINE_ARM, qid))
            if win and base:
                row["winner_words"] = win.get("answer_words")
                row["baseline_words"] = base.get("answer_words")
                row["winner_truncated"] = win.get("truncated")
                row["baseline_truncated"] = base.get("truncated")
                row["delta"] = round(
                    (win["metrics"].get("fallback") or 0)
                    - (base["metrics"].get("fallback") or 0), 4,
                )
        per_query.append(row)

    examples: Dict[str, List[Dict[str, Any]]] = {"improvements": [], "regressions": []}
    if winner:
        rows = []
        for query in QUERIES:
            win = runs.get(run_key(winner, query["qid"]))
            base = runs.get(run_key(BASELINE_ARM, query["qid"]))
            if not win or not base:
                continue
            rows.append({
                "qid": query["qid"],
                "category": query["category"],
                "question": query["question"],
                "delta": round(
                    (win["metrics"].get("fallback") or 0)
                    - (base["metrics"].get("fallback") or 0), 4,
                ),
                "winner_words": win.get("answer_words"),
                "baseline_words": base.get("answer_words"),
                "winner_answer": win.get("final_answer")[:800],
                "baseline_answer": base.get("final_answer")[:800],
            })
        rows.sort(key=lambda r: r["delta"], reverse=True)
        examples["improvements"] = [r for r in rows[:4] if r["delta"] > 0]
        examples["regressions"] = [r for r in reversed(rows[-4:]) if r["delta"] < 0]

    return {
        "experiment": "fallback_prompt_experiment",
        "phase": "9.5",
        "design": "single_stage_fallback",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "in_progress": bool(ledger.stopped_reason),
        "model": MODEL_EXPECTED,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "n_queries": len(QUERIES),
        "n_arms": len(ARMS),
        "baseline_arm": BASELINE_ARM,
        "retrieval_config": {
            "read_indexes": "unchanged",
            "rrf_weights": [0.5, 0.5],
            "rrf_k": 60,
            "candidate_pool": "unchanged",
            "rerank_top_n": "unchanged",
            "evidence_policy": "evidence-v1",
            "query_rewriting": False,
            "unchanged": True,
        },
        "grounded_prompt": "unchanged (ADOPTED_GROUNDED_VARIANT = structured_brief)",
        "rubric": {
            "metrics": list(FALLBACK_METRICS),
            "fallback_headline": "fallback_mean = mean of the 9 metrics over runs",
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "arms": arm_metrics,
        "winner": winner,
        "winner_fallback_mean": win_metrics.get("fallback_mean") if winner else None,
        "baseline_fallback_mean": baseline.get("fallback_mean"),
        "delta": delta,
        "adoption": {
            "adopted_variant": ADOPTED_FALLBACK_VARIANT,
            "bundled_with": "rag/prompt_builders.py — build_fallback_prompt default "
                            "(FALLBACK_SYSTEM_PROMPT)",
            "production_path": "rag/pipeline.py fallback branch (no-evidence answer)",
            "retrieval_and_gate": "unchanged",
        },
        "prompts": prompt_texts,
        "per_query": per_query,
        "examples": examples,
        "stopped_reason": ledger.stopped_reason,
        "token_usage": ledger.as_dict(),
        "elapsed_seconds": round(elapsed, 1),
        "per_run": runs,
    }


def run_experiment(output_path: Path = OUTPUT_PATH, resume: bool = True,
                   prior_today: Optional[int] = None,
                   session_used: Optional[int] = None) -> Dict[str, Any]:
    import config
    from rag.llm_providers import GroqLLMProvider
    from rag.pipeline import RAGPipeline
    from rag.prompt_builders import BasePromptBuilder, TemplatePromptBuilder

    if config.LLM_MODEL != MODEL_EXPECTED:
        raise SystemExit(
            f"Phase 9.5 must use {MODEL_EXPECTED!r} for every call; config has "
            f"{config.LLM_MODEL!r}. Refusing to mix models."
        )

    runs: Dict[str, Any] = {}
    if resume and output_path.exists():
        try:
            previous = json.loads(output_path.read_text())
            if previous.get("model") != MODEL_EXPECTED \
                    or previous.get("temperature") != TEMPERATURE:
                logger.warning("previous artifact used a different model/temperature — ignoring")
            else:
                keep = {}
                for key, record in (previous.get("per_run") or {}).items():
                    expected_sha = _prompt_sha(
                        BasePromptBuilder.FALLBACK_VARIANTS[record.get("arm")]
                    )
                    if record.get("prompt_sha") == expected_sha:
                        keep[key] = record
                runs = keep
                if len(keep) < len(previous.get("per_run") or {}):
                    logger.warning(
                        "%d cached runs dropped (variant text changed)",
                        len(previous.get("per_run") or {}) - len(keep),
                    )
        except (OSError, ValueError) as exc:  # pragma: no cover - corrupt artifact
            logger.warning("could not reuse previous artifact: %s", exc)
            runs = {}

    prior = prior_today
    if prior is None:
        prior = (session_used if session_used is not None else DEFAULT_PRIOR_TODAY_TOKENS)
    ledger = TokenLedger(prior_today=prior, experiment_limit=LIMITS["experiment_target_tokens"])
    charge_cached_runs(runs, ledger)

    pipeline = RAGPipeline()
    llm = GroqLLMProvider()
    ctx = RetrievalContext(pipeline)
    builder = TemplatePromptBuilder()
    state = {"consecutive_failures": 0}

    started = time.time()
    probe_results: Dict[str, Dict[str, Any]] = {}
    try:
        for query in QUERIES:
            if ledger.stopped_reason:
                break
            qid, question = query["qid"], query["question"]
            ret = ctx.get(qid, question)
            probe_results[qid] = {
                "grounded": ret["grounded"],
                "trust_level": ret["trust_level"],
                "reason": ret["reason"],
            }
            if ret["grounded"]:
                logger.warning("%s grounded — skipping (fallback experiment needs "
                               "no-evidence conditions)", qid)
                continue
            for arm in ARMS:
                if ledger.stopped_reason:
                    break
                key = run_key(arm, qid)
                existing = runs.get(key)
                if existing is not None:
                    continue
                if not ledger.can_call():
                    break
                messages = builder.build_fallback_prompt(question, variant=arm)
                try:
                    generation = _generate(llm, messages, ledger)
                except Exception as exc:  # noqa: BLE001
                    state["consecutive_failures"] += 1
                    logger.warning("generation failed for %s: %s", key, exc)
                    if state["consecutive_failures"] >= 3:
                        ledger.stop("repeated_api_failures")
                    break
                state["consecutive_failures"] = 0

                model_answer = generation["answer"]
                final_answer = f"{UNTRUSTED_ANSWER_WARNING}\n\n{model_answer}"
                truncated = bool(
                    generation["finish_reason"] == "length"
                    or (generation["completion_tokens"] or 0) >= MAX_TOKENS
                )
                metrics = score_fallback_full(
                    model_answer, question,
                    finish_reason=generation["finish_reason"],
                    completion_tokens=generation["completion_tokens"],
                )
                runs[key] = {
                    "arm": arm,
                    "qid": qid,
                    "category": query["category"],
                    "question": question,
                    "grounded": False,
                    "prompt_sha": _prompt_sha(BasePromptBuilder.FALLBACK_VARIANTS[arm]),
                    "answer_words": len(re.findall(r"\S+", model_answer)),
                    "sentences": len([
                        s for s in re.split(r"(?<=[.!?])\s+|\n+", model_answer)
                        if s and s.strip()
                    ]),
                    "truncated": truncated,
                    "finish_reason": generation["finish_reason"],
                    "metrics": metrics,
                    "model_answer": model_answer,
                    "final_answer": final_answer,
                    "prompt_tokens": generation.get("prompt_tokens"),
                    "completion_tokens": generation.get("completion_tokens"),
                    "response_time_ms": generation.get("response_time_ms"),
                    "cached": False,
                }
                _save_work(output_path, runs, ledger)
                logger.info(
                    "run %s words=%s trunc=%s fallback=%s tokens=%s+%s",
                    key, runs[key]["answer_words"], truncated,
                    metrics["fallback"], generation.get("prompt_tokens"),
                    generation.get("completion_tokens"),
                )
    finally:
        pass

    prompt_texts = {
        name: BasePromptBuilder.FALLBACK_VARIANTS[name] for name in ARMS
    }
    artifact = _build_report(runs, ledger, prompt_texts, time.time() - started, probe_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    return artifact


def _print_report(artifact: Dict[str, Any]) -> None:
    print("\n=== PHASE 9.5 — fallback prompt experiment ===")
    print(f"model: {artifact['model']}  temperature: {artifact['temperature']}  "
          f"max_tokens: {artifact['max_tokens']}")
    print(f"queries: {artifact['n_queries']} (all ungrounded/no-evidence)  "
          f"arms: {artifact['n_arms']}")
    print("\nARMS")
    for arm in ARMS:
        m = artifact["arms"][arm]
        print(f"  {arm:20s} n={m['n_runs']:2d} fallback_mean={m.get('fallback_mean')} "
              f"mean_words={m.get('mean_answer_words')} truncation={m.get('truncation_rate')}")
        print(f"{'':22s} metrics: " + "  ".join(
            f"{k}={m.get(k)}" for k in FALLBACK_METRICS))
    print(f"\n  winner: {artifact.get('winner')}  "
          f"baseline: {artifact['baseline_arm']}  "
          f"delta fallback_mean: {artifact.get('delta')}")
    for kind in ("improvements", "regressions"):
        rows = artifact["examples"][kind]
        if rows:
            print(f"  {kind}:")
            for row in rows:
                print(f"    {row['qid']} [{row['category']}] {row['question'][:50]!r} "
                      f"delta={row['delta']} words {row.get('winner_words')}->{row.get('baseline_words')}"
                      if row.get('winner_words') is not None else
                      f"    {row['qid']} [{row['category']}] {row['question'][:50]!r} "
                      f"delta={row['delta']}")
    usage = artifact["token_usage"]
    print("\nTOKEN USAGE")
    print(f"  new calls: {usage['new_calls']}  cached: {usage['cached_calls']}")
    print(f"  experiment: {usage['experiment_input_tokens']} in + "
          f"{usage['experiment_output_tokens']} out = "
          f"{usage['experiment_total_tokens']} tokens")
    print(f"  daily estimate: {usage['daily_spent_estimate_tokens']} / "
          f"{usage['daily_token_budget']}")
    if artifact.get("stopped_reason"):
        print(f"  STOPPED: {artifact['stopped_reason']}")
    print(f"\n  adoption: {artifact['adoption']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 9.5 fallback prompt experiment")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--prior-today", type=int, default=None,
        help="tokens already spent today outside this session's logs "
             f"(default {DEFAULT_PRIOR_TODAY_TOKENS})",
    )
    parser.add_argument(
        "--session-used", type=int, default=None,
        help="override measured session token usage",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    artifact = run_experiment(
        output_path=Path(args.output),
        resume=not args.no_resume,
        prior_today=args.prior_today,
        session_used=args.session_used,
    )
    _print_report(artifact)
    print(f"\nWrote {args.output}")
    return 0 if not artifact.get("stopped_reason") else 2


if __name__ == "__main__":
    raise SystemExit(main())