"""Phase 9 — two-stage prompt engineering experiment (free-tier rate limits).

One model for the entire experiment: ``qwen/qwen3.8-27b`` (never switched).
Groq limits for that model: 30 RPM, 1,000 RPD, 8,000 TPM, 200,000 TPD.

Design (per the Phase 9 brief):

STAGE 1 — 12 representative queries (symptoms, diseases, Indian guidance,
          basic care, warning signs, ambiguous symptoms, user-only evidence,
          no-evidence fallback) × the 3 grounded prompt variants = max 36 calls.
          Retrieval/context is computed once per query and reused for every
          variant.  Picks the stage-1 winner.

STAGE 2 — stage-1 winner vs the current production prompt on all 28 medical
          queries = max 56 calls, reusing the same cached retrieval/context.

Rate-limit rules enforced by :class:`TokenLedger` + :func:`_generate`:
    * actual input/output tokens tracked per run and reported per stage
    * experiment target <= 180,000 tokens; daily budget 200,000 TPD
      (prior usage measured from session logs + margin); the experiment
      STOPS rather than exceeding either budget
    * 20 s spacing between calls (≈4.5K TPM, under the 8K TPM limit)
    * conservative 429 handling: max 3 attempts, fixed 30 s / 60 s waits —
      no aggressive retry loops

Runs produced earlier in this session are reused as a cache when they match
the same arm/query/model/temperature, so rate-limited budget is spent only on
runs that do not exist yet.

Retrieval, embeddings, reranker, RRF (0.5/0.5, k=60), candidate pool (20) and
the EvidenceGate are unchanged; the only live-index side effect is a temporary
user upload for the stage-1 user-only query, cleared afterwards.

Usage:
    python3 -m evaluation.prompt_experiment [--stage {1,2,all}]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.evidence import UNTRUSTED_ANSWER_WARNING  # noqa: E402
from rag.prompt_builders import BasePromptBuilder  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_PATH = ROOT / "data" / "medical" / "prompt_experiment_results.json"
GROUND_TRUTH_PATH = ROOT / "data" / "medical" / "medical_ground_truth.json"
FALLBACK_PROMPT_PATH_HINT = "rag/prompt_builders.BasePromptBuilder.FALLBACK_SYSTEM_PROMPT"

MODEL_EXPECTED = "qwen/qwen3.8-27b"
TEMPERATURE = 0.0

#: Published Groq limits for MODEL_EXPECTED (free plan) + experiment budgets.
LIMITS: Dict[str, int] = {
    "rpm": 30,
    "rpd": 1000,
    "tpm": 8000,
    "tpd": 200000,
    "stage1_max_llm_calls": 36,
    "stage2_max_llm_calls": 56,
    "experiment_target_tokens": 180000,
    "daily_token_budget": 200000,
}

#: Seconds between LLM calls (≈4.5K TPM at ~1.4K tokens/call < 8K TPM).
CALL_INTERVAL_SECONDS = float(os.getenv("PROMPT_EXPERIMENT_CALL_INTERVAL", "20"))
#: Head-room kept below the daily budget before refusing further calls.
DAILY_BUFFER_TOKENS = 4000
#: Usage from earlier phases today that session logs cannot measure.
DEFAULT_PRIOR_TODAY_TOKENS = 25000

STAGE1_MAX_CALLS = LIMITS["stage1_max_llm_calls"]
STAGE2_MAX_CALLS = LIMITS["stage2_max_llm_calls"]
STAGE2_QIDS: List[int] = list(range(1, 29))

#: The three grounded prompt variants evaluated in stage 1.
VARIANT_ARMS: List[Dict[str, str]] = [
    {"name": "variant_structured_brief", "kind": "variant", "value": "structured_brief"},
    {"name": "variant_question_first", "kind": "variant", "value": "question_first"},
    {"name": "variant_triage_concise", "kind": "variant", "value": "triage_concise"},
]
#: The pre-adoption production prompt (historic style prompt), kept as the
#: stage-2 challenger.  Production now defaults to ADOPTED_GROUNDED_VARIANT,
#: so baseline arms must pass ``variant=None`` explicitly.
BASELINE_ARM: Dict[str, str] = {
    "name": "baseline_detailed",
    "kind": "style",
    "value": "detailed",
}

#: Stage-1 sample: 11 queries from the 28-query ground truth plus one
#: user-only-evidence query (requires a temporary upload to medibo_user).
STAGE1_SELECTION: List[Dict[str, Any]] = [
    {"qid": 1, "category": "warning_signs"},
    {"qid": 6, "category": "symptoms"},
    {"qid": 17, "category": "symptoms"},
    {"qid": 8, "category": "disease"},
    {"qid": 14, "category": "disease"},
    {"qid": 5, "category": "indian_guidance"},
    {"qid": 15, "category": "indian_guidance"},
    {"qid": 12, "category": "basic_care"},
    {"qid": 18, "category": "basic_care"},
    {"qid": 11, "category": "ambiguous_symptoms"},
    {"qid": 2, "category": "no_evidence_fallback"},
    {"qid": "u1", "category": "user_only_evidence"},
]
STAGE1_QIDS: List[Any] = [entry["qid"] for entry in STAGE1_SELECTION]
STAGE1_CATEGORIES: List[str] = sorted({entry["category"] for entry in STAGE1_SELECTION})

#: The user-only evidence case: a single uploaded file, query keyed by a
#: unique token so the curated corpus cannot answer it.
USER_QUERY: Dict[str, str] = {
    "qid": "u1",
    "question": "What is the CADET-4 asthma action plan?",
    "file_name": "my-clinic-asthma-action-plan.txt",
    "content": (
        "CADET-4 asthma action plan: when peak flow falls below 60 percent of "
        "personal best, give the reliever inhaler, repeat after 20 minutes, and "
        "call the clinic if there is no improvement within 4 hours."
    ),
}

METRIC_RUBRIC: Dict[str, str] = {
    "source_adherence": (
        "1.0 = every source/URL/file/org named in the answer exists in the supplied "
        "citations+context and no ≥40-char verbatim context passage is reproduced; "
        "0.5 if only a verbatim leak; 1 - violations/mentions otherwise"
    ),
    "unsupported_claims": (
        "share of medical claim sentences with ≥0.35 content-word overlap against "
        "the supplied context (1.0 when no medical claims)"
    ),
    "citation_correctness": (
        "0.5*precision([n] markers inside 1..N) + 0.5*coverage(distinct markers/N); "
        "0 when nothing is cited; None when the run is a fallback"
    ),
    "answer_completeness": (
        "0.7*question-term coverage + 0.3*length adequacy (>=40 words)"
    ),
    "verbosity": (
        "1.0 for 50–200 words, linear below 50 and above 200 (0 at 450)"
    ),
    "red_flag_handling": (
        "for warning/first-aid/when-to-seek queries: 1.0 if the answer contains a "
        "seek-care escalation cue; else 0 (None when not applicable)"
    ),
    "symptom_vs_diagnosis": (
        "for symptom/sign queries: 0 for 'you have <condition>' claims (unless "
        "conditional: if/when/unless), else 1.0 with a diagnostic caveat, 0.5 without"
    ),
    "fallback": (
        "mean of: warning prefix present, general-knowledge marker, no [n] markers, "
        "no source attribution (URL/file/org/'according to')"
    ),
}

# ---------------------------------------------------------------------------
# Deterministic rubric helpers (pure functions — unit tested)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "at", "is",
    "are", "was", "were", "be", "been", "do", "does", "did", "can", "could",
    "should", "would", "will", "with", "from", "by", "as", "it", "its", "this",
    "that", "these", "those", "you", "your", "we", "our", "i", "me", "my",
    "what", "when", "where", "who", "why", "how", "which", "if", "into", "over",
    "about", "after", "before", "during", "through", "than", "then", "also",
    "any", "some", "all", "more", "most", "other", "such", "only", "just",
    # question scaffolding that carries little topical signal
    "steps", "step", "basics", "basic", "faq", "general", "public", "list",
    "lists", "explain", "explains", "give", "tell", "please", "need", "want",
}

MEDICAL_TERMS = {
    "fever", "pain", "symptom", "symptoms", "sign", "signs", "treatment",
    "treat", "therapy", "diagnosis", "diagnostic", "infection", "disease",
    "diseases", "condition", "risk", "risks", "emergency", "urgent", "urgency",
    "care", "medication", "medicine", "drug", "drugs", "dose", "doses",
    "cough", "vomiting", "bleeding", "rash", "seizure", "seizures", "stroke",
    "heart", "blood", "pressure", "malaria", "dengue", "cholera", "pneumonia",
    "tuberculosis", "rabies", "snakebite", "heatstroke", "encephalitis",
    "leptospirosis", "influenza", "dehydration", "nausea", "headache",
    "fainting", "chikungunya", "trauma", "antibiotic", "antibiotics", "fluids",
    "hydration", "doctor", "hospital", "clinic", "seek", "warning", "warnings",
    "prevention", "vaccine", "antivenom", "antiseptic", "asthma", "peak",
}

URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
FILE_RE = re.compile(r"\b[\w\-.]+\.(?:json|txt|pdf|xml|html|md|csv)\b", re.IGNORECASE)
CITATION_MARKER_RE = re.compile(r"\[(\d{1,2})\]")

#: Distinctive source/organisation tokens that must exist in the supplied
#: evidence if the answer names them.  ("WHO" is matched case-sensitively.)
ORG_TOKENS = (
    "medlineplus", "icmr", "mohfw", "ncdc", "cdc", "nice", "mayoclinic",
    "mayo clinic", "aiims", "ministry of health", "national centre for disease "
    "control", "national institutes of health", "indian council of medical "
    "research", "world health organization", "centers for disease control",
    "federal agency", "govt", "government of india",
)
ORG_WHO_RE = re.compile(r"\bWHO\b")

CARE_QUERY_RE = re.compile(
    r"warning|red flag|first aid|when to seek|when to worry|seek (medical )?care|"
    r"emergency|breathing problem|trauma|post-exposure|management guidelines|"
    r"initial management|when to go to the hospital",
    re.IGNORECASE,
)
CARE_CUE_RE = re.compile(
    r"seek (medical )?care|emergency|ambulance|hospital|clinic|\bdoctor|"
    r"physician|healthcare|medical attention|immediately|urgent|urgently|"
    r"call (for )?(a )?doctor|\b102\b|\b108\b|escalat|do not wait|same day|"
    r"get (it|this) checked|professional help",
    re.IGNORECASE,
)
SYMPTOM_QUERY_RE = re.compile(r"\bsigns\b|\bsymptoms\b", re.IGNORECASE)
DEFINITIVE_RE = re.compile(
    r"\byou (have|had|'ve got|are suffering from|may have|might have|could have|"
    r"probably have|likely have)\b",
    re.IGNORECASE,
)
CONDITIONAL_PREFIX_RE = re.compile(
    r"\b(if|when|unless|whenever|even if)\b\s*$", re.IGNORECASE
)
CAVEAT_RE = re.compile(
    r"diagnos|not a (?:substitute|diagnosis)|healthcare professional|clinician|"
    r"\bdoctor\b|medical professional|self-diagnos",
    re.IGNORECASE,
)
GENERAL_KNOWLEDGE_RE = re.compile(
    r"general knowledge|general (medical )?information|general (encyclopedic )?"
    r"information|not (available |found )?in the (knowledge base|provided|context)|"
    r"outside (of )?(the )?(knowledge|provided)|"
    r"I do not have (access to )?|no (relevant )?(information|documents)|"
    r"not (from|based on) the (knowledge base|provided|context)|"
    r"no (specific )?(documents|sources) (were )?(provided|available)",
    re.IGNORECASE,
)
SOURCE_ATTR_RE = re.compile(
    r"according to|as (stated|reported|described|mentioned|written) in|"
    r"based on (the )?(provided )?(context|document|source|knowledge base)|"
    r"sources?:|references?:|see reference",
    re.IGNORECASE,
)


def tokenize(text: str) -> List[str]:
    """Lowercased alphanumeric words of length ≥ 3, stopwords removed."""
    words = re.findall(r"[a-z][a-z\-']+", (text or "").lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def word_variants(word: str) -> Iterable[str]:
    """Naive singular/plural variants so coverage is not defeated by morphology."""
    yield word
    if word.endswith("s") and len(word) > 3:
        yield word[:-1]
    elif word.endswith("ies") and len(word) > 4:
        yield word[:-3] + "y"
    else:
        yield word + "s"
        if word.endswith("y"):
            yield word[:-1] + "ies"


def word_in(word: str, bag: set) -> bool:
    return any(v in bag for v in word_variants(word))


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def citation_markers(answer: str) -> List[int]:
    """All ``[n]`` citation markers in reading order."""
    return [int(m) for m in CITATION_MARKER_RE.findall(answer or "")]


def source_mentions(answer: str) -> List[str]:
    """URLs, file names and distinctive source/organisation names in an answer."""
    text = answer or ""
    mentions = URL_RE.findall(text) + FILE_RE.findall(text)
    lowered = text.lower()
    for token in ORG_TOKENS:
        if token in lowered:
            mentions.append(token)
    if ORG_WHO_RE.search(text):
        mentions.append("WHO")
    return mentions


def _evidence_haystack(citations: Sequence[Any], context_texts: Sequence[str]) -> str:
    parts: List[str] = [str(t) for t in context_texts or []]
    for c in citations or []:
        if isinstance(c, dict):
            fields = c
        else:
            fields = {k: getattr(c, k, None) for k in getattr(c, "__dataclass_fields__", {})}
        for value in fields.values():
            if value:
                parts.append(str(value))
    return "\n".join(parts).lower()


def verbatim_leak(answer: str, context_texts: Sequence[str], min_len: int = 40) -> bool:
    """True when the answer reproduces a ≥min_len character context passage."""
    answer_norm = re.sub(r"\s+", " ", (answer or "")).lower().strip()
    if len(answer_norm) < min_len:
        return False
    for text in context_texts or []:
        text_norm = re.sub(r"\s+", " ", (text or "")).lower().strip()
        for start in range(0, max(0, len(text_norm) - min_len + 1), 20):
            fragment = text_norm[start:start + min_len]
            if len(fragment) == min_len and fragment in answer_norm:
                return True
    return False


def score_source_adherence(
    answer: str, citations: Sequence[Any], context_texts: Sequence[str]
) -> float:
    """No invented sources, no verbatim chunk dumps."""
    haystack = _evidence_haystack(citations, context_texts)
    mentions = source_mentions(answer)
    violations = [m for m in mentions if m.lower() not in haystack]
    if not mentions:
        score = 1.0
    else:
        score = max(0.0, 1.0 - len(violations) / len(mentions))
    if verbatim_leak(answer, context_texts):
        score = min(score, 0.5)
    return round(score, 4)


def score_unsupported_claims(answer: str, context_texts: Sequence[str]) -> float:
    """Share of medical claim sentences grounded in the supplied context."""
    context_words = set(tokenize(" ".join(context_texts or [])))
    claims = [
        s for s in split_sentences(answer)
        if set(tokenize(s)) & MEDICAL_TERMS
    ]
    if not claims:
        return 1.0
    supported = 0
    for claim in claims:
        words = tokenize(claim)
        if not words:
            continue
        overlap = sum(1 for w in words if word_in(w, context_words)) / len(words)
        supported += 1 if overlap >= 0.35 else 0
    return round(supported / len(claims), 4)


def score_citation_correctness(answer: str, n_citations: int) -> Optional[float]:
    """Validity and coverage of ``[n]`` markers.  None on fallback runs."""
    if not n_citations:
        return None
    markers = citation_markers(answer)
    if not markers:
        return 0.0
    valid = [m for m in markers if 1 <= m <= n_citations]
    precision = len(valid) / len(markers)
    coverage = len(set(valid)) / n_citations
    return round(0.5 * precision + 0.5 * coverage, 4)


def score_completeness(question: str, answer: str) -> float:
    q_words = tokenize(question)
    answer_words = set(tokenize(answer))
    if q_words:
        hits = sum(1 for w in q_words if word_in(w, answer_words))
        coverage = hits / len(q_words)
    else:
        coverage = 1.0
    length_ok = min(1.0, word_count(answer) / 40)
    return round(0.7 * coverage + 0.3 * length_ok, 4)


def score_verbosity(answer: str) -> float:
    n = word_count(answer)
    if n == 0:
        return 0.0
    if n < 50:
        return round(n / 50, 4)
    if n <= 200:
        return 1.0
    return round(max(0.0, 1.0 - (n - 200) / 250), 4)


def needs_care_guidance(question: str) -> bool:
    return bool(CARE_QUERY_RE.search(question or ""))


def score_red_flag_handling(question: str, answer: str) -> Optional[float]:
    if not needs_care_guidance(question):
        return None
    return 1.0 if CARE_CUE_RE.search(answer or "") else 0.0


def needs_symptom_handling(question: str) -> bool:
    return bool(SYMPTOM_QUERY_RE.search(question or ""))


def _diagnosis_violations(answer: str) -> List[str]:
    """Personal diagnosis claims that are not conditional (if/when/unless …)."""
    violations = []
    text = answer or ""
    for match in DEFINITIVE_RE.finditer(text):
        prefix = text[max(0, match.start() - 12):match.start()]
        if CONDITIONAL_PREFIX_RE.search(prefix):
            continue
        violations.append(match.group(0))
    return violations


def score_symptom_vs_diagnosis(question: str, answer: str) -> Optional[float]:
    if not needs_symptom_handling(question):
        return None
    if _diagnosis_violations(answer):
        return 0.0
    return 1.0 if CAVEAT_RE.search(answer or "") else 0.5


def score_fallback(model_answer: str, final_answer: str) -> Tuple[Dict[str, bool], float]:
    """Deterministic checks for the untrusted-fallback answer."""
    checks = {
        "warning_prefixed": (final_answer or "").strip().startswith(
            UNTRUSTED_ANSWER_WARNING
        ),
        "marks_general_knowledge": bool(GENERAL_KNOWLEDGE_RE.search(model_answer or "")),
        "no_citation_markers": not citation_markers(model_answer),
        "no_source_attribution": not source_mentions(model_answer)
        and not SOURCE_ATTR_RE.search(model_answer or ""),
    }
    score = round(sum(1 for v in checks.values() if v) / len(checks), 4)
    return checks, score


def score_answer(
    question: str,
    answer: str,
    citations: Sequence[Any],
    context_texts: Sequence[str],
) -> Dict[str, Optional[float]]:
    """Full deterministic rubric for one grounded answer."""
    return {
        "source_adherence": score_source_adherence(answer, citations, context_texts),
        "unsupported_claims": score_unsupported_claims(answer, context_texts),
        "citation_correctness": score_citation_correctness(answer, len(citations)),
        "answer_completeness": score_completeness(question, answer),
        "verbosity": score_verbosity(answer),
        "red_flag_handling": score_red_flag_handling(question, answer),
        "symptom_vs_diagnosis": score_symptom_vs_diagnosis(question, answer),
        "fallback": None,
    }


GROUNDED_METRICS = (
    "source_adherence",
    "unsupported_claims",
    "citation_correctness",
    "answer_completeness",
    "verbosity",
    "red_flag_handling",
    "symptom_vs_diagnosis",
)

#: Explicit metric terminology for the artifact and report (Phase 9 finalize).
#: ``grounded_mean`` is a QUALITY score, not a grounded/ungrounded rate — the
#: flag rate is reported separately as ``grounded_rate``.  The headline score
#: is ``overall_mean``; stage-2 ``delta`` is the overall_mean delta and
#: ``delta_grounded_mean`` is the grounded_mean delta.
METRIC_DEFINITIONS: Dict[str, str] = {
    "grounded_mean": (
        "mean rubric score across the 7 grounded metrics "
        f"({', '.join(GROUNDED_METRICS)}) — quality of grounded answers; "
        "NOT the fraction of grounded runs"
    ),
    "grounded_rate": (
        "fraction of runs whose EvidenceGate decision was sufficient "
        "(the grounded/ungrounded flag rate)"
    ),
    "fallback_mean": "mean fallback rubric score on no-evidence answers",
    "overall_mean": (
        "mean(grounded_mean, fallback_mean) over the non-null of the two — "
        "the single headline score; artifact key 'overall'"
    ),
    "stage2_delta_overall_mean": (
        "stage-2 winner overall_mean − baseline overall_mean; "
        "artifact keys 'delta_overall_mean' and 'delta'"
    ),
    "stage2_delta_grounded_mean": (
        "stage-2 winner grounded_mean − baseline grounded_mean; "
        "artifact key 'delta_grounded_mean'"
    ),
}


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-metric means over a list of per-query rubric dicts.

    Terminology (see ``METRIC_DEFINITIONS``): ``grounded_mean`` is the mean
    rubric score of the grounded metrics, ``grounded_rate`` is the fraction of
    runs the EvidenceGate passed, and ``overall_mean`` (alias ``overall``) is
    the headline score mixing grounded and fallback quality.
    """
    metric_means: Dict[str, Optional[float]] = {}
    metric_counts: Dict[str, int] = {}
    for metric in list(GROUNDED_METRICS) + ["fallback"]:
        applicable = [
            r["metrics"][metric] for r in records if r["metrics"].get(metric) is not None
        ]
        metric_means[metric] = mean(applicable)
        metric_counts[metric] = len(applicable)
    grounded = mean([metric_means[m] for m in GROUNDED_METRICS])
    fallback = metric_means["fallback"]
    overall = mean([v for v in (grounded, fallback) if v is not None])
    grounded_rate = mean([1.0 if r.get("grounded") else 0.0 for r in records])
    return {
        "metric_means": metric_means,
        "metric_applicable_n": metric_counts,
        "grounded_mean": grounded,
        "grounded_rate": grounded_rate,
        "fallback_mean": fallback,
        "overall_mean": overall,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# Rate-limit budget tracking
# ---------------------------------------------------------------------------

def estimate_output_tokens(answer_words: int) -> int:
    """English ≈1.3 words per token — used only for cached-run accounting."""
    return int(math.ceil((answer_words or 0) / 1.3))


def measure_logged_tokens() -> int:
    """Tokens already consumed this session (Groq logs from earlier runs)."""
    total = 0
    for log_path in (
        Path("/tmp/opencode/prompt_full.log"),
        Path("/tmp/opencode/prompt_full2.log"),
    ):
        try:
            text = log_path.read_text()
        except OSError:
            continue
        total += sum(int(m) for m in re.findall(r"using model=\S+ \((\d+) tokens\)", text))
    try:
        smoke = json.loads(Path("/tmp/opencode/prompt_smoke.json").read_text())
        runs = list(smoke.get("per_run", {}).values())
        total += sum(r.get("prompt_tokens") or 0 for r in runs)
        total += 300 * len(runs)  # measured average completion size
    except (OSError, ValueError):
        pass
    return total


class TokenLedger:
    """Tracks input/output tokens and stops the experiment before budgets break."""

    def __init__(
        self,
        prior_today: int = DEFAULT_PRIOR_TODAY_TOKENS,
        experiment_limit: int = LIMITS["experiment_target_tokens"],
        daily_limit: int = LIMITS["daily_token_budget"],
        buffer: int = DAILY_BUFFER_TOKENS,
    ):
        self.prior_today = prior_today
        self.experiment_limit = experiment_limit
        self.daily_limit = daily_limit
        self.buffer = buffer
        self.new_input = 0
        self.new_output = 0
        self.cached_input = 0
        self.cached_output = 0
        self.new_calls = 0
        self.cached_calls = 0
        self.stopped_reason: Optional[str] = None
        self.est_call_tokens = 2500

    # -- charging ---------------------------------------------------------
    def add_new(self, input_tokens: int, output_tokens: int) -> None:
        self.new_input += int(input_tokens or 0)
        self.new_output += int(output_tokens or 0)
        self.new_calls += 1
        spent = (input_tokens or 0) + (output_tokens or 0)
        if spent:
            if self.new_calls == 1:
                self.est_call_tokens = spent
            else:  # rolling average
                self.est_call_tokens = int(
                    (self.est_call_tokens * (self.new_calls - 1) + spent) / self.new_calls
                )

    def add_cached(self, input_tokens: int, output_tokens: int) -> None:
        self.cached_input += int(input_tokens or 0)
        self.cached_output += int(output_tokens or 0)
        self.cached_calls += 1

    # -- budgets ----------------------------------------------------------
    def experiment_total(self) -> int:
        return self.new_input + self.new_output + self.cached_input + self.cached_output

    def today_total(self) -> int:
        return self.prior_today + self.new_input + self.new_output

    def can_call(self) -> bool:
        if self.stopped_reason:
            return False
        est = self.est_call_tokens
        if self.experiment_total() + est > self.experiment_limit:
            self.stop("experiment_token_target_exceeded")
            return False
        if self.today_total() + est + self.buffer > self.daily_limit:
            self.stop("daily_token_budget_exceeded")
            return False
        return True

    def stop(self, reason: str) -> None:
        if not self.stopped_reason:
            self.stopped_reason = reason
            logger.warning("Experiment stopped: %s", reason)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "limits": dict(LIMITS),
            "prior_today_estimate_tokens": self.prior_today,
            "new_calls": self.new_calls,
            "cached_calls": self.cached_calls,
            "new_input_tokens": self.new_input,
            "new_output_tokens": self.new_output,
            "cached_input_tokens": self.cached_input,
            "cached_output_tokens": self.cached_output,
            "cached_output_note": "estimated at 1.3 words/token for cached runs",
            "experiment_input_tokens": self.new_input + self.cached_input,
            "experiment_output_tokens": self.new_output + self.cached_output,
            "experiment_total_tokens": self.experiment_total(),
            "experiment_target_tokens": self.experiment_limit,
            "daily_spent_estimate_tokens": self.today_total(),
            "daily_token_budget": self.daily_limit,
            "daily_remaining_estimate_tokens": self.daily_limit - self.today_total(),
            "stopped_reason": self.stopped_reason,
        }


# ---------------------------------------------------------------------------
# Run keys, caching and migration of runs from earlier in this session
# ---------------------------------------------------------------------------

def run_key(arm_name: str, qid: Any) -> str:
    return f"{arm_name}:q{qid}" if isinstance(qid, int) else f"{arm_name}:{qid}"


def migrate_legacy_runs(
    previous: Dict[str, Any],
    previous_model: Optional[str] = None,
    previous_temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Reuse runs from the superseded single-stage harness when they match.

    Only standard ground-truth queries (integer qid) are kept; the old
    out-of-corpus probe runs are not part of the two-stage design.
    """
    if previous_model not in (None, MODEL_EXPECTED) or previous_temperature not in (None, TEMPERATURE):
        return {}
    migrated: Dict[str, Any] = {}
    for key, run in (previous or {}).items():
        arm = run.get("arm")
        qi = run.get("query_index")
        if arm and isinstance(qi, int):
            record = dict(run)
            record["cached"] = True
            migrated[run_key(arm, qi)] = record
    return migrated


def charge_cached_runs(runs: Dict[str, Any], ledger: TokenLedger) -> None:
    """Count reused runs against the experiment budget."""
    for run in runs.values():
        if not run.get("cached"):
            continue
        ledger.add_cached(
            run.get("prompt_tokens") or 0,
            estimate_output_tokens(run.get("answer_words") or 0),
        )


def arm_records(
    runs: Dict[str, Any], arm_name: str, qids: Sequence[Any]
) -> List[Dict[str, Any]]:
    out = []
    for qid in qids:
        record = runs.get(run_key(arm_name, qid))
        if record is not None:
            out.append(record)
    return out


# ---------------------------------------------------------------------------
# Live experiment machinery (ES + Groq)
# ---------------------------------------------------------------------------

def question_for(qid: Any, ground_truth_queries: Sequence[Dict[str, Any]]) -> str:
    if qid == USER_QUERY["qid"]:
        return USER_QUERY["question"]
    return ground_truth_queries[int(qid) - 1]["query"]


def _build_messages(arm: Dict[str, str], question: str, docs, fallback: bool):
    from rag.prompt_builders import TemplatePromptBuilder

    if arm["kind"] == "variant":
        builder = TemplatePromptBuilder(variant=arm["value"])
        style = "detailed"
    else:
        # Historic production prompt (pre-adoption) — variant=None pins it.
        builder = TemplatePromptBuilder(variant=None)
        style = arm["value"]
    if fallback:
        return builder.build_fallback_prompt(question, style=style), builder
    return builder.build_prompt(question, list(docs), style=style), builder


_next_call_at = 0.0


def _throttle() -> None:
    """Block until the next call may be issued (global minimum spacing)."""
    global _next_call_at
    now = time.monotonic()
    wait = _next_call_at - now
    if wait > 0:
        time.sleep(wait)
    _next_call_at = time.monotonic() + CALL_INTERVAL_SECONDS


def _defer(seconds: float) -> None:
    """Push the next allowed call time out after a rate-limit response."""
    global _next_call_at
    _next_call_at = max(_next_call_at, time.monotonic() + seconds)


def _generate(llm, messages, ledger: TokenLedger, max_attempts: int = 3) -> Dict[str, Any]:
    """Conservative generation: ≤3 attempts, fixed 30 s / 60 s 429 waits."""
    from groq import RateLimitError

    for attempt in range(1, max_attempts + 1):
        _throttle()
        try:
            result = llm.generate(messages, temperature=TEMPERATURE, max_tokens=1024)
        except RateLimitError:
            if attempt == max_attempts:
                raise
            wait = 30.0 if attempt == 1 else 60.0
            logger.warning(
                "429 rate limit (attempt %d/%d) — waiting %.0f s", attempt, max_attempts, wait
            )
            _defer(wait)
        except Exception as exc:
            if attempt == max_attempts:
                raise
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, max_attempts, exc)
            _defer(30.0)
        else:
            ledger.add_new(result.get("prompt_tokens", 0), result.get("completion_tokens", 0))
            return result
    raise RuntimeError("unreachable")  # pragma: no cover


class RetrievalContext:
    """Production retrieval, computed once per query and reused by every arm."""

    def __init__(self, pipeline):
        import config

        self._config = config
        self._pipeline = pipeline
        self._strategy = pipeline._get_strategy("hybrid")
        self._reranker = pipeline.reranker
        self._gate = pipeline._get_evidence_gate()
        self._cache: Dict[Any, Dict[str, Any]] = {}

    def get(self, qid: Any, question: str) -> Dict[str, Any]:
        if qid in self._cache:
            return self._cache[qid]
        raw = self._strategy.search(question, top_k=self._config.RETRIEVAL_CANDIDATE_POOL)
        final = self._reranker.rerank(question, raw, top_n=self._config.RERANK_TOP_N)
        decision = self._gate.evaluate(question, final)
        grounded = bool(decision.sufficient)
        docs = list(decision.supporting_docs)
        citations: List[Any] = []
        if grounded:
            from rag.citations import build_citations

            citations = list(build_citations(docs))
        entry = {
            "qid": qid,
            "question": question,
            "grounded": grounded,
            "trust_level": decision.trust_level,
            "reason": decision.reason,
            "docs": docs,
            "citations": citations,
            "context_texts": [d.text for d in docs],
            "retrieval_summary": [
                {
                    "chunk_id": d.chunk_id,
                    "file_name": d.file_name,
                    "rerank_score": d.rerank_score,
                    "dataset_type": (d.metadata or {}).get("dataset_type"),
                    "page_number": d.page_number,
                }
                for d in final
            ],
            "supporting_chunk_ids": [d.chunk_id for d in docs],
        }
        self._cache[qid] = entry
        return entry


def _run_arm(
    ctx: RetrievalContext,
    arm: Dict[str, str],
    qid: Any,
    question: str,
    ledger: TokenLedger,
    runs: Dict[str, Any],
    llm,
    output_path: Path,
    state: Dict[str, int],
) -> Optional[Dict[str, Any]]:
    """Produce (or reuse) one run; returns the run record or None."""
    key = run_key(arm["name"], qid)
    existing = runs.get(key)
    if existing is not None:
        return existing
    if ledger.stopped_reason:
        return None
    if not ledger.can_call():
        return None

    ret = ctx.get(qid, question)
    messages, _ = _build_messages(arm, question, ret["docs"], fallback=not ret["grounded"])
    try:
        generation = _generate(llm, messages, ledger)
    except Exception as exc:
        state["consecutive_failures"] += 1
        logger.warning("generation failed for %s: %s", key, exc)
        if state["consecutive_failures"] >= 3:
            ledger.stop("repeated_api_failures")
        return None
    state["consecutive_failures"] = 0

    model_answer = generation["answer"]
    if ret["grounded"]:
        final_answer = model_answer
        metrics = score_answer(
            question, final_answer, ret["citations"], ret["context_texts"]
        )
        fallback_checks = None
    else:
        final_answer = f"{UNTRUSTED_ANSWER_WARNING}\n\n{model_answer}"
        fallback_checks, score = score_fallback(model_answer, final_answer)
        metrics = {
            "source_adherence": None,
            "unsupported_claims": None,
            "citation_correctness": None,
            "answer_completeness": None,
            "verbosity": None,
            "red_flag_handling": None,
            "symptom_vs_diagnosis": None,
            "fallback": score,
        }

    runs[key] = {
        "arm": arm["name"],
        "qid": qid,
        "question": question,
        "grounded": ret["grounded"],
        "trust_level": ret["trust_level"],
        "reason": ret["reason"],
        "n_citations": len(ret["citations"]),
        "citations": [
            {"index": c.index, "label": c.label, "trust": c.trust}
            for c in ret["citations"]
        ],
        "supporting_chunk_ids": ret["supporting_chunk_ids"],
        "answer_words": word_count(final_answer),
        "metrics": metrics,
        "fallback_checks": fallback_checks,
        "model_answer": model_answer,
        "final_answer": final_answer,
        "prompt_tokens": generation.get("prompt_tokens"),
        "completion_tokens": generation.get("completion_tokens"),
        "response_time_ms": generation.get("response_time_ms"),
        "cached": False,
    }
    _save_work(output_path, runs, ledger)
    logger.info(
        "run %s grounded=%s metrics_mean=%s tokens=%s+%s",
        key, ret["grounded"],
        mean([v for v in metrics.values() if v is not None]),
        generation.get("prompt_tokens"), generation.get("completion_tokens"),
    )
    return runs[key]


def _save_work(output_path: Path, runs: Dict[str, Any], ledger: TokenLedger) -> None:
    payload = {
        "experiment": "prompt_experiment",
        "phase": 9,
        "design": "two_stage",
        "in_progress": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_EXPECTED,
        "temperature": TEMPERATURE,
        "token_usage": ledger.as_dict(),
        "per_run": runs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _ingest_user_stub() -> None:
    from ingestion.medical_index import clear_user_data
    from ingestion.pipeline import IngestionPipeline

    class _UserProtocol:
        name = USER_QUERY["file_name"]

        def getvalue(self):
            return USER_QUERY["content"].encode("utf-8")

    clear_user_data()
    result = IngestionPipeline().run_uploaded_files([_UserProtocol()])
    logger.info(
        "user-only fixture uploaded: %s (%s chunks)",
        result.get("status"), result.get("indexed_count"),
    )


def _restore_user_index() -> None:
    from ingestion.medical_index import clear_user_data, user_vector_store

    clear_user_data()
    user_vector_store().create_index(dimension=384, recreate=False)


# ---------------------------------------------------------------------------
# Stage runners and reporting
# ---------------------------------------------------------------------------

def compute_stage1_report(
    runs: Dict[str, Any], ground_truth_queries: Sequence[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Score stage 1 from cached runs; None until every run exists."""
    arms_report = []
    complete = True
    for arm in VARIANT_ARMS:
        records = arm_records(runs, arm["name"], STAGE1_QIDS)
        if len(records) != len(STAGE1_QIDS):
            complete = False
        arms_report.append({
            "name": arm["name"],
            "variant": arm["value"],
            "n_runs": len(records),
            "n_grounded": sum(1 for r in records if r["grounded"]),
            "n_cached": sum(1 for r in records if r.get("cached")),
            **(aggregate(records) if records else {}),
        })
    if not complete:
        return None

    scored = [a for a in arms_report if a.get("overall_mean") is not None]
    if not scored:
        return None
    ranked = sorted(scored, key=lambda a: (-a["overall_mean"], a["name"]))
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else ranked[0]
    queries = []
    for sel in STAGE1_SELECTION:
        qid = sel["qid"]
        probe = runs.get(run_key(VARIANT_ARMS[0]["name"], qid)) or {}
        queries.append({
            "qid": qid,
            "category": sel["category"],
            "question": question_for(qid, ground_truth_queries),
            "grounded": probe.get("grounded"),
            "trust_level": probe.get("trust_level"),
            "n_citations": probe.get("n_citations"),
        })
    return {
        "n_queries": len(STAGE1_SELECTION),
        "n_arms": len(VARIANT_ARMS),
        "max_llm_calls": STAGE1_MAX_CALLS,
        "categories_covered": STAGE1_CATEGORIES,
        "queries": queries,
        "arms": arms_report,
        "metric_terminology": (
            "winner is ranked by 'overall_mean' (alias 'overall'); "
            "'grounded_mean' is the grounded-metric quality score"
        ),
        "winner": winner["name"],
        "winner_overall": winner["overall_mean"],
        "winner_overall_mean": winner["overall_mean"],
        "winner_grounded_mean": winner["grounded_mean"],
        "margin": round(
            (winner["overall_mean"] or 0) - (runner_up["overall_mean"] or 0), 4
        ),
    }


def compute_stage2_report(
    runs: Dict[str, Any],
    ground_truth_queries: Sequence[Dict[str, Any]],
    winner_arm: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    arms = [winner_arm, BASELINE_ARM]
    arms_report = []
    complete = True
    for arm in arms:
        records = arm_records(runs, arm["name"], STAGE2_QIDS)
        if len(records) != len(STAGE2_QIDS):
            complete = False
        arms_report.append({
            "name": arm["name"],
            "n_runs": len(records),
            "n_grounded": sum(1 for r in records if r["grounded"]),
            "n_cached": sum(1 for r in records if r.get("cached")),
            **(aggregate(records) if records else {}),
        })
    if not complete:
        return None

    per_query = []
    for qi in STAGE2_QIDS:
        win = runs.get(run_key(winner_arm["name"], qi))
        base = runs.get(run_key(BASELINE_ARM["name"], qi))
        if not win or not base:
            continue
        win_overall = mean([v for v in win["metrics"].values() if v is not None])
        base_overall = mean([v for v in base["metrics"].values() if v is not None])
        per_query.append({
            "qid": qi,
            "question": question_for(qi, ground_truth_queries),
            "grounded": win["grounded"],
            "trust_level": win["trust_level"],
            "winner_overall": win_overall,
            "baseline_overall": base_overall,
            "delta": round((win_overall or 0) - (base_overall or 0), 4),
        })

    winner_report, baseline_report = arms_report
    winner_overall = winner_report.get("overall_mean")
    baseline_overall = baseline_report.get("overall_mean")
    delta = None
    if winner_overall is not None and baseline_overall is not None:
        delta = round(winner_overall - baseline_overall, 4)
    winner_grounded = winner_report.get("grounded_mean")
    baseline_grounded = baseline_report.get("grounded_mean")
    delta_grounded = None
    if winner_grounded is not None and baseline_grounded is not None:
        delta_grounded = round(winner_grounded - baseline_grounded, 4)
    return {
        "n_queries": len(STAGE2_QIDS),
        "n_arms": 2,
        "max_llm_calls": STAGE2_MAX_CALLS,
        "winner_arm": winner_arm["name"],
        "baseline_arm": BASELINE_ARM["name"],
        "arms": arms_report,
        "per_query": per_query,
        "metric_terminology": (
            "'overall'/'overall_mean' = headline score; 'grounded_mean' = "
            "grounded-metric quality score (see artifact metric_definitions)"
        ),
        "winner_overall": winner_overall,
        "baseline_overall": baseline_overall,
        "delta": delta,
        "delta_overall_mean": delta,
        "winner_grounded_mean": winner_grounded,
        "baseline_grounded_mean": baseline_grounded,
        "delta_grounded_mean": delta_grounded,
    }


def build_examples(
    runs: Dict[str, Any],
    ground_truth_queries: Sequence[Dict[str, Any]],
    winner_arm: Dict[str, str],
    topn: int = 3,
) -> Dict[str, Any]:
    """Biggest per-query wins and losses of the stage-1 winner vs baseline."""
    rows = []
    for qi in STAGE2_QIDS:
        win = runs.get(run_key(winner_arm["name"], qi))
        base = runs.get(run_key(BASELINE_ARM["name"], qi))
        if not win or not base:
            continue
        win_overall = mean([v for v in win["metrics"].values() if v is not None]) or 0.0
        base_overall = mean([v for v in base["metrics"].values() if v is not None]) or 0.0
        rows.append({
            "qid": qi,
            "question": question_for(qi, ground_truth_queries),
            "delta": round(win_overall - base_overall, 4),
            "winner_overall": win_overall,
            "baseline_overall": base_overall,
            "winner_answer": win["final_answer"][:400],
            "baseline_answer": base["final_answer"][:400],
        })
    rows.sort(key=lambda r: r["delta"], reverse=True)
    return {
        "improvements": [r for r in rows[:topn] if r["delta"] > 0],
        "regressions": [r for r in reversed(rows[-topn:]) if r["delta"] < 0],
    }


def decide_recommendation(stage2: Optional[Dict[str, Any]]) -> str:
    if stage2 is None:
        return "incomplete — stage 2 not evaluated"
    if stage2.get("delta") is None:
        return "incomplete — stage 2 scores missing"
    winner = stage2["winner_arm"]
    if winner == BASELINE_ARM["name"]:
        return "keep current prompt (production prompt won stage 2)"
    if stage2["delta"] > 0.01:
        msg = (
            f"adopt {winner} (stage-2 overall_mean {stage2['winner_overall']} vs baseline "
            f"{stage2['baseline_overall']}, +{stage2['delta']}"
        )
        delta_grounded = stage2.get("delta_grounded_mean")
        if delta_grounded is not None:
            msg += f"; grounded_mean {delta_grounded:+g}"
        return msg + ")"
    return (
        f"keep current prompt (stage-2 overall_mean delta {stage2['delta']} not > 0.01; "
        f"best variant was {winner})"
    )


def run_experiment(
    stage: str = "all",
    output_path: Path = OUTPUT_PATH,
    resume: bool = True,
    prior_today: Optional[int] = None,
    session_used: Optional[int] = None,
) -> Dict[str, Any]:
    import config
    from rag.llm_providers import GroqLLMProvider
    from rag.pipeline import RAGPipeline

    if config.LLM_MODEL != MODEL_EXPECTED:
        raise SystemExit(
            f"Phase 9 must use {MODEL_EXPECTED!r} for every call; config has "
            f"{config.LLM_MODEL!r}. Refusing to mix models."
        )

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    ground_truth_queries = ground_truth["queries"]

    # -- resume + migrate runs cached from earlier in this session --------
    runs: Dict[str, Any] = {}
    previous: Dict[str, Any] = {}
    if resume and output_path.exists():
        try:
            previous = json.loads(output_path.read_text())
            if previous.get("model") != MODEL_EXPECTED or previous.get("temperature") != TEMPERATURE:
                logger.warning("previous artifact used a different model/temperature — ignoring")
                previous = {}
            elif previous.get("design") == "two_stage":
                runs = dict(previous.get("per_run") or {})
            else:
                # superseded single-stage harness: keep its standard-query runs
                runs = migrate_legacy_runs(previous.get("per_run") or {})
        except (OSError, ValueError) as exc:  # pragma: no cover - corrupt artifact
            logger.warning("could not reuse previous artifact: %s", exc)
            runs = {}
            previous = {}

    prior = prior_today
    if prior is None:
        prior = (session_used if session_used is not None else DEFAULT_PRIOR_TODAY_TOKENS)
    ledger = TokenLedger(prior_today=prior)
    charge_cached_runs(runs, ledger)

    pipeline = RAGPipeline()  # real store/reranker/gate; no repository writes
    llm = GroqLLMProvider()
    ctx = RetrievalContext(pipeline)
    state = {"consecutive_failures": 0}

    started = time.time()
    stage1_report: Optional[Dict[str, Any]] = None
    stage2_report: Optional[Dict[str, Any]] = None
    user_upload_live = False

    try:
        # ---------------- STAGE 1 -------------------------------------
        if stage in ("1", "all"):
            missing_u1 = [
                arm for arm in VARIANT_ARMS
                if run_key(arm["name"], USER_QUERY["qid"]) not in runs
            ]
            if missing_u1:
                _ingest_user_stub()
                user_upload_live = True
            calls_before = ledger.new_calls
            for sel in STAGE1_SELECTION:
                if ledger.stopped_reason:
                    break
                qid = sel["qid"]
                question = question_for(qid, ground_truth_queries)
                for arm in VARIANT_ARMS:
                    if ledger.stopped_reason:
                        break
                    _run_arm(
                        ctx, arm, qid, question, ledger, runs, llm,
                        output_path, state,
                    )
            stage1_calls = ledger.new_calls - calls_before
            stage1_report = compute_stage1_report(runs, ground_truth_queries)
            if stage1_report is not None:
                stage1_report["llm_calls_made"] = stage1_calls
        else:
            stage1_calls = 0
            stage1_report = compute_stage1_report(runs, ground_truth_queries)
            if stage1_report is None:
                stage1_report = (previous.get("stage1") or None)

        winner_name = stage1_report.get("winner") if stage1_report else None
        winner_arm = next(
            (a for a in VARIANT_ARMS if a["name"] == winner_name), None
        )
        if stage in ("2", "all") and winner_arm is None:
            raise SystemExit(
                "stage 2 needs a stage-1 winner — run stage 1 first "
                "(python3 -m evaluation.prompt_experiment --stage 1)"
            )

        # ---------------- STAGE 2 -------------------------------------
        if stage in ("2", "all"):
            calls_before = ledger.new_calls
            for qi in STAGE2_QIDS:
                if ledger.stopped_reason:
                    break
                question = question_for(qi, ground_truth_queries)
                for arm in (winner_arm, BASELINE_ARM):
                    if ledger.stopped_reason:
                        break
                    _run_arm(
                        ctx, arm, qi, question, ledger, runs, llm,
                        output_path, state,
                    )
            stage2_calls = ledger.new_calls - calls_before
            stage2_report = compute_stage2_report(
                runs, ground_truth_queries, winner_arm
            )
            if stage2_report is not None:
                stage2_report["llm_calls_made"] = stage2_calls
    finally:
        if user_upload_live:
            _restore_user_index()

    # -- assemble final artifact ----------------------------------------
    artifact = build_artifact(
        runs=runs,
        ledger=ledger,
        stage1=stage1_report,
        stage2=stage2_report,
        ground_truth_queries=ground_truth_queries,
        winner_arm=winner_arm if stage in ("2", "all") else None,
        elapsed=round(time.time() - started, 1),
        stage1_calls=stage1_calls,
        stage2_calls=stage2_calls if stage in ("2", "all") else 0,
    )
    if stage1_report is None and previous.get("stage1"):
        artifact["stage1"] = previous["stage1"]
    artifact["in_progress"] = bool(ledger.stopped_reason)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    return artifact


def build_artifact(
    runs: Dict[str, Any],
    ledger: TokenLedger,
    stage1: Optional[Dict[str, Any]],
    stage2: Optional[Dict[str, Any]],
    ground_truth_queries: Sequence[Dict[str, Any]],
    winner_arm: Optional[Dict[str, str]],
    elapsed: float,
    stage1_calls: int,
    stage2_calls: int,
) -> Dict[str, Any]:
    import config

    from rag.prompt_builders import ADOPTED_GROUNDED_VARIANT, TemplatePromptBuilder

    examples: Dict[str, Any] = {"improvements": [], "regressions": []}
    if stage2 and winner_arm:
        examples = build_examples(runs, ground_truth_queries, winner_arm)

    prompt_texts = {
        arm["name"]: TemplatePromptBuilder(variant=arm["value"]).build_prompt(
            "<question>", [], style="detailed"
        )[0]["content"]
        for arm in VARIANT_ARMS
    }
    prompt_texts[BASELINE_ARM["name"]] = TemplatePromptBuilder(variant=None).build_prompt(
        "<question>", [], style=BASELINE_ARM["value"]
    )[0]["content"]

    usage = ledger.as_dict()
    usage["stage1_new_calls"] = stage1_calls
    usage["stage2_new_calls"] = stage2_calls

    return {
        "experiment": "prompt_experiment",
        "phase": 9,
        "design": "two_stage",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "in_progress": bool(ledger.stopped_reason),
        "model": MODEL_EXPECTED,
        "temperature": TEMPERATURE,
        "ground_truth_file": str(GROUND_TRUTH_PATH.relative_to(ROOT)),
        "retrieval_config": {
            "read_indexes": config.MEDICAL_RETRIEVAL_INDEXES,
            "rrf_weights": [0.5, 0.5],
            "rrf_k": 60,
            "candidate_pool": config.RETRIEVAL_CANDIDATE_POOL,
            "rerank_top_n": config.RERANK_TOP_N,
            "embedding_model": config.EMBEDDING_MODEL,
            "evidence_policy": "evidence-v1",
            "query_rewriting": False,
            "unchanged": True,
        },
        "rubric": METRIC_RUBRIC,
        "metric_definitions": METRIC_DEFINITIONS,
        "adoption": {
            "adopted_variant": ADOPTED_GROUNDED_VARIANT,
            "adopted_after": "stage 2 of this experiment (see recommendation)",
            "production_path": (
                "rag/prompt_builders.py — TemplatePromptBuilder() default; "
                "rag/pipeline.py builds grounded prompts with it"
            ),
            "fallback_prompt": "unchanged (BasePromptBuilder.FALLBACK_SYSTEM_PROMPT)",
            "retrieval_and_gate": "unchanged",
        },
        "prompts": prompt_texts,
        "fallback_prompt": BasePromptBuilder.FALLBACK_SYSTEM_PROMPT,
        "fallback_prompt_source": FALLBACK_PROMPT_PATH_HINT,
        "user_fixture": USER_QUERY,
        "stage1": stage1,
        "stage2": stage2,
        "examples": examples,
        "winner": (stage1 or {}).get("winner"),
        "recommendation": decide_recommendation(stage2)
        if stage2 or stage1 is None
        else "stage 1 complete — run stage 2 for the final recommendation",
        "stopped_reason": ledger.stopped_reason,
        "token_usage": usage,
        "elapsed_seconds": elapsed,
        "per_run": runs,
    }


def _print_report(artifact: Dict[str, Any]) -> None:
    print("\n=== PHASE 9 — two-stage prompt experiment ===")
    print(f"model: {artifact['model']}  temperature: {artifact['temperature']}")
    stage1 = artifact.get("stage1")
    if stage1:
        print(f"\nSTAGE 1 — {stage1['n_queries']} queries x {stage1['n_arms']} variants "
              f"(max {stage1['max_llm_calls']} calls, made {stage1.get('llm_calls_made')})")
        for arm in stage1["arms"]:
            print(f"  {arm['name']:28s} n={arm['n_runs']:2d} "
                  f"grounded_mean={arm.get('grounded_mean')} "
                  f"fallback_mean={arm.get('fallback_mean')} "
                  f"overall_mean={arm.get('overall_mean')}")
        print(f"  winner: {stage1['winner']} (overall_mean {stage1['winner_overall']}, "
              f"margin {stage1['margin']})")
    stage2 = artifact.get("stage2")
    if stage2:
        print(f"\nSTAGE 2 — {stage2['n_queries']} queries x winner vs baseline "
              f"(max {stage2['max_llm_calls']} calls, made {stage2.get('llm_calls_made')})")
        for arm in stage2["arms"]:
            print(f"  {arm['name']:28s} n={arm['n_runs']:2d} "
                  f"grounded_mean={arm.get('grounded_mean')} "
                  f"fallback_mean={arm.get('fallback_mean')} "
                  f"overall_mean={arm.get('overall_mean')}")
        print(f"  delta overall_mean  (winner - baseline): {stage2.get('delta_overall_mean', stage2['delta'])}")
        print(f"  delta grounded_mean (winner - baseline): {stage2.get('delta_grounded_mean')}")
        for kind in ("improvements", "regressions"):
            rows = artifact["examples"][kind]
            if rows:
                print(f"  {kind}:")
                for row in rows:
                    print(f"    q{row['qid']} {row['question'][:52]!r} delta={row['delta']}")
    usage = artifact["token_usage"]
    print("\nTOKEN USAGE")
    print(f"  new calls: {usage['new_calls']} "
          f"({usage['stage1_new_calls']} stage1 + {usage['stage2_new_calls']} stage2), "
          f"cached: {usage['cached_calls']}")
    print(f"  experiment: {usage['experiment_input_tokens']} in + "
          f"{usage['experiment_output_tokens']} out = "
          f"{usage['experiment_total_tokens']} / {usage['experiment_target_tokens']}")
    print(f"  daily estimate: {usage['daily_spent_estimate_tokens']} / "
          f"{usage['daily_token_budget']} (remaining "
          f"{usage['daily_remaining_estimate_tokens']})")
    if artifact.get("stopped_reason"):
        print(f"  STOPPED: {artifact['stopped_reason']}")
    print(f"\n  winner: {artifact.get('winner')}")
    print(f"  recommendation: {artifact['recommendation']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 9 two-stage prompt experiment")
    parser.add_argument("--stage", choices=("1", "2", "all"), default="all")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--prior-today", type=int, default=None,
        help="tokens already spent today outside this session's logs "
             f"(default {DEFAULT_PRIOR_TODAY_TOKENS})",
    )
    parser.add_argument(
        "--session-used", type=int, default=None,
        help="override measured session token usage (logs + smoke)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.session_used is None and args.prior_today is None:
        measured = measure_logged_tokens()
        logger.info("session usage measured from logs: %d tokens (+%d prior margin)",
                    measured, DEFAULT_PRIOR_TODAY_TOKENS)
        prior = measured + DEFAULT_PRIOR_TODAY_TOKENS
    else:
        prior = args.prior_today if args.prior_today is not None else (
            args.session_used + DEFAULT_PRIOR_TODAY_TOKENS
        )

    artifact = run_experiment(
        stage=args.stage,
        output_path=Path(args.output),
        resume=not args.no_resume,
        prior_today=prior,
    )
    _print_report(artifact)
    print(f"\nWrote {args.output}")
    return 0 if not artifact.get("stopped_reason") else 2


if __name__ == "__main__":
    raise SystemExit(main())
