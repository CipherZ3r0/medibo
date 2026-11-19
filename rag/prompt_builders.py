"""
Prompt Builders — Abstract interface and template-based implementation for RAG prompts.
"""

from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Any

from domain.models import SearchResult
from rag.evidence import SOURCE_TRUSTED, SOURCE_UNKNOWN, SOURCE_USER, classify_source

logger = logging.getLogger(__name__)

#: Phase 9 adoption: ``variant_structured_brief`` won the two-stage evaluation
#: (stage-2 overall_mean 0.8438 vs 0.7275 → delta +0.1163; grounded_mean delta
#: +0.2325 — artifact: data/medical/prompt_experiment_results.json) and is now
#: the production grounded-answer prompt, i.e. the default for
#: ``TemplatePromptBuilder()``.  Pass ``variant=None`` to get the historic
#: style-based prompt (kept byte-for-byte as the experiment's baseline arm).
ADOPTED_GROUNDED_VARIANT = "structured_brief"

#: Phase 9.5 adoption: the fallback prompt experiment (`evaluation/
#: fallback_prompt_experiment.py`) evaluated the legacy fallback prompt against
#: three concise variants and adopted the winner as the production fallback
#: prompt (artifact: data/medical/fallback_prompt_experiment_results.json).
#: The value of ``FALLBACK_SYSTEM_PROMPT`` below IS that adopted prompt.
ADOPTED_FALLBACK_VARIANT = None


class BasePromptBuilder(ABC):
    """Abstract Base Class for building LLM context prompts."""

    #: Shown to the model when the evidence gate found no sufficient evidence.
    #: The answer must be general-knowledge only and must never be attributed
    #: to any document or "knowledge base".  Phase 9.5 adopted an optimized
    #: concise prompt here (see ``ADOPTED_FALLBACK_VARIANT``); the variants
    #: compared by the experiment are kept in ``FALLBACK_VARIANTS``.
    FALLBACK_SYSTEM_PROMPT = (
        "You are Medibo's fallback responder. The Medibo knowledge base has no "
        "relevant document for this question, so answer ONLY from your own general "
        "knowledge — never imply Medibo or any document supports the answer, and "
        "never cite, quote or name any source.\n"
        "Reply in this exact order, keeping the whole reply under 90 words:\n"
        "1. Lead with \"From general knowledge:\", then answer the question directly "
        "in one sentence.\n"
        "2. Add at most two short supporting sentences or bullets — only what "
        "genuinely helps.\n"
        "3. Add one calm \"Seek care:\" line ONLY when the question involves possible "
        "emergency or urgent symptoms (real warning signs); otherwise omit it.\n"
        "Rules: no preamble, no repetition, no closing summary; no headings, tables "
        "or long lists; be explicit about uncertainty; never give a diagnosis or "
        "personal medical advice — say symptoms are signs a clinician must interpret; "
        "avoid alarm and fear language."
    )

    #: Phase 9.5 fallback prompt variants plus the pre-experiment legacy prompt,
    #: as evaluated by ``evaluation/fallback_prompt_experiment.py``.  Production
    #: always uses ``FALLBACK_SYSTEM_PROMPT`` (set to the adopted variant).
    FALLBACK_VARIANTS: Dict[str, str] = {
        "legacy": (
            "You are a helpful medical information assistant. The Medibo knowledge base "
            "does not contain information relevant to this question, so answer from your "
            "own general knowledge only. Do NOT cite, quote or attribute your answer to "
            "any document, source, file or knowledge base, and never invent references. "
            "Be explicit about uncertainty, and remember this is general information "
            "rather than personal medical advice."
        ),
        "direct_brief": (
            "You are Medibo's fallback responder. The Medibo knowledge base has no "
            "relevant document for this question, so answer ONLY from your own general "
            "knowledge — never imply Medibo or any document supports the answer, and "
            "never cite, quote or name any source.\n"
            "Reply in this exact order, keeping the whole reply under 90 words:\n"
            "1. Lead with \"From general knowledge:\", then answer the question "
            "directly in one sentence.\n"
            "2. Add at most two short supporting sentences or bullets — only what "
            "genuinely helps.\n"
            "3. Add one calm \"Seek care:\" line ONLY when the question involves "
            "possible emergency or urgent symptoms (real warning signs); otherwise "
            "omit it.\n"
            "Rules: no preamble, no repetition, no closing summary; no headings, "
            "tables or long lists; be explicit about uncertainty; never give a "
            "diagnosis or personal medical advice — say symptoms are signs a "
            "clinician must interpret; avoid alarm and fear language."
        ),
        "structured_minimal": (
            "You are Medibo's fallback responder. The Medibo knowledge base does not "
            "contain a relevant document for this question, so answer ONLY from your "
            "own general knowledge — never attribute the answer to Medibo, a document, "
            "or any source, and never invent references.\n"
            "Reply with this exact format, keeping the whole reply under 110 words:\n"
            "**Answer** — begin with \"From general knowledge:\", then answer the "
            "question in one or two sentences.\n"
            "**Details** — at most 3 short bullets, only if they add useful "
            "information; omit the section entirely when they do not.\n"
            "**Seek care** — one short line ONLY when the question involves possible "
            "emergency or urgent symptoms (calm wording, real warning signs); "
            "otherwise omit it entirely.\n"
            "Rules: symptoms are signs, never a diagnosis; no personal medical advice; "
            "be explicit about uncertainty; use the same formatting for every answer; "
            "no tables or long lists; avoid alarm, fear language or reassurance "
            "beyond the facts."
        ),
        "compact_paragraph": (
            "You are Medibo's fallback responder. This question has no matching "
            "document in the Medibo knowledge base, so answer ONLY from your own "
            "general knowledge — never imply Medibo or any source supports the "
            "answer, and never invent citations.\n"
            "Reply in one short paragraph of at most 4 sentences and under 120 words:\n"
            "1. Open with \"From general knowledge:\" and the direct answer.\n"
            "2. Add at most one or two brief supportive facts that genuinely help.\n"
            "3. End, ONLY if the question involves possible emergency or urgent "
            "symptoms, with one calm sentence on what to do; otherwise stop.\n"
            "Rules: no disclaimers beyond the opening note, no repetition, no "
            "summary; no headings, bullets or tables; be explicit about uncertainty; "
            "never give a diagnosis or personal medical advice; avoid alarm and fear "
            "language."
        ),
    }

    @abstractmethod
    def build_prompt(
        self, question: str, search_results: List[SearchResult], style: str = "detailed"
    ) -> List[Dict[str, str]]:
        """Construct system & user messages list for chat completions API."""
        pass

    def build_fallback_prompt(
        self, question: str, style: str = "detailed", variant: str = None
    ) -> List[Dict[str, str]]:
        """Prompt used when retrieval produced no sufficient evidence.

        Deliberately carries **no context documents**, so the model answers
        from general knowledge and cannot be tempted to fabricate citations.

        ``variant`` selects a Phase 9.5 fallback variant by name
        (``FALLBACK_VARIANTS``); when omitted the production fallback prompt
        (``FALLBACK_SYSTEM_PROMPT``, the adopted Phase 9.5 winner) is used.
        """
        system_prompt = self.FALLBACK_SYSTEM_PROMPT
        if variant is not None:
            try:
                system_prompt = self.FALLBACK_VARIANTS[variant]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown fallback variant {variant!r}; "
                    f"expected one of {sorted(self.FALLBACK_VARIANTS)}"
                ) from exc
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]


class TemplatePromptBuilder(BasePromptBuilder):
    """Template-driven prompt builder supporting concise, detailed, and structured styles.

    Phase 9 added named *prompt variants* for the grounded-answer path and the
    evaluation adopted ``variant_structured_brief`` as production: the default
    builder instance (``TemplatePromptBuilder()``) now emits that prompt, with
    its provenance context template.  ``variant=None`` still selects the
    historic style-based behaviour byte-for-byte (it is the baseline arm of
    ``evaluation/prompt_experiment.py``); any other named variant selects that
    variant explicitly.  The fallback prompt is shared across styles; Phase 9.5
    added concise fallback variants (see ``FALLBACK_VARIANTS``), with the
    adopted winner as the production ``FALLBACK_SYSTEM_PROMPT``.
    """

    SYSTEM_PROMPTS = {
        "concise": (
            "You are a helpful internal knowledge assistant. Answer the user's question "
            "based ONLY on the provided context. Be concise and direct. If the context "
            "doesn't contain enough information to answer, say so clearly."
        ),
        "detailed": (
            "You are a helpful internal knowledge assistant. Answer the user's question "
            "based ONLY on the provided context. Provide a comprehensive and detailed "
            "answer, explaining your reasoning. If the context doesn't contain enough "
            "information to answer completely, explain what information is available."
        ),
        "structured": (
            "You are a helpful internal knowledge assistant. Answer the user's question "
            "based ONLY on the provided context. Structure your response as follows:\n"
            "1. **Answer**: Direct answer to the question\n"
            "2. **Details**: Supporting details from the context\n"
            "3. **Sources**: Reference which document(s) the information came from\n"
        ),
    }

    #: Phase 9 grounded-answer prompt variants (evaluated by
    #: ``evaluation/prompt_experiment.py``).  Each entry may override the
    #: system prompt, the context template and the user template.
    PROMPT_VARIANTS: Dict[str, Dict[str, str]] = {
        "structured_brief": {
            "system": (
                "You are Medibo, a medical information assistant. Answer ONLY from the "
                "supplied context documents, which are numbered 1..N; the same numbers "
                "are shown to the reader as citations.\n"
                "Format your reply exactly as:\n"
                "**Answer** — one or two sentences that answer the question asked, first, "
                "with no preamble.\n"
                "**Details** — at most 5 short bullets of what the context supports; end "
                "every bullet that comes from context document n with [n].\n"
                "**Seek care** — one short line with the warning signs or urgency the "
                "context actually gives; if it gives none, write \"Seek care: not "
                "specified in the provided sources.\"\n"
                "Rules:\n"
                "- Answer the user's actual question first; do not open with "
                "background or disclaimers.\n"
                "- Symptoms are not a diagnosis: report symptoms as symptoms, never tell "
                "the user they have a condition, and note that diagnosis is a clinician's "
                "job if the question mixes the two.\n"
                "- Give basic/normal care (fluids, rest, home measures) only when the "
                "context supports it; do not invent care steps.\n"
                "- Include an escalation line only from the context's own warning signs; "
                "keep the tone calm and do not add alarm, fear language or emergencies "
                "the context does not state.\n"
                "- Attribute sources by issuing body and country (e.g. ICMR or MoHFW — "
                "India; MedlinePlus — global/US); keep Indian and global guidance "
                "distinct. If context documents disagree, say so briefly and give each "
                "position with its [n].\n"
                "- Claims only from the context: no numbers, doses, statistics or "
                "prognoses that are not there. Paraphrase — never reproduce context "
                "passages verbatim — and never invent sources or references."
            ),
        },
        "question_first": {
            "system": (
                "You are Medibo, a medical information assistant. Use ONLY the supplied "
                "context documents (numbered 1..N; the reader sees the same numbers as "
                "citations).\n"
                "Start with a direct one-sentence answer to the exact question. Then add "
                "at most 4 compact supporting sentences or bullets, each carrying the "
                "citation number of the document it came from (e.g. [2]). Stop when you "
                "have answered the question — no summaries, no closing advice, no "
                "restating the question. Target 80-140 words in total.\n"
                "Medical rules:\n"
                "- Distinguish symptoms from diagnosis: describe symptoms and signs; "
                "never assert what the user personally has or is diagnosed with.\n"
                "- Mention basic/normal self-care only where the context states it.\n"
                "- State warning signs and when to seek care plainly when the context "
                "gives them; otherwise say the context does not specify. Use the "
                "context's urgency, no more and no less — no alarmism.\n"
                "- Name each source's issuing body and country so Indian guidance "
                "(ICMR, MoHFW, NCDC) is never merged with global/US guidance "
                "(MedlinePlus, WHO); when sources conflict, present both with [n] and "
                "say they differ.\n"
                "- Every medical claim must be supported by the context; do not add "
                "external facts, and do not copy long passages from the context."
            ),
            "user_template": (
                "Context documents:\n{context}\n"
                "--- Question ---\n{question}\n\n"
                "Reply with the answer only."
            ),
        },
        "triage_concise": {
            "system": (
                "You are Medibo, a medical information assistant answering from supplied "
                "evidence only. Context documents are numbered 1..N and shown to the "
                "reader as citations [n].\n"
                "Keep the whole reply under 90 words: first line = the direct answer to "
                "the question; then at most 3 bullets of key supported facts with their "
                "[n]; last line = \"Seek care:\" plus the context's escalation advice, or "
                "\"Seek care: not specified in the provided sources.\"\n"
                "Never diagnose or tell the user what they have — symptoms are not a "
                "diagnosis. Include routine care only if the context supports it. Use only "
                "the context's facts and only its level of urgency: no alarm, no "
                "reassurance beyond it, no invented numbers. Keep Indian sources (ICMR, "
                "MoHFW, NCDC) and global sources (MedlinePlus, WHO) identified separately; "
                "if the context conflicts, note the disagreement in one clause with [n]. "
                "Paraphrase rather than quoting, and invent nothing."
            ),
        },
    }

    #: Appended to every grounded system prompt: a claim may only be presented
    #: as source-backed when the context actually supports it.
    GROUNDING_RULE = (
        " Only present a claim as source-backed when it is supported by the provided "
        "context documents, and never attribute anything to a document that is not in "
        "the context. Documents marked 'user-provided' are personal uploads, not part "
        "of the curated medical knowledge base — say so when you rely on them."
    )

    SOURCE_LABELS = {
        SOURCE_TRUSTED: "Trusted medical source (Medibo knowledge base)",
        SOURCE_USER: "User-provided upload (not part of the curated knowledge base)",
        SOURCE_UNKNOWN: "Unknown provenance — untrusted, do not attribute",
    }

    CONTEXT_TEMPLATE = """--- Context Document {index} ---
Source: {file_name}
Source type: {source_type}
{page_info}
Content:
{text}
"""

    #: Variant context template — exposes provenance (authority, region, URL,
    #: title, section) so the model can preserve Indian/global source
    #: distinctions and cite page/section accurately.
    CONTEXT_TEMPLATE_PROVENANCE = """--- Context Document {index} ---
Title: {title}
Source: {file_name}
Issued by: {authority}
Region: {region}
Source URL: {source_url}
Source type: {source_type}
{page_info}
{section_info}Content:
{text}
"""

    USER_TEMPLATE = """Based on the following context documents, answer the question.

{context}

--- Question ---
{question}
"""

    def __init__(self, variant: str = ADOPTED_GROUNDED_VARIANT):
        if variant is not None and variant not in self.PROMPT_VARIANTS:
            raise ValueError(
                f"Unknown prompt variant {variant!r}; "
                f"expected one of {sorted(self.PROMPT_VARIANTS)}"
            )
        self.variant = variant

    def _context_fields(self, res: SearchResult, index: int) -> Dict[str, Any]:
        meta = getattr(res, "metadata", None) or {}
        source_kind = classify_source(res)
        return {
            "index": index,
            "file_name": res.file_name or "Unknown",
            "source_type": self.SOURCE_LABELS.get(source_kind, source_kind),
            "page_info": f"Page: {res.page_number}" if res.page_number else "",
            "section_info": f"Section: {meta['section']}\n" if meta.get("section") else "",
            "title": meta.get("title") or "",
            "authority": meta.get("authority") or meta.get("source") or "",
            "region": meta.get("region") or "",
            "source_url": meta.get("source_url") or "",
            "text": res.text,
        }

    def build_prompt(
        self, question: str, search_results: List[SearchResult], style: str = "detailed"
    ) -> List[Dict[str, str]]:
        variant = self.PROMPT_VARIANTS.get(self.variant) if self.variant else None
        if variant is not None:
            system_prompt = variant["system"] + self.GROUNDING_RULE
            context_template = variant.get("context_template") or (
                self.CONTEXT_TEMPLATE_PROVENANCE
            )
            user_template = variant.get("user_template") or self.USER_TEMPLATE
        else:
            system_prompt = self.SYSTEM_PROMPTS.get(style, self.SYSTEM_PROMPTS["detailed"])
            system_prompt = system_prompt + self.GROUNDING_RULE
            context_template = self.CONTEXT_TEMPLATE
            user_template = self.USER_TEMPLATE

        context_parts = []
        for i, res in enumerate(search_results, start=1):
            context_parts.append(context_template.format(**self._context_fields(res, i)))
        context_str = "\n".join(context_parts)
        user_message = user_template.format(context=context_str, question=question)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
