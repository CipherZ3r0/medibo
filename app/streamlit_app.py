"""
Medibo — Streamlit Application

Clean UI integrating with RAGPipeline and IngestionPipeline domain services.
"""

import html
import os
import re
import sys
from urllib.parse import urlparse

# Ensure root is in PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import config
st.set_page_config(
    page_title="Medibo",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
from domain.models import FeedbackRecord
from ingestion.loaders import CompositeDocumentLoader
from ingestion.medical_index import clear_user_data, dataset_counts
from ingestion.pipeline import IngestionPipeline
from rag.citations import Citation, citations_as_dicts
from rag.evidence import (
    SOURCE_TRUSTED,
    SOURCE_UNKNOWN,
    SOURCE_USER,
    TRUST_LEVEL_USER_ONLY,
    USER_ONLY_BANNER,
    classify_dataset_type,
    classify_source,
)
from rag.pipeline import RAGPipeline
from app.database import PostgresDatabaseRepository

# Initialize services
@st.cache_resource
def get_db_repository():
    repo = PostgresDatabaseRepository()
    try:
        repo.init_db()
    except Exception as e:
        print(f"DB init deferred: {e}")
    return repo

@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline(repository=get_db_repository())

db_repo = get_db_repository()
rag_pipeline = get_rag_pipeline()


# --- Custom CSS ---
# Reading station: cool near-black ink, surgical-jade accent, serif answers
# (Source Serif 4) on an Archivo chrome, steel/amber for the two other
# trust states.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap');

    :root {
        --ink: #0A1216;
        --panel: #0F181E;
        --card: #131E25;
        --line: #1F2E37;
        --text: #E7EEF0;
        --body: #DBE4E7;
        --muted: #93A6AE;
        --jade: #46C895;
        --jade-tint: rgba(70, 200, 149, 0.13);
        --steel: #7FA9E0;
        --steel-tint: rgba(127, 169, 224, 0.13);
        --amber: #E5A54B;
        --amber-tint: rgba(229, 165, 75, 0.12);
        --red: #E5726C;
        --serif: 'Source Serif 4', Georgia, 'Times New Roman', serif;
        --sans: 'Archivo', system-ui, -apple-system, 'Segoe UI', sans-serif;
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp > header {background-color: transparent;}
    [data-testid="stDecoration"] {display: none;}

    /* Base */
    html, body {
        font-family: var(--sans);
        color: var(--text);
        background-color: var(--ink);
    }
    [data-testid="stAppViewContainer"], .stApp {
        background-color: var(--ink);
    }
    p, li { color: var(--body); }
    strong { color: #F2F7F8; }
    a { color: var(--jade); }

    /* Focus visibility */
    button:focus-visible, a:focus-visible, input:focus-visible,
    textarea:focus-visible, [role="tab"]:focus-visible,
    [data-testid="stChatInput"]:focus-within {
        outline: 2px solid var(--jade);
        outline-offset: 2px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--panel);
        border-right: 1px solid var(--line);
        width: 304px !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.1rem;
    }

    /* Main column */
    .main .block-container {
        max-width: 900px !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        padding-top: 1.3rem !important;
        padding-bottom: 7rem !important;
        margin: 0 auto !important;
    }

    /* Brand */
    .med-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        margin-bottom: 6px;
    }
    .med-mark {
        width: 34px;
        height: 34px;
        border: 1px solid var(--line);
        border-radius: 9px;
        background: var(--jade-tint);
        display: flex;
        align-items: center;
        justify-content: center;
        flex: none;
    }
    .med-name {
        font-size: 17.5px;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: var(--text);
        line-height: 1.15;
    }
    .med-tag {
        font-size: 12px;
        color: var(--muted);
        line-height: 1.3;
        margin-top: 1px;
    }
    .med-provider {
        font-size: 12.5px;
        color: var(--muted);
        margin-bottom: 18px;
    }
    .med-provider b {color: var(--text); font-weight: 500;}

    /* Sidebar section labels (sentence case, informational) */
    .side-label {
        font-size: 13px;
        font-weight: 600;
        color: var(--text);
        margin: 4px 0 2px;
    }
    .side-hint {
        font-size: 12.5px;
        color: var(--muted);
        line-height: 1.55;
        margin-bottom: 10px;
    }

    /* Knowledge base ledger */
    .kb {
        border: 1px solid var(--line);
        border-radius: 10px;
        background: var(--card);
        padding: 4px 14px;
        margin-bottom: 12px;
    }
    .kb-row {
        padding: 10px 0;
        border-bottom: 1px solid var(--line);
    }
    .kb-row:last-child {border-bottom: none;}
    .kb-head {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 10px;
    }
    .kb-name {
        font-size: 13px;
        font-weight: 500;
        color: var(--text);
    }
    .kb-count {
        font-size: 15px;
        font-weight: 600;
        color: var(--text);
        font-variant-numeric: tabular-nums;
    }
    .kb-note {
        font-size: 12px;
        color: var(--muted);
        margin-top: 3px;
        line-height: 1.45;
    }
    .kb-status {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .kb-status-val {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-size: 13px;
        color: var(--text);
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--card);
        border: 1px solid var(--line);
        color: var(--text);
        border-radius: 8px;
        font-family: var(--sans);
        font-weight: 500;
        font-size: 13.5px;
        transition: border-color 0.15s ease, background-color 0.15s ease, color 0.15s ease;
    }
    .stButton > button:hover {
        border-color: var(--jade);
        background-color: #16232A;
        color: var(--text);
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stMultiSelect > div > div > div {
        background-color: var(--ink);
        color: var(--text);
        border: 1px solid var(--line);
        border-radius: 8px;
        font-family: var(--sans);
    }
    .stSelectbox [data-baseweb="select"] > div {color: var(--text);}
    .stCheckbox label {color: var(--body); font-size: 13px;}

    /* Upload dropzone */
    [data-testid="stFileUploadDropzone"] {
        background-color: var(--card) !important;
        border: 1px dashed var(--line) !important;
        border-radius: 10px !important;
        padding: 20px 14px !important;
        transition: border-color 0.15s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: var(--jade) !important;
    }
    [data-testid="stFileUploadDropzone"] svg {fill: var(--muted) !important;}
    [data-testid="stFileUploadDropzone"] div {color: var(--muted) !important;}

    /* Expander */
    .stExpander {
        background-color: var(--card) !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
        margin-top: 10px !important;
    }
    .stExpander details {border: none !important; padding: 0 !important;}
    .stExpander details summary {
        padding: 9px 13px !important;
        color: var(--body) !important;
        font-size: 13px !important;
        font-weight: 500;
    }
    .stExpander details summary:hover {color: var(--jade) !important;}
    .stExpander details [data-testid="stExpanderDetails"] {
        padding: 2px 13px 11px 13px !important;
        border-top: 1px solid var(--line) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background-color: transparent;
        border-bottom: 1px solid var(--line);
        padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 34px;
        white-space: nowrap;
        background-color: transparent;
        border: none;
        color: var(--muted);
        font-family: var(--sans);
        font-size: 13px;
        font-weight: 500;
        padding: 0 2px;
    }
    .stTabs [data-baseweb="tab"]:hover {color: var(--text);}
    .stTabs [aria-selected="true"] {
        color: var(--jade) !important;
        border-bottom: 2px solid var(--jade) !important;
    }

    /* Empty state hero */
    .med-hero {
        padding: 18px 0 6px;
        margin-bottom: 14px;
    }
    .med-hero .med-mark {
        width: 46px;
        height: 46px;
        border-radius: 12px;
        margin-bottom: 18px;
    }
    .hero-title {
        font-family: var(--serif);
        font-size: 30px;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: var(--text);
        margin: 0 0 10px;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 15px;
        color: var(--muted);
        max-width: 560px;
        line-height: 1.6;
        margin: 0;
    }
    .suggest-label {
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 10px;
    }

    /* Suggestion buttons (row after the marker) */
    div:has(.suggestion-marker) + div div[data-testid="stHorizontalBlock"] {
        gap: 10px;
    }
    div:has(.suggestion-marker) + div div[data-testid="stHorizontalBlock"] button {
        background-color: var(--card) !important;
        border: 1px solid var(--line) !important;
        color: var(--body) !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        font-family: var(--sans);
        font-weight: 400 !important;
        text-align: left !important;
        font-size: 13.5px !important;
        line-height: 1.45 !important;
        transition: border-color 0.15s ease, background-color 0.15s ease !important;
        height: auto !important;
        min-height: 60px !important;
        display: flex !important;
        align-items: center !important;
        box-shadow: none !important;
    }
    div:has(.suggestion-marker) + div div[data-testid="stHorizontalBlock"] button:hover {
        border-color: var(--jade) !important;
        background-color: #16232A !important;
        color: var(--text) !important;
    }

    /* Feedback buttons (row after the marker) */
    div:has(.feedback-marker) + div div[data-testid="stHorizontalBlock"] {
        gap: 6px;
    }
    div:has(.feedback-marker) + div div[data-testid="stHorizontalBlock"] button {
        background-color: transparent !important;
        border: 1px solid var(--line) !important;
        color: var(--muted) !important;
        border-radius: 8px !important;
        font-size: 12.5px !important;
        min-height: unset !important;
        height: 30px !important;
        width: 30px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: none !important;
    }
    div:has(.feedback-marker) + div div[data-testid="stHorizontalBlock"] button:hover {
        border-color: var(--jade) !important;
        color: var(--jade) !important;
        background-color: var(--jade-tint) !important;
    }

    /* Chat messages: user = quiet right-aligned note, assistant = answer card */
    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 4px 0 10px;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 10px 14px;
        width: fit-content;
        max-width: 88%;
        align-self: flex-end;
        margin-left: auto;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
        margin: 0;
        font-size: 14.5px;
        color: var(--body);
    }
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {
        display: none;
    }

    /* Answer card */
    .ans-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-left: 3px solid var(--line);
        border-radius: 14px;
        padding: 16px 20px 14px;
        font-family: var(--sans);
    }
    .ans-card.state-trusted {border-left-color: var(--jade);}
    .ans-card.state-user {border-left-color: var(--steel);}
    .ans-card.state-fallback {border-left-color: var(--amber);}

    .ans-meta {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        padding-bottom: 12px;
        margin-bottom: 4px;
        border-bottom: 1px solid var(--line);
    }
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: var(--sans);
        font-size: 12px;
        font-weight: 600;
        border-radius: 999px;
        padding: 4px 11px;
        border: 1px solid transparent;
    }
    .chip-ok {background: var(--jade-tint); color: var(--jade); border-color: rgba(70, 200, 149, 0.35);}
    .chip-user {background: var(--steel-tint); color: var(--steel); border-color: rgba(127, 169, 224, 0.35);}
    .chip-warn {background: var(--amber-tint); color: var(--amber); border-color: rgba(229, 165, 75, 0.35);}
    .ans-count {
        margin-left: auto;
        font-size: 12.5px;
        color: var(--muted);
    }

    .ans-note {
        font-size: 13.5px;
        line-height: 1.55;
        border-radius: 8px;
        padding: 9px 12px;
        margin: 12px 0 4px;
    }
    .ans-note.note-warn {background: var(--amber-tint); color: var(--amber); border: 1px solid rgba(229, 165, 75, 0.3);}
    .ans-note.note-user {background: var(--steel-tint); color: var(--steel); border: 1px solid rgba(127, 169, 224, 0.3);}

    .ans-sec {margin-top: 14px;}
    .ans-sec-label {
        font-family: var(--sans);
        font-size: 12px;
        font-weight: 600;
        color: var(--muted);
        margin-bottom: 5px;
    }
    .ans-sec-body {
        font-family: var(--serif);
        font-size: 16px;
        line-height: 1.65;
        color: var(--body);
    }
    .ans-sec-body p {margin: 0 0 8px; color: var(--body);}
    .ans-sec-body p:last-child {margin-bottom: 0;}
    .ans-sec-body ul, .ans-sec-body ol {
        margin: 4px 0 0;
        padding-left: 20px;
    }
    .ans-sec-body li {margin-bottom: 5px; color: var(--body);}
    .ans-sec-body strong {color: #F2F7F8; font-weight: 600;}
    .ans-sec-body a {color: var(--jade);}
    .ans-sec-body code {
        font-family: var(--sans);
        font-size: 0.88em;
        background: rgba(147, 166, 174, 0.14);
        padding: 1px 5px;
        border-radius: 4px;
    }

    /* Source rows (journal reference list) */
    .ans-sources {
        margin-top: 16px;
        padding-top: 12px;
        border-top: 1px solid var(--line);
    }
    .ref {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 8px 9px;
        margin: 0 -9px;
        border-radius: 8px;
        text-decoration: none;
        color: inherit;
        transition: background-color 0.15s ease;
    }
    .ref:hover {background: rgba(147, 166, 174, 0.08);}
    .ref-idx {
        flex: none;
        width: 20px;
        height: 20px;
        border-radius: 6px;
        background: var(--jade-tint);
        color: var(--jade);
        font-family: var(--sans);
        font-size: 11.5px;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .ref-idx.idx-user {background: var(--steel-tint); color: var(--steel);}
    .ref-txt {min-width: 0; flex: 1;}
    .ref-name {
        display: block;
        font-family: var(--sans);
        font-size: 13.5px;
        font-weight: 500;
        color: var(--text);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    a.ref:hover .ref-name {color: var(--jade);}
    .ref-meta {
        display: block;
        font-size: 12px;
        color: var(--muted);
        margin-top: 1px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .ref-tag {
        flex: none;
        font-size: 11px;
        font-weight: 600;
        color: var(--jade);
        background: var(--jade-tint);
        border-radius: 999px;
        padding: 3px 9px;
    }
    .ref-tag.tag-user {color: var(--steel); background: var(--steel-tint);}
    .ref-url {
        flex: none;
        font-size: 12px;
        color: var(--muted);
        max-width: 150px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    a.ref:hover .ref-url {color: var(--jade);}

    .ans-foot {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid var(--line);
        font-size: 12px;
        color: var(--muted);
    }

    .ans-error {
        background: rgba(229, 114, 108, 0.1);
        border: 1px solid rgba(229, 114, 108, 0.35);
        border-radius: 10px;
        color: var(--red);
        padding: 11px 14px;
        font-size: 13.5px;
    }

    /* Retrieved passages expander */
    .source-card {
        background-color: var(--card);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 11px 13px;
        margin-top: 8px;
        font-size: 0.85rem;
        color: var(--muted);
    }
    .source-card .file-name {
        color: var(--text);
        font-weight: 500;
        display: block;
        margin-bottom: 3px;
        font-size: 13px;
    }
    .src-group {
        font-size: 12.5px;
        font-weight: 600;
        color: var(--muted);
        margin: 12px 0 2px;
    }
    .src-group:first-child {margin-top: 4px;}
    .src-meta {font-size: 12px; color: var(--muted);}
    .src-url {color: var(--jade); font-size: 12px; word-break: break-all;}

    /* Chat composer — the primary action */
    [data-testid="stChatInput"] {
        background-color: var(--panel) !important;
        border: 1px solid var(--line) !important;
        border-radius: 14px !important;
        padding: 7px 8px !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--jade) !important;
        box-shadow: 0 0 0 3px rgba(70, 200, 149, 0.15) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text) !important;
        font-family: var(--sans) !important;
        font-size: 15px !important;
        line-height: 1.5 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {color: var(--muted);}
    [data-testid="stChatInput"] button {
        background-color: var(--jade) !important;
        color: #06130D !important;
        border: none !important;
        border-radius: 10px !important;
    }
    [data-testid="stChatInput"] button:hover {filter: brightness(1.07);}
    [data-testid="stChatInput"] button svg {fill: #06130D !important;}

    hr {border-top: 1px solid var(--line);}

    /* Settings dialog — same ink surface as the app */
    div[data-testid="stDialog"] {
        background-color: rgba(6, 10, 13, 0.66) !important;
    }
    div[role="dialog"] {
        background-color: var(--panel) !important;
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        box-shadow: 0 24px 64px rgba(0, 0, 0, 0.5) !important;
        color: var(--text);
        font-family: var(--sans);
    }
    div[role="dialog"] label {
        color: var(--body) !important;
        font-family: var(--sans) !important;
    }
    div[role="dialog"] button[aria-label="Close"] {
        color: var(--muted) !important;
    }
    div[role="dialog"] button[aria-label="Close"]:hover {
        color: var(--text) !important;
        background-color: transparent !important;
    }
    div[role="dialog"] [data-baseweb="select"],
    div[role="dialog"] input {
        color: var(--text) !important;
    }

    @media (max-width: 980px) {
        .main .block-container {
            max-width: 100% !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
        }
    }
    @media (prefers-reduced-motion: reduce) {
        * {transition: none !important; animation: none !important;}
    }
</style>
""", unsafe_allow_html=True)


# --- State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "clear_my_data_pending" not in st.session_state:
    st.session_state.clear_my_data_pending = False


# --- Generation settings ---
# The controls live in a dialog (top-right of the workspace) but their values
# live in session state, so they survive the dialog closing and are read fresh
# on every full-script run: the sidebar provider line and each query both use
# `current_generation_settings()`.
_PROVIDER_OPTIONS = ["Groq (Cloud)", "Ollama (Local)"]
_GROQ_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "Custom..."]
_OLLAMA_MODELS = ["llama3", "mistral", "gemma2", "phi3", "Custom..."]
_SEARCH_OPTIONS = ["hybrid", "vector", "text"]
_STYLE_OPTIONS = ["detailed", "concise", "structured"]


def _models_for(provider_label: str) -> list:
    return _GROQ_MODELS if "Groq" in provider_label else _OLLAMA_MODELS


def _init_settings_state() -> None:
    """Seed generation-setting defaults, mirroring the old expander defaults.

    Values are (re)confirmed on every script run: Streamlit only pushes a
    widget's value to the frontend when its key was written during the current
    run. Without re-writing here, a value seeded on an earlier run would render
    as the widget's default (index-0 option / unchecked) when the dialog first
    opens. User changes are preserved — the write is a read-back touch.
    """
    ss = st.session_state
    if "gen_provider" not in ss:
        ss.gen_provider = (
            _PROVIDER_OPTIONS[0]
            if config.LLM_PROVIDER.lower() == "groq"
            else _PROVIDER_OPTIONS[1]
        )
    is_groq = "Groq" in ss.gen_provider
    cfg_model = config.LLM_MODEL if is_groq else config.OLLAMA_MODEL
    if "gen_model" not in ss:
        ss.gen_model = (
            cfg_model if cfg_model in _models_for(ss.gen_provider) else "Custom..."
        )
    if "gen_model_custom" not in ss:
        ss.gen_model_custom = (
            "" if cfg_model in _models_for(ss.gen_provider) else cfg_model
        )
    if "gen_search" not in ss:
        ss.gen_search = _SEARCH_OPTIONS[0]
    if "gen_rerank" not in ss:
        ss.gen_rerank = True
    if "gen_rewrite" not in ss:
        ss.gen_rewrite = True
    if "gen_style" not in ss:
        ss.gen_style = _STYLE_OPTIONS[0]
    for key in (
        "gen_provider",
        "gen_model",
        "gen_model_custom",
        "gen_search",
        "gen_rerank",
        "gen_rewrite",
        "gen_style",
    ):
        ss[key] = ss[key]


def current_generation_settings() -> tuple:
    """Read-only view of the live settings for this script run."""
    ss = st.session_state
    is_groq = "Groq" in ss.gen_provider
    models = _models_for(ss.gen_provider)
    option = ss.gen_model
    if option not in models:
        cfg_model = config.LLM_MODEL if is_groq else config.OLLAMA_MODEL
        option = cfg_model if cfg_model in models else "Custom..."
    if option == "Custom...":
        ui_model = str(ss.gen_model_custom or "").strip()
    else:
        ui_model = option
    return (
        "groq" if is_groq else "ollama",
        ui_model,
        ss.gen_search,
        ss.gen_rerank,
        ss.gen_rewrite,
        ss.gen_style,
    )


@st.dialog("Generation settings", width="large")
def generation_settings_dialog() -> None:
    st.selectbox("LLM Provider", _PROVIDER_OPTIONS, key="gen_provider")
    models = _models_for(st.session_state.gen_provider)
    if st.session_state.gen_model not in models:
        # Provider just changed inside this dialog run — keep the model valid.
        is_groq = "Groq" in st.session_state.gen_provider
        cfg_model = config.LLM_MODEL if is_groq else config.OLLAMA_MODEL
        st.session_state.gen_model = (
            cfg_model if cfg_model in models else "Custom..."
        )
    st.selectbox("Model", models, key="gen_model")
    if st.session_state.gen_model == "Custom...":
        st.text_input("Enter model name", key="gen_model_custom")
    st.selectbox("Search Strategy", _SEARCH_OPTIONS, key="gen_search")
    c_gen1, c_gen2 = st.columns(2)
    with c_gen1:
        st.checkbox("Rerank", key="gen_rerank")
    with c_gen2:
        st.checkbox("Rewrite", key="gen_rewrite")
    st.selectbox("Output Style", _STYLE_OPTIONS, key="gen_style")
    st.markdown(
        "<div class='side-hint' style='margin: 12px 0 0;'>"
        "Applied from your next message.</div>",
        unsafe_allow_html=True,
    )


_init_settings_state()
(
    selected_provider_key,
    ui_model,
    retrieval_method,
    use_reranking,
    use_query_rewrite,
    prompt_style,
) = current_generation_settings()


# --- Answer rendering ---------------------------------------------------
# A small, dependency-free markdown renderer for the subset the answer
# prompt produces (paragraphs, bullet/numbered lists, bold, italics,
# inline code, links). Everything is HTML-escaped first.

_INLINE_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_INLINE_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_INLINE_EM_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_SECTION_RE = re.compile(
    r"(?m)^[ \t]*(?:\d+\.[ \t]+)?\*\*(Answer|Details|Seek care|Sources)\*\*"
    r"[ \t]*(?:[—–:-][ \t]*)?"
)


def _inline_md(text: str) -> str:
    s = html.escape(text, quote=True)
    s = _INLINE_LINK_RE.sub(r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
    s = _INLINE_BOLD_RE.sub(r"<strong>\1</strong>", s)
    s = _INLINE_EM_RE.sub(r"<em>\1</em>", s)
    s = _INLINE_CODE_RE.sub(r"<code>\1</code>", s)
    return s


def _markdown_subset(text: str) -> str:
    """Render the answer's markdown subset to HTML (escaped first)."""
    blocks = []
    para: list[str] = []
    list_tag = None
    items: list[str] = []

    def flush_para():
        if para:
            blocks.append("<p>" + _inline_md(" ".join(para)) + "</p>")
            para.clear()

    def flush_list():
        nonlocal list_tag
        if list_tag:
            blocks.append(f"<{list_tag}>" + "".join(items) + f"</{list_tag}>")
            items.clear()
            list_tag = None

    for raw in (text or "").split("\n"):
        line = raw.rstrip()
        stripped = line.strip()
        m_ul = re.match(r"[-*][ \t]+(.*)", stripped)
        m_ol = re.match(r"\d+\.[ \t]+(.*)", stripped)
        if m_ul or m_ol:
            flush_para()
            want = "ul" if m_ul else "ol"
            if list_tag and list_tag != want:
                flush_list()
            list_tag = want
            items.append("<li>" + _inline_md((m_ul or m_ol).group(1)) + "</li>")
        elif not stripped:
            flush_para()
            flush_list()
        else:
            flush_list()
            para.append(stripped)
    flush_para()
    flush_list()
    return "\n".join(blocks)


def _split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split an answer into (intro before first marker, [(section, body)])."""
    matches = list(_SECTION_RE.finditer(text or ""))
    if not matches:
        return (text or "").strip(), []
    intro = (text[: matches[0].start()]).strip()
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if body:
            sections.append((m.group(1), body))
    return intro, sections


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def _ref_row_html(c: Citation) -> str:
    idx = html.escape(str(c.index))
    name = html.escape(c.name)
    meta_bits = [str(b) for b in (c.authority, c.region, c.locator) if b]
    meta = html.escape(" · ".join(meta_bits))
    is_user = c.trust == SOURCE_USER
    tag_cls = " ref-tag tag-user" if is_user else "ref-tag"
    tag_txt = "Your upload" if is_user else "Medibo source"
    idx_cls = "ref-idx idx-user" if is_user else "ref-idx"
    url = str(c.source_url or "").strip()
    linked = url.startswith(("http://", "https://"))
    domain = html.escape(_domain(url)) if linked else ""
    href = html.escape(url, quote=True) if linked else ""
    inner = (
        f'<span class="{idx_cls}">{idx}</span>'
        f'<span class="ref-txt"><span class="ref-name">{name}</span>'
        + (f'<span class="ref-meta">{meta}</span>' if meta else "")
        + "</span>"
        f'<span class="{tag_cls}">{tag_txt}</span>'
        + (f'<span class="ref-url">{domain}</span>' if domain else "")
    )
    if linked:
        return (
            f'<a class="ref" href="{href}" target="_blank" '
            f'rel="noopener">{inner}</a>'
        )
    return f'<div class="ref">{inner}</div>'


def answer_card_html(
    answer: str,
    citations: list,
    state: str,
    note: str | None = None,
    foot: str | None = None,
) -> str:
    """Build the structured medical answer card as one HTML block."""
    notes = {
        "trusted": ("chip-ok", "✓ Trusted Medibo source"),
        "user": ("chip-user", "Your uploads only"),
        "fallback": ("chip-warn", "⚠ General knowledge"),
    }
    chip_cls, chip_txt = notes.get(state, notes["fallback"])
    n = len(citations or [])
    count = f'<span class="ans-count">{n} source{"s" if n != 1 else ""}</span>' if n else ""

    intro, sections = _split_sections(answer)
    parts = [
        f'<div class="ans-card state-{html.escape(state)}">',
        f'<div class="ans-meta"><span class="chip {chip_cls}">{chip_txt}</span>{count}</div>',
    ]
    if intro:
        parts.append(f"<p>{_inline_md(intro)}</p>")
    if note:
        note_cls = "note-user" if state == "user" else "note-warn"
        parts.append(f'<div class="ans-note {note_cls}">{_inline_md(note)}</div>')
    for label, body in sections:
        parts.append(
            f'<div class="ans-sec"><div class="ans-sec-label">{html.escape(label)}</div>'
            f'<div class="ans-sec-body">{_markdown_subset(body)}</div></div>'
        )
    if not sections and not intro and answer:
        parts.append(f'<div class="ans-sec-body">{_markdown_subset(answer)}</div>')
    if citations:
        rows = "".join(_ref_row_html(_as_citation(c)) for c in citations)
        parts.append(
            '<div class="ans-sources"><div class="ans-sec-label">Sources</div>'
            f"{rows}</div>"
        )
    if foot:
        parts.append(f'<div class="ans-foot">{html.escape(foot)}</div>')
    parts.append("</div>")
    return "".join(parts)


def render_answer_card(
    answer: str,
    citations,
    state: str,
    note: str | None = None,
    response_time_ms=None,
    total_tokens=None,
) -> None:
    foot = None
    if response_time_ms and total_tokens:
        foot = f"Answered in {response_time_ms} ms · {total_tokens} tokens"
    st.markdown(
        answer_card_html(answer, citations or [], state, note=note, foot=foot),
        unsafe_allow_html=True,
    )


def _answer_state(evidence_status: str, trust_level: str) -> str:
    if evidence_status != "sufficient":
        return "fallback"
    if trust_level == TRUST_LEVEL_USER_ONLY:
        return "user"
    return "trusted"


def render_source_card(src: dict) -> str:
    """HTML card for one retrieved source, including medical metadata."""
    meta = src.get("metadata") or {}
    label = html.escape(str(src.get("file_name") or "Unknown"))
    parts = []
    if src.get("page_number"):
        parts.append(f"Page {src['page_number']}")
    for key in ("section", "source", "title"):
        if meta.get(key):
            parts.append(str(meta[key]))
    prefix = " · ".join(html.escape(p) for p in parts)
    body = html.escape(str(src.get("text") or ""))[:200]
    card = f'<div class="source-card"><span class="file-name">{label}</span>'
    if prefix:
        card += f'<div class="src-meta">{prefix}</div>'
    if body:
        card += f'<div style="margin-top: 4px;">{body}...</div>'
    url = meta.get("source_url")
    if url:
        safe_url = html.escape(str(url), quote=True)
        card += (
            f'<div style="margin-top: 6px;"><a class="src-url" href="{safe_url}" '
            f'target="_blank" rel="noopener">{html.escape(_domain(str(url)) or str(url))}</a></div>'
        )
    card += "</div>"
    return card


def doc_to_source(doc) -> dict:
    """UI source dict for one retrieved document, tagged with its trust class.

    Phase 8: the class comes from ``dataset_type`` only — ``default`` is
    trusted, ``user`` is user-provided, anything else is unknown/untrusted.
    """
    trust = classify_source(doc)
    return {
        "file_name": doc.file_name,
        "page_number": doc.page_number,
        "text": doc.text,
        "metadata": doc.metadata,
        "trust": trust,
        "trusted": trust == SOURCE_TRUSTED,
    }


def _source_trust(src: dict) -> str:
    """Trust class of a stored source dict (handles legacy messages too)."""
    trust = src.get("trust")
    if trust in (SOURCE_TRUSTED, SOURCE_USER, SOURCE_UNKNOWN):
        return trust
    meta = src.get("metadata") or {}
    return classify_dataset_type(meta.get("dataset_type"))


def split_sources(sources: list) -> dict:
    """Bucket source dicts into trusted / user-provided / unknown.

    Unknown or missing ``dataset_type`` never lands in the trusted bucket.
    """
    groups = {"trusted": [], "user": [], "unknown": []}
    for src in sources or []:
        groups[_source_trust(src)].append(src)
    return groups


def render_sources_block(groups: dict) -> None:
    """Draw the passages expander, keeping curated and user uploads visually apart."""
    trusted = groups.get("trusted") or []
    user = groups.get("user") or []
    unknown = groups.get("unknown") or []
    if not trusted and not user and not unknown:
        return
    headers = (
        (trusted, "From the Medibo knowledge base"),
        (user, "From your uploads"),
        (unknown, "Unverified origin — never cited"),
    )
    with st.expander("Retrieved passages", expanded=False):
        for i, (items, title) in enumerate(headers):
            if not items:
                continue
            margin = "" if i == 0 else "margin: 14px 0 6px; "
            st.markdown(
                f"<div class='src-group' style='{margin}'>{title}</div>",
                unsafe_allow_html=True,
            )
            for src in items:
                st.markdown(render_source_card(src), unsafe_allow_html=True)


def _as_citation(item) -> Citation:
    return item if isinstance(item, Citation) else Citation(**item)


def kb_panel_html(default_count: int, user_count: int,
                  status_text: str, status_color: str) -> str:
    return f"""
        <div class="kb">
            <div class="kb-row">
                <div class="kb-head">
                    <span class="kb-name">Default medical knowledge</span>
                    <span class="kb-count">{default_count:,}</span>
                </div>
                <div class="kb-note">Permanent and read-only — never changed here</div>
            </div>
            <div class="kb-row">
                <div class="kb-head">
                    <span class="kb-name">My data</span>
                    <span class="kb-count">{user_count:,}</span>
                </div>
                <div class="kb-note">Your uploads — only you can clear them</div>
            </div>
            <div class="kb-row kb-status">
                <span class="kb-name">Status</span>
                <span class="kb-status-val">
                    <span class="status-dot" style="background-color: {status_color};"></span>
                    {status_text}
                </span>
            </div>
        </div>
    """


MARK_SVG = (
    '<svg width="18" height="18" viewBox="0 0 20 20" aria-hidden="true">'
    '<rect x="7.6" y="2.6" width="4.8" height="14.8" rx="2" fill="#46C895"/>'
    '<rect x="2.6" y="7.6" width="14.8" height="4.8" rx="2" fill="#46C895"/>'
    "</svg>"
)
MARK_SVG_LG = MARK_SVG.replace('width="18" height="18"', 'width="26" height="26"')


# --- Sidebar ---
with st.sidebar:
    st.markdown(f"""
        <div class="med-brand">
            <div class="med-mark">{MARK_SVG}</div>
            <div>
                <div class="med-name">Medibo</div>
                <div class="med-tag">Medical knowledge retrieval</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    header_placeholder = st.empty()

    if st.button(
        "Settings",
        key="btn_open_settings",
        icon=":material/settings:",
        use_container_width=True,
    ):
        generation_settings_dialog()

    st.markdown("<div class='side-label'>Knowledge base</div>", unsafe_allow_html=True)

    # --- Knowledge Base Stats: default corpus and my data are counted apart ---
    try:
        from ingestion.indexers import ElasticsearchVectorStore

        counts = dataset_counts()
        es_store = ElasticsearchVectorStore()
        if es_store.es and es_store.es.ping():
            status_text = "Online"
            status_color = "#46C895"
        else:
            status_text = "Offline"
            status_color = "#E5726C"
        st.markdown(
            kb_panel_html(counts["default"], counts["user"], status_text, status_color),
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            kb_panel_html(0, 0, "Offline", "#E5726C"), unsafe_allow_html=True
        )

    # --- Clear my data: deletes the user dataset only ---
    # Separate from the "Rebuild index" checkbox above: rebuilding re-ingests
    # your uploads, clearing removes them. Neither touches medibo_default.
    if st.button("Clear my data", key="btn_clear_my_data", use_container_width=True):
        st.session_state.clear_my_data_pending = True
        st.rerun()

    if st.session_state.get("clear_my_data_pending"):
        st.warning(
            "This deletes only your uploaded data. "
            "Default medical knowledge is read-only and stays in place."
        )
        c_clear_yes, c_clear_no = st.columns(2)
        with c_clear_yes:
            if st.button("Confirm", key="btn_clear_my_data_confirm", use_container_width=True):
                try:
                    result = clear_user_data()
                    st.toast(
                        "My data cleared"
                        if result.get("deleted")
                        else "No uploaded data to clear"
                    )
                except Exception as e:
                    st.error(f"Could not clear my data: {e}")
                st.session_state.clear_my_data_pending = False
                st.rerun()
        with c_clear_no:
            if st.button("Cancel", key="btn_clear_my_data_cancel", use_container_width=True):
                st.session_state.clear_my_data_pending = False
                st.rerun()

    st.markdown("<div class='side-label' style='margin-top: 14px;'>Add to My data</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='side-hint'>Files and folders you add land in My data. "
        "The default medical knowledge base is never rebuilt or deleted here.</div>",
        unsafe_allow_html=True,
    )

    tab_files, tab_folder = st.tabs(["Upload", "Folder"])

    with tab_files:
        supported_types = CompositeDocumentLoader.get_supported_extensions_without_dot()
        uploaded_files = st.file_uploader(
            "Drop documents here",
            type=supported_types,
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        recreate_idx_files = st.checkbox("Rebuild index", value=False, key="recreate_files")
        if st.button("Ingest Files", use_container_width=True, key="btn_ingest_files"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    try:
                        pipeline = IngestionPipeline()
                        result = pipeline.run_uploaded_files(
                            uploaded_files=uploaded_files,
                            recreate_index=recreate_idx_files
                        )
                        st.success(f"Indexed {result.get('indexed_count', 0)} chunks into My data")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please select a file.")

    with tab_folder:
        raw_folder_path = st.text_input(
            "Local folder path",
            placeholder=r"C:\documents",
            label_visibility="collapsed"
        )
        recreate_idx_folder = st.checkbox("Rebuild index", value=False, key="recreate_folder")
        if st.button("Ingest Folder", use_container_width=True, key="btn_ingest_folder"):
            clean_path = raw_folder_path.strip(" '\"") if raw_folder_path else ""
            if clean_path:
                clean_path = os.path.normpath(clean_path)
                if os.path.isdir(clean_path):
                    with st.spinner("Processing..."):
                        try:
                            pipeline = IngestionPipeline()
                            result = pipeline.run(folder_path=clean_path, recreate_index=recreate_idx_folder)
                            st.success(f"Indexed {result.get('indexed_count', 0)} chunks into My data")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Invalid folder path.")

    # Provider line reflects the settings read at the top of this run
    provider_label = "Ollama (local)" if selected_provider_key == "ollama" else "Groq (cloud)"
    header_placeholder.markdown(
        f"<div class='med-provider'>Provider: <b>{provider_label}</b></div>",
        unsafe_allow_html=True,
    )


# --- Generation settings ------------------------------------------------

# --- Render Empty State or Conversation History ---
if not st.session_state.messages:
    st.markdown(f"""
        <div class="med-hero">
            <div class="med-mark">{MARK_SVG_LG}</div>
            <h2 class="hero-title">Ask a medical question.</h2>
            <p class="hero-sub">Medibo answers from the medical knowledge in this
            workspace and anything you upload here — and shows the source behind
            every answer.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<p class='suggest-label'>Try one of these</p>", unsafe_allow_html=True
    )
    st.markdown("<div class='suggestion-marker'></div>", unsafe_allow_html=True)

    suggestions = [
        "What are the warning signs of dengue?",
        "How should a snakebite be treated in the first hour?",
        "When does a fever need urgent medical care?",
    ]

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_prompt = suggestion
                st.rerun()
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                if msg.get("evidence_status") in ("sufficient", "insufficient"):
                    state = _answer_state(
                        msg.get("evidence_status"), msg.get("trust_level") or "none"
                    )
                    note = None
                    if state == "user":
                        note = USER_ONLY_BANNER
                    elif state == "fallback":
                        note = "No sources cited — general-knowledge answer."
                    render_answer_card(
                        msg.get("content") or "",
                        msg.get("citations") or [],
                        state,
                        note=note,
                        response_time_ms=msg.get("response_time_ms"),
                        total_tokens=msg.get("total_tokens"),
                    )
                    if state != "fallback" and msg.get("sources"):
                        render_sources_block(split_sources(msg["sources"]))
                elif (msg.get("content") or "").startswith("Error:"):
                    st.markdown(
                        f"<div class='ans-error'>{html.escape(msg.get('content') or '')}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(msg.get("content") or "")

                if msg.get("conversation_id"):
                    conv_id = msg["conversation_id"]
                    st.markdown("<div class='feedback-marker'></div>", unsafe_allow_html=True)
                    c1, c2, _ = st.columns([1, 1, 10])
                    with c1:
                        if st.button("👍", key=f"up_hist_{conv_id}"):
                            db_repo.save_feedback(FeedbackRecord(conversation_id=conv_id, sentiment="positive"))
                            st.toast("Feedback recorded: 👍")
                    with c2:
                        if st.button("👎", key=f"down_hist_{conv_id}"):
                            db_repo.save_feedback(FeedbackRecord(conversation_id=conv_id, sentiment="negative"))
                            st.toast("Feedback recorded: 👎")
            else:
                st.markdown(msg.get("content") or "")


# --- Chat Input & Execution ---
active_prompt = None

if st.session_state.pending_prompt:
    active_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

chat_input_val = st.chat_input("Ask a medical question…")
if chat_input_val:
    active_prompt = chat_input_val

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating..."):
            try:
                response = rag_pipeline.query(
                    user_query=active_prompt,
                    retrieval_method=retrieval_method,
                    use_reranking=use_reranking,
                    use_query_rewriting=use_query_rewrite,
                    prompt_style=prompt_style,
                    llm_provider_name=selected_provider_key,
                    llm_model=ui_model,
                )

                evidence = getattr(response, "evidence", None)
                grounded = bool(evidence is not None and evidence.sufficient)
                groups = {"trusted": [], "user": [], "unknown": []}

                if grounded:
                    groups = {
                        "trusted": [doc_to_source(doc) for doc in evidence.trusted_docs],
                        "user": [doc_to_source(doc) for doc in evidence.user_docs],
                        "unknown": [
                            doc_to_source(doc)
                            for doc in getattr(evidence, "unknown_docs", ())
                        ],
                    }
                    state = _answer_state(
                        "sufficient", evidence.trust_level or "none"
                    )
                    note = USER_ONLY_BANNER if state == "user" else None
                    # Citations come from the supporting chunks only — unknown
                    # provenance is shown separately and never cited.
                    render_answer_card(
                        response.answer,
                        response.citations,
                        state,
                        note=note,
                        response_time_ms=response.response_time_ms,
                        total_tokens=response.total_tokens,
                    )
                    render_sources_block(groups)
                else:
                    # The answer already carries the explicit warning line —
                    # never attach sources or citations to an ungrounded answer.
                    render_answer_card(
                        response.answer,
                        [],
                        "fallback",
                        note="No sources cited — general-knowledge answer.",
                        response_time_ms=response.response_time_ms,
                        total_tokens=response.total_tokens,
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": groups["trusted"] + groups["user"] + groups["unknown"],
                    "citations": citations_as_dicts(response.citations),
                    "evidence_status": "sufficient" if grounded else "insufficient",
                    "trust_level": evidence.trust_level if evidence else "none",
                    "conversation_id": response.conversation_id,
                    "response_time_ms": response.response_time_ms,
                    "total_tokens": response.total_tokens
                })

            except Exception as e:
                err_text = f"Error: {str(e)}"
                st.error(err_text)
                st.session_state.messages.append({"role": "assistant", "content": err_text})

    st.rerun()
