"""
potbot — Streamlit Application

Clean UI integrating with RAGPipeline and IngestionPipeline domain services.
"""

import os
import sys

# Ensure root is in PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import config
st.set_page_config(
    page_title="potbot",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)
from domain.models import FeedbackRecord
from ingestion.loaders import CompositeDocumentLoader
from ingestion.medical_index import clear_user_data, dataset_counts
from ingestion.pipeline import IngestionPipeline
from rag.citations import Citation, citations_as_dicts, citations_markdown
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
# Theme: Background (#0F1117), Sidebar (#14161D), Cards (#181B23), Borders (#262A35), 
# White text (#F5F5F5), Muted (#A1A1AA), Orange Accent (#F97316)
st.markdown("""
<style>
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp > header {background-color: transparent;}
    
    /* Global Theme & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #F5F5F5;
        background-color: #0F1117;
    }
    
    /* Backgrounds & Sidebar */
    .stApp {
        background-color: #0F1117;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #14161D;
        border-right: 1px solid #262A35;
        width: 300px !important;
    }
    
    /* Constrain main block width & center it */
    .main .block-container {
        max-width: 840px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 0.5rem !important;
        padding-bottom: 7rem !important;
        margin: 0 auto !important;
    }

    /* Cards */
    .sidebar-card {
        background-color: #181B23;
        border: 1px solid #262A35;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    /* Compact Header */
    .app-header {
        display: flex;
        align-items: center;
        gap: 12px;
        border-bottom: 1px solid #262A35;
        padding-bottom: 12px;
        margin-bottom: 16px;
    }
    .app-header-logo {
        background: #F97316;
        color: #181B23;
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 18px;
        font-weight: 600;
    }
    .app-header-title {
        font-size: 20px;
        font-weight: 600;
        margin: 0;
        color: #F5F5F5;
        line-height: 1.2;
    }
    .app-header-subtitle {
        font-size: 13px;
        color: #A1A1AA;
        margin: 0;
        line-height: 1.2;
    }
    
    /* Empty State style */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 20px 20px 32px;
        margin-top: 0;
        margin-bottom: 20px;
    }
    .empty-state-icon {
        font-size: 32px;
        color: #F97316;
        margin-bottom: 20px;
        background: rgba(249, 115, 22, 0.1);
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .empty-state-title {
        font-size: 22px;
        font-weight: 600;
        color: #F5F5F5;
        margin-bottom: 10px;
    }
    .empty-state-subtitle {
        font-size: 14px;
        color: #A1A1AA;
        max-width: 480px;
        line-height: 1.6;
        margin-bottom: 24px;
    }
    
    /* Target horizontal block after the suggestion marker */
    div:has(.suggestion-marker) + div div[data-testid="stHorizontalBlock"] button {
        background-color: #181B23 !important;
        border: 1px solid #262A35 !important;
        color: #F5F5F5 !important;
        border-radius: 8px !important;
        padding: 12px 14px !important;
        font-weight: 400 !important;
        text-align: left !important;
        font-size: 13px !important;
        transition: all 0.2s ease !important;
        height: auto !important;
        min-height: 64px !important;
        display: flex !important;
        align-items: center !important;
        box-shadow: none !important;
    }
    div:has(.suggestion-marker) + div div[data-testid="stHorizontalBlock"] button:hover {
        border-color: #F97316 !important;
        background-color: #1c202a !important;
        color: #F5F5F5 !important;
    }
    
    /* Target horizontal block after the feedback marker */
    div:has(.feedback-marker) + div div[data-testid="stHorizontalBlock"] button {
        background-color: #181B23 !important;
        border: 1px solid #262A35 !important;
        color: #A1A1AA !important;
        border-radius: 6px !important;
        padding: 4px 8px !important;
        font-size: 12px !important;
        min-height: unset !important;
        height: 32px !important;
        width: 32px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: none !important;
    }
    div:has(.feedback-marker) + div div[data-testid="stHorizontalBlock"] button:hover {
        border-color: #F97316 !important;
        color: #F97316 !important;
        background-color: rgba(249, 115, 22, 0.05) !important;
    }
    
    /* Source Cards inside Chat */
    .source-card {
        background-color: #181B23;
        border: 1px solid #262A35;
        border-radius: 6px;
        padding: 12px;
        margin-top: 8px;
        font-size: 0.85rem;
        color: #A1A1AA;
    }
    .source-card .file-name {
        color: #F97316;
        font-weight: 500;
        display: block;
        margin-bottom: 4px;
    }
    
    /* Premium Upload Area */
    [data-testid="stFileUploadDropzone"] {
        background-color: #181B23 !important;
        border: 1px dashed #262A35 !important;
        border-radius: 8px !important;
        padding: 24px 16px !important;
        transition: border-color 0.2s, background-color 0.2s;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #F97316 !important;
        background-color: #1c202a !important;
    }
    [data-testid="stFileUploadDropzone"] svg {
        fill: #F97316 !important;
    }
    [data-testid="stFileUploadDropzone"] div {
        color: #A1A1AA !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #181B23;
        border: 1px solid #262A35;
        color: #F5F5F5;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #F97316;
        background-color: #1c202a;
        color: #F97316;
    }
    
    /* Inputs */
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > div {
        background-color: #0F1117;
        color: #F5F5F5;
        border: 1px solid #262A35;
        border-radius: 6px;
    }
    
    /* Chat Input styling */
    [data-testid="stChatInput"] {
        background-color: #181B23 !important;
        border: 1px solid #262A35 !important;
        border-radius: 8px !important;
        padding: 8px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #F97316 !important;
        box-shadow: 0 0 0 1px #F97316 !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #F5F5F5 !important;
        font-size: 15px !important;
        line-height: 1.5 !important;
    }

    /* Chat Messages styling with smooth fade-in */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stChatMessage {
        background-color: transparent;
        padding: 1rem 0;
        animation: fadeIn 0.3s ease-out forwards;
    }
    [data-testid="chatAvatarIcon-user"] {
        background-color: #262A35;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #F97316;
    }
    
    /* Sidebar stats layout */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        background-color: #181B23;
        border: 1px solid #262A35;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 20px;
    }
    .stat-item {
        text-align: center;
    }
    .stat-val {
        font-size: 15px;
        font-weight: 600;
        color: #F5F5F5;
        line-height: 1.2;
    }
    .stat-lbl {
        font-size: 10px;
        color: #A1A1AA;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid #262A35;
        padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 36px;
        white-space: nowrap;
        background-color: transparent;
        border: none;
        color: #A1A1AA;
        font-size: 13px;
        font-weight: 500;
        padding: 0 4px;
        transition: color 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F5F5F5;
    }
    .stTabs [aria-selected="true"] {
        color: #F97316 !important;
        border-bottom: 2px solid #F97316 !important;
    }
    
    /* Expander overrides */
    .stExpander {
        background-color: #181B23 !important;
        border: 1px solid #262A35 !important;
        border-radius: 6px !important;
        margin-top: 8px !important;
    }
    .stExpander details {
        border: none !important;
        padding: 0 !important;
    }
    .stExpander details summary {
        padding: 8px 12px !important;
        color: #A1A1AA !important;
        font-size: 13px !important;
    }
    .stExpander details summary:hover {
        color: #F5F5F5 !important;
    }
    .stExpander details [data-testid="stExpanderDetails"] {
        padding: 0 12px 12px 12px !important;
        border-top: 1px solid #262A35 !important;
        background-color: #14161D !important;
    }
    
    hr {
        border-top: 1px solid #262A35;
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


def render_source_card(src: dict) -> str:
    """HTML card for one retrieved source, including medical metadata."""
    import html as _html

    meta = src.get("metadata") or {}
    label = _html.escape(str(src.get("file_name") or "Unknown"))
    parts = []
    if src.get("page_number"):
        parts.append(f"Page {src['page_number']}")
    for key in ("section", "source", "title"):
        if meta.get(key):
            parts.append(str(meta[key]))
    prefix = " • ".join(_html.escape(p) for p in parts)
    body = _html.escape(str(src.get("text") or ""))[:200]
    card = f'<div class="source-card"><span class="file-name">{label}</span>'
    if prefix:
        card += f'<div style="font-size: 12px; color: #A1A1AA;">{prefix}</div>'
    card += f'<div style="margin-top: 4px;">{body}...</div>'
    url = meta.get("source_url")
    if url:
        safe_url = _html.escape(str(url))
        card += (
            f'<div style="margin-top: 6px;"><a href="{safe_url}" '
            f'target="_blank" rel="noopener" style="color: #F97316; '
            f'font-size: 12px; word-break: break-all;">{safe_url}</a></div>'
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
    """Draw the Sources expander, keeping curated and user uploads visually apart."""
    trusted = groups.get("trusted") or []
    user = groups.get("user") or []
    unknown = groups.get("unknown") or []
    if not trusted and not user and not unknown:
        return
    headers = (
        (trusted, "Medibo knowledge base — trusted"),
        (user, "Your uploaded data — user-provided"),
        (unknown, "Unverified source — origin unknown"),
    )
    with st.expander("Sources", expanded=False):
        for i, (items, title) in enumerate(headers):
            if not items:
                continue
            margin = "" if i == 0 else "margin: 10px 0 6px; "
            st.markdown(
                "<div style='font-size: 11px; font-weight: 600; color: #A1A1AA; "
                f"{margin}text-transform: uppercase;'>{title}</div>",
                unsafe_allow_html=True,
            )
            for src in items:
                st.markdown(render_source_card(src), unsafe_allow_html=True)


def _as_citation(item) -> Citation:
    return item if isinstance(item, Citation) else Citation(**item)


def render_citations(citations) -> None:
    """Render the numbered citation list of a grounded answer (or history)."""
    items = [_as_citation(c) for c in citations or ()]
    if not items:
        return
    st.markdown(citations_markdown(items))


# --- Sidebar ---
with st.sidebar:
    header_placeholder = st.empty()

    # Document Ingestion Section
    st.markdown("<div style='font-size: 11px; font-weight: 600; color: #A1A1AA; margin-bottom: 8px; text-transform: uppercase;'>Knowledge Base</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 11px; color: #A1A1AA; margin-bottom: 10px; line-height: 1.5;'>"
        "Uploads go to <span style='color: #F97316;'>My data</span>. "
        "Default medical knowledge is read-only and is never rebuilt or deleted here."
        "</div>",
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

    st.write("") # Spacer

    # --- Knowledge Base Stats: default corpus and my data are counted apart ---
    try:
        from ingestion.indexers import ElasticsearchVectorStore

        counts = dataset_counts()
        es_store = ElasticsearchVectorStore()
        if es_store.es and es_store.es.ping():
            status_text = "Online"
            status_color = "#10B981"
        else:
            status_text = "Offline"
            status_color = "#EF4444"

        st.markdown(f"""
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-val">{counts['default']:,}</div>
                    <div class="stat-lbl">Default knowledge</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val">{counts['user']:,}</div>
                    <div class="stat-lbl">My data</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val" style="color: {status_color}; display: flex; align-items: center; justify-content: center; gap: 6px;">
                        <span class="status-dot" style="background-color: {status_color};"></span>
                        {status_text}
                    </div>
                    <div class="stat-lbl">Status</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown("""
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-val">0</div>
                    <div class="stat-lbl">Default knowledge</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val">0</div>
                    <div class="stat-lbl">My data</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val" style="color: #EF4444; display: flex; align-items: center; justify-content: center; gap: 6px;">
                        <span class="status-dot" style="background-color: #EF4444;"></span>
                        Offline
                    </div>
                    <div class="stat-lbl">Status</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

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

    st.write("") # Spacer

    # --- SETTINGS SECTION ---
    st.markdown("<div style='font-size: 11px; font-weight: 600; color: #A1A1AA; margin-bottom: 8px; margin-top: 16px; text-transform: uppercase;'>Provider & Model</div>", unsafe_allow_html=True)
    
    default_provider_index = 0 if config.LLM_PROVIDER.lower() == "groq" else 1
    ui_provider = st.selectbox("LLM Provider", ["Groq (Cloud)", "Ollama (Local)"], index=default_provider_index, label_visibility="collapsed")
    selected_provider_key = "groq" if "Groq" in ui_provider else "ollama"
    
    if selected_provider_key == "groq":
        groq_models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "Custom..."]
        is_custom = config.LLM_MODEL not in [m for m in groq_models if m != "Custom..."]
        
        default_idx = groq_models.index("Custom...") if is_custom else groq_models.index(config.LLM_MODEL)
        selected_option = st.selectbox("Model", groq_models, index=default_idx)
        
        if selected_option == "Custom...":
            ui_model = st.text_input("Enter model name", value=config.LLM_MODEL if is_custom else "").strip()
        else:
            ui_model = selected_option
    else:
        ollama_models = ["llama3", "mistral", "gemma2", "phi3", "Custom..."]
        is_custom = config.OLLAMA_MODEL not in [m for m in ollama_models if m != "Custom..."]
        
        if is_custom and config.OLLAMA_MODEL:
            default_idx = ollama_models.index("Custom...")
        else:
            default_idx = ollama_models.index(config.OLLAMA_MODEL) if config.OLLAMA_MODEL in ollama_models else 0
            
        selected_option = st.selectbox("Model", ollama_models, index=default_idx)
        
        if selected_option == "Custom...":
            ui_model = st.text_input("Enter model name", value=config.OLLAMA_MODEL if is_custom else "").strip()
        else:
            ui_model = selected_option
        
    st.markdown("<div style='font-size: 11px; font-weight: 600; color: #A1A1AA; margin-bottom: 8px; margin-top: 16px; text-transform: uppercase;'>Search Strategy</div>", unsafe_allow_html=True)
    retrieval_method = st.selectbox("Search Strategy", ["hybrid", "vector", "text"], index=0, label_visibility="collapsed")
    
    c1, c2 = st.columns(2)
    with c1:
        use_reranking = st.checkbox("Rerank", value=True)
    with c2:
        use_query_rewrite = st.checkbox("Rewrite", value=True)
        
    prompt_style = st.selectbox("Output Style", ["detailed", "concise", "structured"], index=0)

    # Render dynamic header
    provider_color = "#10B981" if selected_provider_key == "ollama" else "#3B82F6"
    header_placeholder.markdown(f"""
        <div style='margin-bottom: 24px;'>
            <div style='font-size: 18px; font-weight: 600; color: #F5F5F5;'>potbot workspace</div>
            <div style='font-size: 12px; color: #A1A1AA;'>Provider: <span style='color: {provider_color}; font-weight: 600;'>{selected_provider_key.upper()}</span></div>
        </div>
    """, unsafe_allow_html=True)


# --- Main Header ---
st.markdown("""
<div class="app-header">
    <div class="app-header-logo">pb</div>
    <div>
        <h1 class="app-header-title">potbot</h1>
        <p class="app-header-subtitle">Internal Intelligence Platform</p>
    </div>
</div>
""", unsafe_allow_html=True)


# --- Render Empty State or Conversation History ---
if not st.session_state.messages:
    st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🔒</div>
            <h2 class="empty-state-title">Secure Enterprise RAG</h2>
            <p class="empty-state-subtitle">Ask questions across your internal documents. Fully private embeddings and execution ensure no data leaks.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='suggestion-marker'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 11px; color: #A1A1AA; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 12px;'>Suggested Prompts</p>", unsafe_allow_html=True)
    
    suggestions = [
        "Summarize the recent engineering guidelines.",
        "How do I setup a local database model?",
        "What are the main security protocols for internal documents?"
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
            st.markdown(msg["content"])
            
            if msg["role"] == "assistant":
                if msg.get("response_time_ms"):
                    st.markdown(
                        f"<div style='font-size: 12px; color: #A1A1AA; margin-top: 8px;'>"
                        f"Generated in {msg['response_time_ms']}ms • {msg['total_tokens']} tokens"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                if msg.get("evidence_status") == "insufficient":
                    st.caption("No sources cited — general-knowledge answer.")
                elif msg.get("sources"):
                    if msg.get("trust_level") == TRUST_LEVEL_USER_ONLY:
                        st.info(USER_ONLY_BANNER)
                    render_citations(msg.get("citations"))
                    render_sources_block(split_sources(msg["sources"]))
                
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


# --- Chat Input & Execution ---
active_prompt = None

if st.session_state.pending_prompt:
    active_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

chat_input_val = st.chat_input("Message potbot...")
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

                st.markdown(response.answer)

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
                    if evidence.trust_level == TRUST_LEVEL_USER_ONLY:
                        st.info(USER_ONLY_BANNER)
                    # Citations come from the supporting chunks only — unknown
                    # provenance is shown separately and never cited.
                    render_citations(response.citations)
                    render_sources_block(groups)
                else:
                    # The answer already carries the explicit warning line —
                    # never attach sources or citations to an ungrounded answer.
                    st.caption("No sources cited — general-knowledge answer.")

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
