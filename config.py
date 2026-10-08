"""
potbot - Centralized configuration.

Loads settings from environment variables (.env file) with sensible defaults.
All ingestion tuning knobs are prefixed with INGESTION_ and can be set per
environment (laptop vs. server) without changing any source code.

LLM Provider selection
----------------------
Set  LLM_PROVIDER=groq   (default) to use the Groq cloud API.
Set  LLM_PROVIDER=ollama          to use a local Ollama instance.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# --- LLM Provider Selection ---
# "groq" (default) or "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# --- Groq LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")

# --- Ollama (local / offline LLM) ---
# When running locally:  http://localhost:11434
# When running in Docker Compose with the ollama profile: http://ollama:11434
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# --- Embeddings (local, sentence-transformers) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Elasticsearch ---
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "potbot_documents")

# --- Medical datasets (Phase 6) ---
# medibo_default = shipped read-only corpus; medibo_user = user uploads.
MEDICAL_DEFAULT_INDEX = os.getenv("MEDICAL_DEFAULT_INDEX", "medibo_default")
MEDICAL_USER_INDEX = os.getenv("MEDICAL_USER_INDEX", "medibo_user")

# Phase 6.6 — application read path. Every query searches the default corpus
# plus whatever the user has uploaded; the default half is always present, so
# retrieval keeps working when the user dataset has been cleared (Elasticsearch
# is asked to ignore a missing index in the list).
MEDICAL_RETRIEVAL_INDEXES = os.getenv(
    "MEDICAL_RETRIEVAL_INDEXES",
    f"{MEDICAL_DEFAULT_INDEX},{MEDICAL_USER_INDEX}",
)

# Phase 6.6 — application write path. UI uploads and folder ingestion land in
# the user dataset only; medibo_default is written exclusively by the backend
# build script (which opts in with allow_default_write=True).
MEDICAL_UPLOAD_INDEX = os.getenv("MEDICAL_UPLOAD_INDEX", MEDICAL_USER_INDEX)

# --- PostgreSQL ---
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "potbot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "potbot")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "potbot_secret")

DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Retrieval ---
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))
# Number of hybrid candidates fetched before re-ranking. Keep separate from
# TOP_K_RETRIEVAL (final chunks sent to the LLM) and RERANK_TOP_N (reranker
# top-N). 20 is the measured practical default; larger pools cost latency.
RETRIEVAL_CANDIDATE_POOL = int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "20"))
# RRF fusion weights for hybrid retrieval (vector + BM25). Must sum to 1.0;
# 0.5/0.5 measured best (recovers BM25-strong queries with no regressions).
VECTOR_RRF_WEIGHT = float(os.getenv("VECTOR_RRF_WEIGHT", "0.5"))
TEXT_RRF_WEIGHT = float(os.getenv("TEXT_RRF_WEIGHT", "0.5"))
# RRF smoothing constant applied to every rank (kept at the measured default).
RRF_K = int(os.getenv("RRF_K", "60"))

# --- Phase 7: trusted evidence gate ---
# The gate runs after retrieval/re-ranking and only READS their scores; RRF,
# the candidate pool, the re-ranker and the embedding model are untouched.
# Thresholds calibrated on medibo_default: relevant queries score +4.8..+7.4
# on the cross-encoder logit (absent queries -7.6..-11.2) and 0.68..0.75
# cosine query/chunk similarity (absent queries 0.19..0.32).
# Policy name is recorded on every response so offline threshold sweeps can
# be attributed later.
EVIDENCE_POLICY_NAME = os.getenv("EVIDENCE_POLICY_NAME", "evidence-v1")
# Minimum number of retrieved documents that must clear the relevance threshold
# for the answer to be treated as knowledge-base grounded.
EVIDENCE_MIN_RELEVANT_DOCS = int(os.getenv("EVIDENCE_MIN_RELEVANT_DOCS", "1"))
# Cross-encoder rerank logit threshold (used whenever re-ranking is enabled).
EVIDENCE_MIN_RERANK_SCORE = float(os.getenv("EVIDENCE_MIN_RERANK_SCORE", "1.0"))
# Cosine query/chunk similarity threshold (used when re-ranking is disabled).
EVIDENCE_MIN_SIMILARITY = float(os.getenv("EVIDENCE_MIN_SIMILARITY", "0.60"))

# LLM query rewriting. Off by default (measured net-negative for retrieval);
# explicit opt-in per-query is still available via use_query_rewriting=True.
USE_QUERY_REWRITING = os.getenv("USE_QUERY_REWRITING", "false").lower() == "true"

# --- Ingestion Pipeline Parallelism ---
# ThreadPoolExecutor workers for I/O-bound file loading
INGESTION_LOADER_WORKERS = int(os.getenv("INGESTION_LOADER_WORKERS", "4"))
# ProcessPoolExecutor workers for CPU-bound text chunking
INGESTION_CHUNKER_WORKERS = int(os.getenv("INGESTION_CHUNKER_WORKERS", "2"))
# Number of chunks per embedding batch
INGESTION_EMBED_BATCH_SIZE = int(os.getenv("INGESTION_EMBED_BATCH_SIZE", "64"))
# Number of embedded chunks per Elasticsearch bulk request
INGESTION_INDEX_BULK_SIZE = int(os.getenv("INGESTION_INDEX_BULK_SIZE", "200"))
# Max items in the inter-stage queue (backpressure). 0 = unlimited
INGESTION_QUEUE_SIZE = int(os.getenv("INGESTION_QUEUE_SIZE", "500"))

# --- Ingestion Device ---
# Embedding compute device: "auto" | "cuda" | "mps" | "cpu"
INGESTION_DEVICE = os.getenv("INGESTION_DEVICE", "auto")

# --- Incremental Ingestion & Checkpointing ---
INGESTION_CHECKPOINT_PATH = os.getenv("INGESTION_CHECKPOINT_PATH", ".ingestion_state.db")

# --- Embedding Cache ---
EMBED_CACHE_ENABLED = os.getenv("EMBED_CACHE_ENABLED", "true").lower() == "true"
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE_PATH", ".embed_cache.db")
# Max entries in LRU cache. <= 0 means unlimited.
EMBED_CACHE_MAX_ENTRIES = int(os.getenv("EMBED_CACHE_MAX_ENTRIES", "100000"))
