"""
Medical embedding evaluation experiment (Phase 5).
Compare: all-MiniLM-L6-v2 (current), pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb (medical),
and sentence-transformers/all-mpnet-base-v2 (strong general).
"""
import json
import time
from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path('data/medical/chunks/medical_chunks.json')
GT_PATH = Path('data/medical/medical_ground_truth.json')
RES_PATH = Path('data/medical/embedding_experiment_results.json')


def load_chunks():
    with open(CHUNKS_PATH) as f:
        return json.load(f)


def load_ground_truth():
    with open(GT_PATH) as f:
        return json.load(f)


def compute_metrics(model_name: str):
    chunks = load_chunks()
    gt = load_ground_truth()
    # embed model
    t0 = time.time()
    st = SentenceTransformer(model_name)
    load_time = time.time() - t0
    # embed chunks
    texts = [c['text'] for c in chunks]
    t1 = time.time()
    vecs = st.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    embed_time = time.time() - t1
    # build index of vecs in memory (numpy)
    import numpy as np
    V = np.array(vecs, dtype=np.float32)
    dim = V.shape[1]
    # embed queries
    hits1 = hits3 = hits5 = 0
    mrr5_sum = 0
    qtexts = [q['query'] for q in gt['queries']]
    Q = st.encode(qtexts, batch_size=16, show_progress_bar=False, normalize_embeddings=True)
    Q = np.array(Q, dtype=np.float32)
    t2 = time.time()
    for i, qitem in enumerate(gt['queries']):
        qv = Q[i]
        sims = V @ qv
        # top k
        k = 5
        idxs = np.argsort(-sims)[:k]
        relevant = set(qitem['relevant_doc_ids'])
        found_at = -1
        top5_docs = []
        for j, ci in enumerate(idxs):
            c = chunks[ci]
            top5_docs.append(c['doc_id'])
            if c['doc_id'] in relevant and found_at == -1:
                found_at = j + 1
        if found_at == 1:
            hits1 += 1
        if found_at <= 3:
            hits3 += 1
        if found_at >= 1 and found_at <= 5:
            hits5 += 1
        if found_at >= 1:
            mrr5_sum += 1.0 / found_at
    n = len(gt['queries'])
    lat = (time.time() - t2) / n if n > 0 else 0
    return {
        'model': model_name,
        'dim': dim,
        'num_chunks': len(chunks),
        'embed_time_s': embed_time,
        'load_time_s': load_time,
        'avg_retrieval_latency_s': lat,
        'hits@1': hits1 / n,
        'hits@3': hits3 / n,
        'hits@5': hits5 / n,
        'mrr@5': mrr5_sum / n,
        'vector_storage_mb_est': (V.size * 4) / (1024*1024),
        'n': n,
    }


def run_all():
    models = [
        'all-MiniLM-L6-v2',
        'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb',
        'sentence-transformers/all-mpnet-base-v2',
    ]
    results = []
    for m in models:
        try:
            res = compute_metrics(m)
            results.append(res)
            print(json.dumps(res))
        except Exception as e:
            results.append({'model': m, 'error': str(e)})
    out = {'results': results, 'timestamp': time.time()}
    with open(RES_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == '__main__':
    import json
    run_all()
