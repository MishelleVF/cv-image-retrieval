from __future__ import annotations

import numpy as np


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_rank(query_hist: np.ndarray, database_hists: np.ndarray, top_k: int | None = 5) -> np.ndarray:
    query = query_hist.astype(np.float32)
    if query.ndim != 1:
        query = np.ravel(query)

    normalized_query = query / max(np.linalg.norm(query), 1e-12)
    normalized_db = normalize_rows(database_hists.astype(np.float32))
    scores = normalized_db @ normalized_query
    order = np.argsort(scores)[::-1]
    return order if top_k is None else order[:top_k]


def euclidean_rank(query_vector: np.ndarray, database_vectors: np.ndarray, top_k: int | None = 5) -> np.ndarray:
    query = np.ravel(query_vector.astype(np.float32))
    distances = np.linalg.norm(database_vectors.astype(np.float32) - query, axis=1)
    order = np.argsort(distances)
    return order if top_k is None else order[:top_k]
