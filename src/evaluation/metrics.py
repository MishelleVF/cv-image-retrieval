from __future__ import annotations

from typing import Iterable, Sequence


def precision_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    top_k = set(retrieved[:k])
    if k <= 0:
        return 0.0
    return len(top_k & relevant) / float(k)


def average_precision(relevant: set[str], retrieved: Sequence[str]) -> float:
    if not relevant:
        return 0.0

    hits = 0
    total_precision = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            total_precision += hits / rank

    return total_precision / len(relevant)


def mean_average_precision(query_relevant_map: dict[str, set[str]], ranked_lists: dict[str, Sequence[str]]) -> float:
    scores = []
    for query_id, relevant in query_relevant_map.items():
        retrieved = ranked_lists.get(query_id, [])
        scores.append(average_precision(relevant, retrieved))

    return sum(scores) / len(scores) if scores else 0.0
