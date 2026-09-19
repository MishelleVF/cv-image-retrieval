from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import (
    build_class_ground_truth,
    build_query_map,
    class_label,
    ensure_dataset_ready,
    iter_image_paths,
    load_ground_truth,
    load_image,
)
from src.evaluation.metrics import mean_average_precision, precision_at_k
from src.features.bovw import build_vocabulary_from_dataset
from src.features.hog import compute_hog_matrix, image_hog
from src.features.metric_learning import LearnedMetricEmbedding, compute_fused_feature_matrix, fused_descriptor_for_path
from src.retrieval.rerank import sift_match_score
from src.retrieval.search import cosine_rank, euclidean_rank
from src.utils.config import OUTPUT_DIR, PROJECT_ROOT, ensure_dirs
from src.utils.progress import configure_logging

TOP_K = 5
FLOW3_CANDIDATES = 75
FLOW4_CANDIDATES = 100
LOGGER = logging.getLogger(__name__)


def _score_vector(query_vector, matrix, metric: str) -> np.ndarray:
    query = np.ravel(np.asarray(query_vector, dtype=np.float32))
    database = np.asarray(matrix, dtype=np.float32)
    if metric == "cosine":
        db_norms = np.linalg.norm(database, axis=1)
        db_norms[db_norms == 0] = 1.0
        query_norm = max(float(np.linalg.norm(query)), 1e-12)
        return (database @ query) / (db_norms * query_norm)
    if metric == "euclidean":
        distances = np.linalg.norm(database - query, axis=1)
        return 1.0 / (1.0 + distances)
    raise ValueError(f"Unsupported metric: {metric}")


def _rank_query_with_scores(
    query_id: str,
    query_vector,
    matrix,
    ids: list[str],
    metric: str,
) -> tuple[list[str], dict[str, float]]:
    if metric == "cosine":
        order = cosine_rank(query_vector, matrix, top_k=None)
    elif metric == "euclidean":
        order = euclidean_rank(query_vector, matrix, top_k=None)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    scores = _score_vector(query_vector, matrix, metric)
    ranked = [ids[index] for index in order if ids[index] != query_id]
    score_map = {ids[index]: float(scores[index]) for index in order if ids[index] != query_id}
    return ranked, score_map


def _evaluate_ranked_lists(
    ranked_lists: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
) -> dict:
    query_relevant_map = {
        query_id: set(relevant)
        for query_id, relevant in ground_truth.items()
        if query_id in ranked_lists
    }
    return {
        "mAP": mean_average_precision(query_relevant_map, ranked_lists),
        "mean_precision_at_5": sum(
            precision_at_k(relevant, ranked_lists.get(query_id, []), TOP_K)
            for query_id, relevant in query_relevant_map.items()
        )
        / max(len(query_relevant_map), 1),
        "precision_at_5_by_query": {
            query_id: precision_at_k(relevant, ranked_lists.get(query_id, []), TOP_K)
            for query_id, relevant in query_relevant_map.items()
        },
    }


def _evaluate_matrix_flow(
    matrix,
    ids: list[str],
    query_vectors: dict[str, object],
    ground_truth: dict[str, list[str]],
    metric: str = "cosine",
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]], dict]:
    ranked_lists: dict[str, list[str]] = {}
    score_maps: dict[str, dict[str, float]] = {}
    for query_id, query_vector in query_vectors.items():
        ranked, scores = _rank_query_with_scores(query_id, query_vector, matrix, ids, metric)
        ranked_lists[query_id] = ranked
        score_maps[query_id] = scores
    return ranked_lists, score_maps, _evaluate_ranked_lists(ranked_lists, ground_truth)


def _feature_row(
    query_id: str,
    candidate_id: str,
    dataset_root: Path,
    score_sources: dict[str, dict[str, dict[str, float]]],
    geometry_cache: dict[tuple[str, str], float],
) -> list[float]:
    key = (query_id, candidate_id)
    if key not in geometry_cache:
        geometry_cache[key] = sift_match_score(
            dataset_root / query_id,
            dataset_root / candidate_id,
            top_matches=20,
        )
    return [
        score_sources["flow1_bovw"].get(query_id, {}).get(candidate_id, 0.0),
        score_sources["enhanced_bovw"].get(query_id, {}).get(candidate_id, 0.0),
        score_sources["flow2_metric"].get(query_id, {}).get(candidate_id, 0.0),
        score_sources["hog"].get(query_id, {}).get(candidate_id, 0.0),
        geometry_cache[key],
    ]


def _candidate_pool(query_id: str, ranked_sources: dict[str, dict[str, list[str]]], limit: int) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for source in ranked_sources.values():
        for candidate_id in source.get(query_id, [])[:limit]:
            if candidate_id not in seen and candidate_id != query_id:
                seen.add(candidate_id)
                candidates.append(candidate_id)
    return candidates


def _normalize_candidate_scores(candidates: list[str], scores: dict[str, float]) -> dict[str, float]:
    if not candidates:
        return {}
    values = np.asarray([scores.get(candidate_id, 0.0) for candidate_id in candidates], dtype=np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    span = max(max_value - min_value, 1e-12)
    return {
        candidate_id: (scores.get(candidate_id, 0.0) - min_value) / span
        for candidate_id in candidates
    }


def _logistic_fusion_flow(
    ground_truth: dict[str, list[str]],
    dataset_root: Path,
    ranked_sources: dict[str, dict[str, list[str]]],
    score_sources: dict[str, dict[str, dict[str, float]]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]], dict]:
    geometry_cache: dict[tuple[str, str], float] = {}
    training_rows: list[list[float]] = []
    training_labels: list[int] = []

    LOGGER.info("Training logistic score fusion over top-%d candidate pools", FLOW3_CANDIDATES)
    for query_id, relevant_items in ground_truth.items():
        relevant = set(relevant_items)
        for candidate_id in _candidate_pool(query_id, ranked_sources, FLOW3_CANDIDATES):
            training_rows.append(_feature_row(query_id, candidate_id, dataset_root, score_sources, geometry_cache))
            training_labels.append(1 if candidate_id in relevant else 0)

    if len(set(training_labels)) < 2:
        LOGGER.warning("Logistic fusion saw only one class; falling back to flow2 ranking")
        fallback = {query_id: list(ranked) for query_id, ranked in ranked_sources["flow2_metric"].items()}
        fallback_scores = {
            query_id: {
                candidate_id: 1.0 / (rank + 1.0)
                for rank, candidate_id in enumerate(ranked)
            }
            for query_id, ranked in fallback.items()
        }
        return fallback, fallback_scores, _evaluate_ranked_lists(fallback, ground_truth)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(np.asarray(training_rows, dtype=np.float32), np.asarray(training_labels, dtype=np.int32))

    fused_rankings: dict[str, list[str]] = {}
    fused_scores: dict[str, dict[str, float]] = {}
    for query_id in ground_truth:
        candidates = _candidate_pool(query_id, ranked_sources, FLOW3_CANDIDATES)
        rows = np.asarray(
            [_feature_row(query_id, candidate_id, dataset_root, score_sources, geometry_cache) for candidate_id in candidates],
            dtype=np.float32,
        )
        probabilities = model.predict_proba(rows)[:, 1] if len(candidates) else np.array([], dtype=np.float32)
        scored = sorted(zip(candidates, probabilities), key=lambda item: float(item[1]), reverse=True)
        reranked = [candidate_id for candidate_id, _ in scored]
        fused_scores[query_id] = {candidate_id: float(probability) for candidate_id, probability in scored}
        seen = set(reranked)
        reranked.extend(candidate_id for candidate_id in ranked_sources["flow2_metric"].get(query_id, []) if candidate_id not in seen)
        fused_rankings[query_id] = reranked

    summary = _evaluate_ranked_lists(fused_rankings, ground_truth)
    summary["model"] = {
        "name": "LogisticRegression",
        "feature_names": [
            "flow1_bovw_cosine",
            "rootsift_tfidf_bovw_cosine",
            "flow2_metric_cosine",
            "hog_cosine",
            "sift_homography_score",
        ],
        "training_pairs": len(training_labels),
        "positive_pairs": int(sum(training_labels)),
        "candidate_pool_per_query": FLOW3_CANDIDATES,
        "note": "The logistic fusion is trained and evaluated on the available query labels for comparison, not as a held-out generalization estimate.",
    }
    return fused_rankings, fused_scores, summary


def _balanced_hybrid_flow(
    ground_truth: dict[str, list[str]],
    dataset_root: Path,
    base_rankings: dict[str, list[str]],
    ranked_sources: dict[str, dict[str, list[str]]],
    score_sources: dict[str, dict[str, dict[str, float]]],
    logistic_scores: dict[str, dict[str, float]],
) -> tuple[dict[str, list[str]], dict]:
    geometry_cache: dict[tuple[str, str], float] = {}
    hybrid_rankings: dict[str, list[str]] = {}
    weights = {
        "rootsift_tfidf": 0.40,
        "logistic": 0.25,
        "metric": 0.20,
        "base_bovw": 0.10,
        "homography": 0.05,
    }

    LOGGER.info("Flow 4: balanced hybrid reranking over top-%d RootSIFT-TFIDF candidates", FLOW4_CANDIDATES)
    for query_id in ground_truth:
        candidates = list(base_rankings.get(query_id, [])[:FLOW4_CANDIDATES])
        for extra_id in _candidate_pool(query_id, ranked_sources, FLOW4_CANDIDATES):
            if extra_id not in candidates:
                candidates.append(extra_id)

        homography_scores = {}
        for candidate_id in candidates:
            key = (query_id, candidate_id)
            if key not in geometry_cache:
                geometry_cache[key] = sift_match_score(
                    dataset_root / query_id,
                    dataset_root / candidate_id,
                    top_matches=20,
                )
            homography_scores[candidate_id] = geometry_cache[key]

        normalized = {
            "rootsift_tfidf": _normalize_candidate_scores(candidates, score_sources["enhanced_bovw"].get(query_id, {})),
            "logistic": _normalize_candidate_scores(candidates, logistic_scores.get(query_id, {})),
            "metric": _normalize_candidate_scores(candidates, score_sources["flow2_metric"].get(query_id, {})),
            "base_bovw": _normalize_candidate_scores(candidates, score_sources["flow1_bovw"].get(query_id, {})),
            "homography": _normalize_candidate_scores(candidates, homography_scores),
        }

        scored = []
        for candidate_id in candidates:
            final_score = sum(
                weight * normalized[name].get(candidate_id, 0.0)
                for name, weight in weights.items()
            )
            scored.append((candidate_id, final_score))

        scored.sort(key=lambda item: item[1], reverse=True)
        reranked = [candidate_id for candidate_id, _ in scored]
        seen = set(reranked)
        reranked.extend(candidate_id for candidate_id in base_rankings.get(query_id, []) if candidate_id not in seen)
        hybrid_rankings[query_id] = reranked

    summary = _evaluate_ranked_lists(hybrid_rankings, ground_truth)
    summary["weights"] = weights
    summary["candidate_pool_per_query"] = FLOW4_CANDIDATES
    summary["note"] = "Balanced hybrid reranks the RootSIFT-TFIDF top candidates using normalized classical and learned scores."
    return hybrid_rankings, summary


def _save_qualitative_figures(
    rankings: dict[str, list[str]],
    dataset_root: Path,
    output_dir: Path,
    max_figures: int = 4,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for query_id, retrieved in list(rankings.items())[:max_figures]:
        fig, axes = plt.subplots(1, TOP_K + 1, figsize=(12, 3))
        paths = [dataset_root / query_id] + [dataset_root / item for item in retrieved[:TOP_K]]
        titles = ["query"] + [f"top {i}" for i in range(1, TOP_K + 1)]
        for axis, path, title in zip(axes, paths, titles):
            axis.imshow(load_image(path))
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        fig.tight_layout()
        target = output_dir / f"{Path(query_id).stem}_top5.png"
        fig.savefig(target, dpi=140)
        plt.close(fig)
        saved.append(target.as_posix())
    return saved


def _print_comparison_table(flows: dict[str, dict]) -> None:
    print("\nFLOW COMPARISON")
    print(f"{'flow':42s} {'mAP':>10s} {'P@5':>10s}")
    print("-" * 66)
    for name, payload in flows.items():
        summary = payload["summary"]
        print(f"{name:42s} {summary['mAP']:10.4f} {summary['mean_precision_at_5']:10.4f}")

    best_map = max(flows.items(), key=lambda item: item[1]["summary"]["mAP"])
    best_p5 = max(flows.items(), key=lambda item: item[1]["summary"]["mean_precision_at_5"])
    print("-" * 66)
    print(f"Best by mAP: {best_map[0]} ({best_map[1]['summary']['mAP']:.4f})")
    print(f"Best by P@5: {best_p5[0]} ({best_p5[1]['summary']['mean_precision_at_5']:.4f})")


def main() -> None:
    configure_logging()
    ensure_dirs()
    dataset_root = PROJECT_ROOT / "data" / "local"
    gt_path = dataset_root / "ground_truth.json"
    allow_download = os.environ.get("CVIR_ALLOW_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes", "on"}
    LOGGER.info("Preparing dataset at %s", dataset_root)
    ensure_dataset_ready(dataset_root, gt_path, allow_download=allow_download)

    all_image_paths = list(iter_image_paths(dataset_root))
    if not all_image_paths:
        raise FileNotFoundError(f"No images found in: {dataset_root}")

    ground_truth = load_ground_truth(gt_path, dataset_root)
    if not ground_truth:
        ground_truth = build_class_ground_truth(dataset_root)
    if not ground_truth:
        raise FileNotFoundError("No ground truth could be loaded or inferred from class folders/file prefixes.")

    image_paths = all_image_paths
    query_map = build_query_map(dataset_root)
    ground_truth = {
        query_id: [item for item in relevant if (dataset_root / item).exists()]
        for query_id, relevant in ground_truth.items()
        if query_id in query_map and dataset_root / query_id in image_paths
    }
    ground_truth = {query_id: relevant for query_id, relevant in ground_truth.items() if relevant}
    if not ground_truth:
        raise ValueError("No evaluable queries remain after applying the bounded image subset.")
    LOGGER.info("Evaluating %d images and %d queries", len(image_paths), len(ground_truth))

    LOGGER.info("Flow 1: base SIFT BoVW vocabulary=96")
    base_bovw = build_vocabulary_from_dataset(dataset_root, vocabulary_size=96)
    base_matrix, base_ids = base_bovw.compute_database_histograms(image_paths, dataset_root)
    base_query_vectors = {query_id: base_bovw.image_histogram(dataset_root / query_id) for query_id in ground_truth}
    flow1_rankings, flow1_scores, flow1_summary = _evaluate_matrix_flow(
        base_matrix, base_ids, base_query_vectors, ground_truth, metric="cosine"
    )

    LOGGER.info("Computing HOG once for score fusion and comparison")
    hog_matrix, hog_ids = compute_hog_matrix(image_paths, dataset_root)
    hog_query_vectors = {query_id: image_hog(dataset_root / query_id) for query_id in ground_truth}
    hog_rankings, hog_scores, hog_summary = _evaluate_matrix_flow(
        hog_matrix, hog_ids, hog_query_vectors, ground_truth, metric="cosine"
    )

    LOGGER.info("Flow 2: RootSIFT BoVW vocabulary=512 with TF-IDF and learned distance function")
    enhanced_bovw = build_vocabulary_from_dataset(
        dataset_root,
        vocabulary_size=512,
        use_root_sift=True,
        use_tfidf=True,
    )
    enhanced_matrix, enhanced_ids = enhanced_bovw.compute_database_histograms(image_paths, dataset_root)
    enhanced_query_vectors = {
        query_id: enhanced_bovw.image_histogram(dataset_root / query_id)
        for query_id in ground_truth
    }
    enhanced_rankings, enhanced_scores, enhanced_summary = _evaluate_matrix_flow(
        enhanced_matrix, enhanced_ids, enhanced_query_vectors, ground_truth, metric="cosine"
    )

    enhanced_fused_features, enhanced_fused_ids = compute_fused_feature_matrix(image_paths, dataset_root, enhanced_bovw)
    enhanced_fused_labels = [class_label(dataset_root / image_id, dataset_root) for image_id in enhanced_fused_ids]
    learned_embedding = LearnedMetricEmbedding(
        n_components=min(64, enhanced_fused_features.shape[1], max(1, enhanced_fused_features.shape[0] - 1))
    )
    learned_embedding.fit(enhanced_fused_features, enhanced_fused_labels)
    learned_matrix = learned_embedding.transform(enhanced_fused_features)
    learned_query_vectors = {
        query_id: learned_embedding.transform(
            fused_descriptor_for_path(dataset_root / query_id, enhanced_bovw, dataset_root).reshape(1, -1)
        )[0]
        for query_id in ground_truth
    }
    flow2_rankings, flow2_scores, flow2_summary = _evaluate_matrix_flow(
        learned_matrix, enhanced_fused_ids, learned_query_vectors, ground_truth, metric="cosine"
    )

    LOGGER.info("Flow 3: logistic score fusion")
    ranked_sources = {
        "flow1_bovw": flow1_rankings,
        "enhanced_bovw": enhanced_rankings,
        "flow2_metric": flow2_rankings,
        "hog": hog_rankings,
    }
    score_sources = {
        "flow1_bovw": flow1_scores,
        "enhanced_bovw": enhanced_scores,
        "flow2_metric": flow2_scores,
        "hog": hog_scores,
    }
    flow3_rankings, flow3_scores, flow3_summary = _logistic_fusion_flow(
        ground_truth,
        dataset_root,
        ranked_sources,
        score_sources,
    )

    LOGGER.info("Flow 4: balanced hybrid ranking")
    flow4_rankings, flow4_summary = _balanced_hybrid_flow(
        ground_truth,
        dataset_root,
        enhanced_rankings,
        ranked_sources,
        score_sources,
        flow3_scores,
    )
    LOGGER.info("All flows complete")

    flows = {
        "flow1_base_sift_bovw": {
            "description": "SIFT + BoVW vocabulary 96 + cosine ranking",
            "summary": flow1_summary,
        },
        "flow2_rootsift_tfidf_dist_func": {
            "description": "RootSIFT + BoVW vocabulary 512 + TF-IDF + NCA learned distance function",
            "auxiliary_rootsift_tfidf_bovw_summary": enhanced_summary,
            "summary": flow2_summary,
        },
        "flow3_logistic_score_fusion": {
            "description": "Logistic regression fusion over BoVW, enhanced BoVW, learned metric, HOG, and homography scores",
            "summary": flow3_summary,
        },
        "flow4_balanced_hybrid": {
            "description": "Balanced hybrid reranking over RootSIFT-TFIDF candidates using weighted normalized scores",
            "summary": flow4_summary,
        },
        "reference_hog": {
            "description": "HOG cosine baseline used as a fusion feature",
            "summary": hog_summary,
        },
    }
    rankings = {
        "flow1_base_sift_bovw": {query_id: ranked[:TOP_K] for query_id, ranked in flow1_rankings.items()},
        "flow2_rootsift_tfidf_dist_func": {query_id: ranked[:TOP_K] for query_id, ranked in flow2_rankings.items()},
        "flow3_logistic_score_fusion": {query_id: ranked[:TOP_K] for query_id, ranked in flow3_rankings.items()},
        "flow4_balanced_hybrid": {query_id: ranked[:TOP_K] for query_id, ranked in flow4_rankings.items()},
        "reference_hog": {query_id: ranked[:TOP_K] for query_id, ranked in hog_rankings.items()},
    }

    metrics = {
        "constraints": {
            "dataset_subsampling": "none",
            "query_subsampling": "none",
            "flow1": "SIFT + BoVW vocabulary 96 + cosine ranking",
            "flow2": "RootSIFT + BoVW vocabulary 512 + TF-IDF + NCA learned distance function",
            "flow3": "Logistic regression score fusion over classical scores",
            "flow4": "Balanced hybrid reranking over RootSIFT-TFIDF candidates",
            "note": "Flows run sequentially. Flow 3 trains and evaluates on the available query labels for project comparison.",
        },
        "dataset": {
            "root": dataset_root.as_posix(),
            "available_images": len(all_image_paths),
            "evaluated_images": len(image_paths),
            "evaluated_queries": len(ground_truth),
        },
        "flows": flows,
    }

    (OUTPUT_DIR / "metrics" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "rankings" / "top5_rankings.json").write_text(json.dumps(rankings, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "representations" / "ground_truth_used.json").write_text(
        json.dumps(ground_truth, indent=2), encoding="utf-8"
    )

    figure_paths = _save_qualitative_figures(flow4_rankings, dataset_root, OUTPUT_DIR / "figures")
    LOGGER.info("Writing metrics, rankings, manifest, and %d qualitative figures", len(figure_paths))
    manifest = {
        "metrics": "outputs/metrics/metrics.json",
        "rankings": "outputs/rankings/top5_rankings.json",
        "ground_truth_used": "outputs/representations/ground_truth_used.json",
        "qualitative_figures": figure_paths,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _print_comparison_table(flows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
