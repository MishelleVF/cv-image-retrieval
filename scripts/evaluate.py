from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

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
from src.retrieval.rerank import rerank_by_sift_similarity
from src.retrieval.search import cosine_rank, euclidean_rank
from src.utils.config import OUTPUT_DIR, PROJECT_ROOT, ensure_dirs
from src.utils.progress import configure_logging

TOP_K = 5
LOGGER = logging.getLogger(__name__)


def _rank_query(
    query_id: str,
    query_vector,
    matrix,
    ids: list[str],
    metric: str,
) -> list[str]:
    if metric == "cosine":
        order = cosine_rank(query_vector, matrix, top_k=None)
    elif metric == "euclidean":
        order = euclidean_rank(query_vector, matrix, top_k=None)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return [ids[index] for index in order if ids[index] != query_id]


def _rerank_query(query_id: str, initial_ranked: list[str], dataset_root: Path, top_candidates: int = 0) -> list[str]:
    if top_candidates <= 0:
        return list(initial_ranked)
    candidates = initial_ranked[: max(1, top_candidates)]
    return rerank_by_sift_similarity(query_id, candidates, dataset_root, top_candidates=len(candidates), top_matches=8)


def _evaluate_representation(
    name: str,
    matrix,
    ids: list[str],
    query_vectors: dict[str, object],
    ground_truth: dict[str, list[str]],
    metrics: tuple[str, ...],
    dataset_root: Path,
    rerank_candidates: int = 0,
) -> tuple[dict, dict]:
    rankings: dict[str, dict[str, list[str]]] = {}
    summary: dict[str, dict] = {}

    for metric in metrics:
        ranked_lists = {
            query_id: _rank_query(query_id, query_vectors[query_id], matrix, ids, metric)
            for query_id in query_vectors
        }
        query_relevant_map = {
            query_id: set(relevant)
            for query_id, relevant in ground_truth.items()
            if query_id in ranked_lists
        }

        raw_summary = {
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

        if rerank_candidates <= 0:
            reranked_lists = {query_id: list(ranked) for query_id, ranked in ranked_lists.items()}
            reranked_summary = raw_summary
        else:
            reranked_lists = {
                query_id: _rerank_query(query_id, ranked, dataset_root, top_candidates=rerank_candidates)
                for query_id, ranked in ranked_lists.items()
            }
            reranked_summary = {
                "mAP": mean_average_precision(query_relevant_map, reranked_lists),
                "mean_precision_at_5": sum(
                    precision_at_k(relevant, reranked_lists.get(query_id, []), TOP_K)
                    for query_id, relevant in query_relevant_map.items()
                )
                / max(len(query_relevant_map), 1),
                "precision_at_5_by_query": {
                    query_id: precision_at_k(relevant, reranked_lists.get(query_id, []), TOP_K)
                    for query_id, relevant in query_relevant_map.items()
                },
            }

        rankings[metric] = {query_id: ranked[:TOP_K] for query_id, ranked in reranked_lists.items()}
        summary[metric] = {
            "coarse_candidate_pool": rerank_candidates,
            "raw": raw_summary,
            "reranked": reranked_summary,
        }

    return rankings, summary


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

    bovw = build_vocabulary_from_dataset(dataset_root, vocabulary_size=96)
    bovw_matrix, bovw_ids = bovw.compute_database_histograms(image_paths, dataset_root)
    bovw_query_vectors = {query_id: bovw.image_histogram(dataset_root / query_id) for query_id in ground_truth}

    hog_matrix, hog_ids = compute_hog_matrix(image_paths, dataset_root)
    hog_query_vectors = {query_id: image_hog(dataset_root / query_id) for query_id in ground_truth}

    fused_features, fused_ids = compute_fused_feature_matrix(image_paths, dataset_root, bovw)
    fused_labels = [class_label(dataset_root / image_id, dataset_root) for image_id in fused_ids]
    learned_embedding = LearnedMetricEmbedding(n_components=min(64, fused_features.shape[1], max(1, fused_features.shape[0] - 1)))
    learned_embedding.fit(fused_features, fused_labels)
    learned_matrix = learned_embedding.transform(fused_features)
    learned_query_vectors = {
        query_id: learned_embedding.transform(
            fused_descriptor_for_path(dataset_root / query_id, bovw, dataset_root).reshape(1, -1)
        )[0]
        for query_id in ground_truth
    }

    bovw_rankings, bovw_metrics = _evaluate_representation(
        "bovw",
        bovw_matrix,
        bovw_ids,
        bovw_query_vectors,
        ground_truth,
        ("cosine", "euclidean"),
        dataset_root=dataset_root,
        rerank_candidates=0,
    )
    hog_rankings, hog_metrics = _evaluate_representation(
        "hog",
        hog_matrix,
        hog_ids,
        hog_query_vectors,
        ground_truth,
        ("cosine", "euclidean"),
        dataset_root=dataset_root,
        rerank_candidates=0,
    )
    learned_rankings, learned_metrics = _evaluate_representation(
        "learned",
        learned_matrix,
        fused_ids,
        learned_query_vectors,
        ground_truth,
        ("cosine",),
        dataset_root=dataset_root,
        rerank_candidates=0,
    )
    LOGGER.info("Ranking and metric computation complete")

    metrics = {
        "constraints": {
            "bovw_vocabulary_size": 96,
            "dataset_subsampling": "none",
            "query_subsampling": "none",
            "note": "All available images and all evaluable queries are used. The default pipeline uses raw coarse retrieval only; SIFT geometric reranking is currently disabled because it degraded mAP in validation. The learned embedding is a linear metric-learning projection trained on landmark labels.",
        },
        "dataset": {
            "root": dataset_root.as_posix(),
            "available_images": len(all_image_paths),
            "evaluated_images": len(image_paths),
            "evaluated_queries": len(ground_truth),
        },
        "representations": {
            "bovw": bovw_metrics,
            "hog": hog_metrics,
            "learned": learned_metrics,
        },
        "reranking": {
            "candidate_pool_size": 0,
            "method": "disabled",
            "description": "Geometric reranking is disabled in the default pipeline. Raw coarse retrieval is used for all reported metrics."
        },
    }
    rankings = {"bovw": bovw_rankings, "hog": hog_rankings, "learned": learned_rankings}

    (OUTPUT_DIR / "metrics" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "rankings" / "top5_rankings.json").write_text(json.dumps(rankings, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "representations" / "ground_truth_used.json").write_text(
        json.dumps(ground_truth, indent=2), encoding="utf-8"
    )

    figure_paths = _save_qualitative_figures(bovw_rankings["cosine"], dataset_root, OUTPUT_DIR / "figures")
    LOGGER.info("Writing metrics, rankings, manifest, and %d qualitative figures", len(figure_paths))
    manifest = {
        "metrics": "outputs/metrics/metrics.json",
        "rankings": "outputs/rankings/top5_rankings.json",
        "ground_truth_used": "outputs/representations/ground_truth_used.json",
        "qualitative_figures": figure_paths,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
