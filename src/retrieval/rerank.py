from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def _read_grayscale(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    max_side = 1024
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))))
    return image


def _select_top_matches(
    query_descriptors: np.ndarray,
    candidate_descriptors: np.ndarray,
    ratio: float = 0.75,
    top_matches: int = 10,
) -> list[cv2.DMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(query_descriptors, candidate_descriptors, k=2)
    good_matches: list[cv2.DMatch] = []
    for pair in matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first is None or second is None:
            continue
        if first.distance < ratio * second.distance:
            good_matches.append(first)

    if not good_matches:
        return []

    good_matches.sort(key=lambda match: match.distance)
    return good_matches[: max(1, min(top_matches, len(good_matches)))]


def sift_match_score(
    query_path: str | Path,
    candidate_path: str | Path,
    ratio: float = 0.75,
    top_matches: int = 8,
) -> float:
    query_gray = _read_grayscale(query_path)
    candidate_gray = _read_grayscale(candidate_path)

    sift = cv2.SIFT_create(nfeatures=1024)
    kp_q, des_q = sift.detectAndCompute(query_gray, None)
    kp_c, des_c = sift.detectAndCompute(candidate_gray, None)

    if des_q is None or des_c is None or len(kp_q) < 2 or len(kp_c) < 2:
        LOGGER.debug("Skipping %s vs %s: missing descriptors", query_path, candidate_path)
        return 0.0

    good_matches = _select_top_matches(des_q, des_c, ratio=ratio, top_matches=top_matches)
    if len(good_matches) < 4:
        LOGGER.debug("Low match count for %s vs %s: %d; rejecting as non-geometric", query_path, candidate_path, len(good_matches))
        return 0.0

    query_pts = np.float32([kp_q[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    candidate_pts = np.float32([kp_c[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(query_pts, candidate_pts, cv2.RANSAC, 5.0)
    if homography is None or mask is None:
        LOGGER.debug("No homography for %s vs %s; kept %d tentative matches", query_path, candidate_path, len(good_matches))
        return 0.0

    inlier_count = int(mask.sum())
    if inlier_count < 4:
        LOGGER.debug("Too few inliers for %s vs %s: %d", query_path, candidate_path, inlier_count)
        return 0.0

    good_match_ratio = inlier_count / max(len(good_matches), 1)
    coverage = inlier_count / max(min(len(kp_q), len(kp_c)), 1)
    score = min(1.0, 0.75 * good_match_ratio + 0.25 * coverage)
    LOGGER.debug(
        "Candidate %s scored %.4f (inliers=%d/%d, coverage=%.4f)",
        candidate_path,
        score,
        inlier_count,
        len(good_matches),
        coverage,
    )
    return float(score)


def rerank_by_sift_similarity(
    query_id: str,
    initial_ranked: list[str],
    dataset_root: str | Path,
    top_candidates: int = 20,
    top_matches: int = 8,
) -> list[str]:
    dataset_root = Path(dataset_root)
    query_path = dataset_root / query_id

    candidates = [item for item in initial_ranked[: max(1, top_candidates)] if item != query_id]
    LOGGER.info("Reranking query %s over %d candidates with top %d matches for rough homography", query_id, len(candidates), top_matches)

    scored = []
    for item in candidates:
        score = sift_match_score(query_path, dataset_root / item, top_matches=top_matches)
        scored.append((item, score))

    positive = [(item, score) for item, score in scored if score > 0.0]
    if not positive:
        LOGGER.info("No positive geometric rerank for %s; keeping coarse ordering", query_id)
        return [item for item in candidates]

    positive.sort(key=lambda pair: pair[1], reverse=True)
    ranked = [item for item, _ in positive]
    LOGGER.info("Finished rerank for %s: top score=%.4f, top item=%s", query_id, positive[0][1], positive[0][0])
    return ranked
