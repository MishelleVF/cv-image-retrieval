from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis

from src.data.loader import class_label, relative_image_id
from src.features.bovw import BoVW, build_vocabulary_from_dataset
from src.features.hog import image_hog
from src.utils.progress import parallel_map


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def fused_descriptor_for_path(
    path: str | Path,
    bovw_model: BoVW,
    dataset_root: str | Path,
) -> np.ndarray:
    path = Path(path)
    bovw = bovw_model.image_histogram(path)
    hog = image_hog(path)
    bovw_norm = _normalize_rows(bovw.reshape(1, -1))[0]
    hog_norm = _normalize_rows(hog.reshape(1, -1))[0]
    return np.concatenate([bovw_norm, hog_norm]).astype(np.float32)


def compute_fused_feature_matrix(
    image_paths: list[str | Path],
    dataset_root: str | Path,
    bovw_model: BoVW,
) -> tuple[np.ndarray, list[str]]:
    def extract(path: str | Path) -> tuple[np.ndarray, str]:
        vector = fused_descriptor_for_path(path, bovw_model, dataset_root)
        image_id = relative_image_id(Path(path), Path(dataset_root))
        return vector, image_id

    results = list(parallel_map(extract, image_paths, "Fused BoVW+HOG feature extraction"))
    if not results:
        raise ValueError("No fused feature vectors were computed.")

    vectors = np.vstack([item for item, _ in results]).astype(np.float32)
    ids = [image_id for _, image_id in results]
    return vectors, ids


class LearnedMetricEmbedding:
    def __init__(self, n_components: int = 64, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca: PCA | None = None
        self.model: NeighborhoodComponentsAnalysis | None = None

    def fit(self, features: np.ndarray, labels: list[str] | np.ndarray) -> "LearnedMetricEmbedding":
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels)
        if features.ndim != 2:
            raise ValueError("Features for metric learning must be a 2D array.")
        if features.shape[0] == 0:
            raise ValueError("No training features available for metric learning.")

        effective_dim = min(self.n_components, features.shape[1], max(1, features.shape[0] - 1))
        self.pca = PCA(n_components=effective_dim, random_state=self.random_state)
        reduced = self.pca.fit_transform(features)

        nca_dim = min(reduced.shape[1], max(1, len(np.unique(labels)) - 1))
        self.model = NeighborhoodComponentsAnalysis(
            n_components=nca_dim,
            init="pca",
            max_iter=500,
            random_state=self.random_state,
        )
        self.model.fit(reduced, labels)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.pca is None or self.model is None:
            raise ValueError("The learned metric has not been fitted yet.")
        features = np.asarray(features, dtype=np.float32)
        reduced = self.pca.transform(features)
        return self.model.transform(reduced).astype(np.float32)

    def fit_from_paths(
        self,
        image_paths: list[str | Path],
        dataset_root: str | Path,
        vocabulary_size: int = 96,
    ) -> "LearnedMetricEmbedding":
        dataset_root = Path(dataset_root)
        bovw_model = build_vocabulary_from_dataset(dataset_root, vocabulary_size=vocabulary_size)
        features, ids = compute_fused_feature_matrix(image_paths, dataset_root, bovw_model)
        labels = [class_label(Path(dataset_root / image_id), dataset_root) for image_id in ids]
        return self.fit(features, labels)


def build_learned_metric(
    image_paths: list[str | Path],
    dataset_root: str | Path,
    vocabulary_size: int = 96,
    n_components: int = 64,
) -> tuple[LearnedMetricEmbedding, np.ndarray, list[str]]:
    embedding = LearnedMetricEmbedding(n_components=n_components)
    dataset_root = Path(dataset_root)
    bovw_model = build_vocabulary_from_dataset(dataset_root, vocabulary_size=vocabulary_size)
    features, ids = compute_fused_feature_matrix(image_paths, dataset_root, bovw_model)
    labels = [class_label(Path(dataset_root / image_id), dataset_root) for image_id in ids]
    embedding.fit(features, labels)
    learned_matrix = embedding.transform(features)
    return embedding, learned_matrix, ids
