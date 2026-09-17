from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from src.data.loader import iter_image_paths, load_image, relative_image_id
from src.utils.progress import parallel_map

import logging


LOGGER = logging.getLogger(__name__)


class BoVW:
    def __init__(self, vocabulary_size: int = 96, nfeatures: int = 800, max_descriptors_per_image: int = 48):
        self.vocabulary_size = vocabulary_size
        self.nfeatures = nfeatures
        self.max_descriptors_per_image = max_descriptors_per_image
        self.kmeans = None

    def sample_descriptors(
        self,
        image_paths: Sequence[str | Path],
        max_images: int | None = None,
        max_total_descriptors: int | None = None,
        seed: int = 42,
    ) -> np.ndarray:
        samples: List[np.ndarray] = []

        paths = list(image_paths)
        if max_images is not None and max_images < len(paths):
            indices = np.linspace(0, len(paths) - 1, num=max_images, dtype=int)
            paths = [paths[index] for index in np.unique(indices)]
        LOGGER.info("Sampling vocabulary descriptors from %d representative images", len(paths))

        def extract(item: tuple[int, str | Path]) -> np.ndarray | None:
            index, path = item
            image = load_image(path)
            gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (384, 256))
            sift = cv2.SIFT_create(nfeatures=self.nfeatures)
            _, descriptors = sift.detectAndCompute(gray, None)
            if descriptors is None or len(descriptors) == 0:
                return None
            if len(descriptors) > self.max_descriptors_per_image:
                rng = np.random.default_rng(seed + index)
                idx = rng.choice(len(descriptors), size=self.max_descriptors_per_image, replace=False)
                descriptors = descriptors[idx]
            return descriptors.astype(np.float32)

        for descriptors in parallel_map(extract, enumerate(paths), "vocabulary descriptor extraction"):
            if descriptors is not None:
                samples.append(descriptors)

        if max_total_descriptors is not None:
            combined = np.vstack(samples)
            if len(combined) > max_total_descriptors:
                rng = np.random.default_rng(seed)
                indices = rng.choice(len(combined), size=max_total_descriptors, replace=False)
                samples = [combined[np.sort(indices)]]

        if not samples:
            raise ValueError("No descriptors were extracted. Check the image dataset or the file paths.")

        return np.vstack(samples)

    def fit_vocabulary(self, descriptor_matrix: np.ndarray) -> "BoVW":
        if descriptor_matrix.shape[0] < self.vocabulary_size:
            raise ValueError(
                "Not enough descriptors to train a vocabulary of this size. "
                "Reduce vocabulary_size or increase sampled descriptor count."
            )

        self.kmeans = MiniBatchKMeans(
            n_clusters=self.vocabulary_size,
            random_state=42,
            batch_size=256,
            n_init="auto",
            max_iter=200,
        )
        self.kmeans.fit(descriptor_matrix)
        return self

    def image_histogram(self, image_path: str | Path) -> np.ndarray:
        if self.kmeans is None:
            raise ValueError("Vocabulary is not trained yet. Call fit_vocabulary() first.")

        image = load_image(image_path)
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (384, 256))
        sift = cv2.SIFT_create(nfeatures=self.nfeatures)
        _, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.vocabulary_size, dtype=np.float32)

        labels = self.kmeans.predict(descriptors.astype(np.float32))
        hist, _ = np.histogram(labels, bins=np.arange(self.vocabulary_size + 1))
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    def compute_database_histograms(
        self,
        image_paths: Sequence[str | Path],
        dataset_root: str | Path,
        batch_size: int = 256,
    ) -> Tuple[np.ndarray, List[str]]:
        if self.kmeans is None:
            raise ValueError("Vocabulary is not trained yet. Call fit_vocabulary() first.")

        paths = list(image_paths)

        def extract(path: str | Path) -> tuple[np.ndarray, str]:
            hist = self.image_histogram(path)
            return hist, relative_image_id(Path(path), Path(dataset_root))

        results = list(parallel_map(extract, paths, "BoVW database histogram extraction"))
        histograms = [hist for hist, _ in results]
        ids = [image_id for _, image_id in results]

        if not histograms:
            raise ValueError("No image histograms were computed.")

        matrix = np.vstack(histograms).astype(np.float32)
        return matrix, ids


def build_vocabulary_from_dataset(
    dataset_root: str | Path,
    vocabulary_size: int = 96,
    max_images: int | None = None,
) -> BoVW:
    paths = list(iter_image_paths(dataset_root))
    if len(paths) == 0:
        raise ValueError(f"No images were found in dataset root: {dataset_root}")

    extractor = BoVW(vocabulary_size=vocabulary_size)
    samples = extractor.sample_descriptors(paths, max_images=max_images, max_total_descriptors=120_000)
    LOGGER.info("Fitting %d-word visual vocabulary", vocabulary_size)
    extractor.fit_vocabulary(samples)
    LOGGER.info("Visual vocabulary training complete")
    return extractor
