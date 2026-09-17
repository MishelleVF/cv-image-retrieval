from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from skimage.feature import hog

from src.data.loader import load_image, relative_image_id
from src.utils.progress import parallel_map


def image_hog(path: str | Path, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    image = load_image(path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, size)
    vector = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return vector.astype(np.float32)


def compute_hog_matrix(image_paths: Sequence[str | Path], dataset_root: str | Path) -> tuple[np.ndarray, list[str]]:
    paths = list(image_paths)
    vectors = list(parallel_map(image_hog, paths, "HOG extraction"))
    ids = [relative_image_id(Path(path), Path(dataset_root)) for path in paths]
    if not vectors:
        raise ValueError("No HOG vectors were computed.")
    return np.vstack(vectors).astype(np.float32), ids
