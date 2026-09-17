from __future__ import annotations

import csv
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_DATASET_HANDLE = "skylord/oxbuildings"


def _find_image_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def iter_image_paths(root: str | Path, recursive: bool = True) -> Iterator[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return
    iterator = root_path.rglob("*") if recursive else root_path.iterdir()
    for path in sorted(iterator):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(destination)
        return

    if archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix.lower() in {".tgz", ".gz"}:
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(destination)
        return

    if archive_path.suffix.lower() == ".tar":
        with tarfile.open(archive_path, "r:") as archive:
            archive.extractall(destination)
        return

    raise ValueError(f"Unsupported archive format for extraction: {archive_path}")


def _copy_dataset_contents(source_root: Path, target_root: Path) -> None:
    for child in source_root.iterdir():
        target_path = target_root / child.name
        if child.is_dir():
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(child, target_path, dirs_exist_ok=True)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists():
                shutil.copy2(child, target_path)


def download_kaggle_dataset(dataset_root: str | Path, dataset_handle: str = DEFAULT_DATASET_HANDLE, force_download: bool = False) -> Path:
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "kagglehub is not installed. Install the project dependencies first: pip install kagglehub"
        ) from exc

    if _find_image_files(dataset_root):
        return dataset_root

    download_root = dataset_root.parent / "downloads" / "oxbuildings"
    download_root.mkdir(parents=True, exist_ok=True)

    if force_download:
        for child in download_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    try:
        downloaded_root = Path(
            kagglehub.dataset_download(
                dataset_handle,
                output_dir=str(download_root),
                force_download=force_download,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download dataset '{dataset_handle}' using KaggleHub. "
            "Set CVIR_ALLOW_DOWNLOAD=1 only when credentials and access are available."
        ) from exc

    image_files = _find_image_files(downloaded_root)
    if image_files:
        for source in image_files:
            relative = source.relative_to(downloaded_root)
            target = dataset_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.is_file():
                continue
            shutil.copy2(source, target)
        return dataset_root

    archive_candidates = sorted(download_root.rglob("*"))
    archive_candidates = [path for path in archive_candidates if path.is_file() and path.suffix.lower() in {".zip", ".gz", ".tar", ".tgz"}]
    if archive_candidates:
        archive_path = archive_candidates[0]
        extracted_root = download_root / "extracted"
        extracted_root.mkdir(parents=True, exist_ok=True)
        _extract_archive(archive_path, extracted_root)
        extracted_images = _find_image_files(extracted_root)
        if extracted_images:
            _copy_dataset_contents(extracted_root, dataset_root)
            return dataset_root

    raise FileNotFoundError(
        f"Dataset {dataset_handle} was downloaded to {downloaded_root}, but no image files were found there. "
        "Check Kaggle credentials and archive structure."
    )


def ensure_dataset_ready(
    dataset_root: str | Path,
    ground_truth_path: str | Path | None = None,
    dataset_handle: str = DEFAULT_DATASET_HANDLE,
    allow_download: bool = True,
    force_download: bool = False,
) -> Path:
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    if ground_truth_path is not None:
        gt_path = Path(ground_truth_path)
        gt_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        gt_path = dataset_root / "ground_truth.json"

    if _find_image_files(dataset_root):
        return dataset_root

    env_flag = os.environ.get("CVIR_ALLOW_DOWNLOAD", "0").strip().lower()
    if not allow_download and env_flag not in {"1", "true", "yes", "on"}:
        raise FileNotFoundError(
            f"Dataset is missing at {dataset_root}. Set allow_download=True or CVIR_ALLOW_DOWNLOAD=1 to fetch {dataset_handle} automatically."
        )

    download_kaggle_dataset(dataset_root, dataset_handle=dataset_handle, force_download=force_download)

    for child in dataset_root.iterdir():
        if child.is_dir() and _find_image_files(child):
            dataset_root = child
            break

    if not _find_image_files(dataset_root):
        raise FileNotFoundError(
            f"No images were found in the dataset after download. Checked: {dataset_root}"
        )

    return dataset_root


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def relative_image_id(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def class_label(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    stem = path.stem
    return stem.split("_")[0] if "_" in stem else stem


def _try_match_query_name(query_spec: str, all_paths: Sequence[Path], root: Path) -> str | None:
    normalized = query_spec.replace("\\", "/")
    for path in all_paths:
        rel = relative_image_id(path, root)
        if normalized in {path.name, rel, rel.split("/")[-1]}:
            return rel
    return None


def load_ground_truth(gt_path: str | Path, dataset_root: str | Path) -> Dict[str, List[str]]:
    gt_file = Path(gt_path)
    dataset_root = Path(dataset_root)
    all_paths = list(iter_image_paths(dataset_root))
    if not gt_file.exists():
        return {}

    suffix = gt_file.suffix.lower()

    if suffix == ".json":
        with gt_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        result: Dict[str, List[str]] = {}
        for query_key, relevant_keys in payload.items():
            matched_query = _try_match_query_name(str(query_key), all_paths, dataset_root)
            if matched_query is None:
                matched_query = str(query_key)
            resolved = []
            for item in relevant_keys:
                match = _try_match_query_name(str(item), all_paths, dataset_root)
                resolved.append(match if match is not None else str(item))
            result[matched_query] = resolved
        return result

    if suffix == ".csv":
        result: Dict[str, List[str]] = {}
        with gt_file.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                q = row.get("query") or row.get("image") or row.get("query_path")
                rel = row.get("relevant") or row.get("positive") or row.get("relevant_paths")
                if q is None:
                    continue
                if rel is None:
                    result[str(q)] = []
                    continue
                values = [part.strip() for part in str(rel).split(";") if part.strip()]
                if not values:
                    values = []
                resolved_query = _try_match_query_name(str(q), all_paths, dataset_root) or str(q)
                resolved = []
                for item in values:
                    resolved_item = _try_match_query_name(str(item), all_paths, dataset_root) or str(item)
                    resolved.append(resolved_item)
                result[resolved_query] = resolved
        return result

    raise ValueError(f"Unsupported ground-truth file format: {gt_file}")


def build_class_ground_truth(
    dataset_root: str | Path,
    max_queries: int | None = None,
    max_relevant_per_query: int | None = None,
) -> Dict[str, List[str]]:
    root = Path(dataset_root)
    paths = list(iter_image_paths(root))
    by_label: Dict[str, List[Path]] = {}
    for path in paths:
        by_label.setdefault(class_label(path, root), []).append(path)

    ground_truth: Dict[str, List[str]] = {}
    for label in sorted(by_label):
        members = by_label[label]
        if len(members) < 2:
            continue
        query = members[0]
        relevant_members = members[1:] if max_relevant_per_query is None else members[1 : max_relevant_per_query + 1]
        relevant = [relative_image_id(path, root) for path in relevant_members]
        ground_truth[relative_image_id(query, root)] = relevant
        if max_queries is not None and len(ground_truth) >= max_queries:
            break
    return ground_truth


def build_query_map(dataset_root: str | Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in iter_image_paths(dataset_root):
        mapping[relative_image_id(path, Path(dataset_root))] = path
    return mapping
