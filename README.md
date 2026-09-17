# Minimal Classical Image Retrieval

This repo is a bounded MVP for the midterm image retrieval project. It generates the data needed for the report/presentation without a large experiment framework.

## Usage

Default, supported run:

```bash
python3 scripts/evaluate.py
```

This project currently uses raw coarse retrieval only. Geometric reranking is disabled in the evaluation pipeline because it degraded mAP on this corpus and was not a valid improvement.

If the dataset is not already present locally, you can allow the project to fetch the Oxford-style archive automatically:

```bash
CVIR_ALLOW_DOWNLOAD=1 python3 scripts/evaluate.py
```

The pipeline logs timestamped progress for dataset preparation, vocabulary training,
BoVW extraction, HOG extraction, ranking, and artifact writing. Image-level feature
work runs concurrently; set `CVIR_WORKERS` to control the number of workers and
`CVIR_LOG_LEVEL` to change verbosity:

```bash
CVIR_ALLOW_DOWNLOAD=1 CVIR_WORKERS=16 CVIR_LOG_LEVEL=INFO python3 scripts/evaluate.py
```

The default baseline is the strongest validated option in this repo: raw coarse ranking with the learned metric embedding when enabled by the representation pipeline.

The run creates:

- `outputs/metrics/metrics.json`
- `outputs/rankings/top5_rankings.json`
- `outputs/representations/ground_truth_used.json`
- `outputs/figures/*.png`
- `outputs/manifest.json`

## What is implemented

- BoVW: SIFT descriptors, sampled MiniBatchKMeans vocabulary, normalized visual-word histograms.
- HOG: fixed-size global descriptor.
- Similarity: cosine and Euclidean ranking.
- Evaluation: Precision@5 and mAP.
- Qualitative output: top-5 image panels.

## Runtime notes

- All available dataset images are indexed and evaluated.
- All evaluable queries are used.
- SIFT descriptors are capped per image.
- Vocabulary training uses descriptors from the full sorted image set, with a deterministic
	representative cap of 120,000 descriptors.
- HOG uses 128x128 grayscale images.

See `PROJECT_MAP.md` for the deliverable map and requirement checklist.
