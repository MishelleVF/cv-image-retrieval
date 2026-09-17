# Project Map

## MVP goal
Produce the minimum reproducible data needed for the image retrieval deliverable:

- top-5 retrieval rankings
- Precision@5 and mAP metrics
- comparison of two classical feature representations
- comparison of two similarity metrics
- qualitative top-5 figures for the report

Run everything with:

```bash
python3 scripts/evaluate.py
```

## Main files
- `scripts/evaluate.py`: complete bounded pipeline and artifact generator.
- `src/data/loader.py`: image loading, dataset readiness, ground-truth loading, and fallback class-label ground truth.
- `src/features/bovw.py`: SIFT descriptors plus MiniBatchKMeans visual vocabulary and BoVW histograms.
- `src/features/hog.py`: fixed-size HOG vectors.
- `src/retrieval/search.py`: cosine and Euclidean ranking.
- `src/evaluation/metrics.py`: Precision@k, AP, and mAP.
- `config.yaml`: lightweight project settings.

## Generated deliverable data
- `outputs/metrics/metrics.json`: quantitative results for BoVW and HOG.
- `outputs/rankings/top5_rankings.json`: top-5 rankings per query.
- `outputs/representations/ground_truth_used.json`: exact query/relevance sets used in the run.
- `outputs/figures/*.png`: qualitative top-5 panels.
- `outputs/manifest.json`: index of generated deliverable artifacts.

## Requirement coverage
- Classical methods only: SIFT, BoVW, HOG, k-means, cosine distance, Euclidean distance.
- At least two representations: BoVW and HOG.
- Visual vocabulary: built with bounded MiniBatchKMeans.
- Fixed-length vectors: BoVW histograms and HOG vectors.
- Retrieval output: top-5 JSON rankings and figure panels.
- Quantitative evaluation: Precision@5 and mAP.
- Parameter/similarity comparison: BoVW vs HOG and cosine vs Euclidean.

## Runtime notes
- Evaluation indexes all available dataset images.
- Evaluation uses all evaluable queries.
- SIFT descriptors are capped per image and extracted from resized grayscale images.
- HOG vectors are extracted from 128x128 grayscale images.
- No dataset or query subsampling is applied unless the code is explicitly changed to do so.
