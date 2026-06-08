# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An "Organization Object Detector": a user photographs a messy space (desk, shelf, closet),
the system runs open-vocabulary object detection, maps each object to one of 6 tidying
categories, and an LLM produces a discard / relocate / keep decision per object plus a
cleanup guide. Spec docs and inline comments are in Korean; code identifiers are English.

The design has gone through revisions (see [README/v1.md](README/v1.md), [v2.md](README/v2.md),
[v3.md](README/v3.md)). **v3 reframes the project as a CV-course assignment**: the *primary*
deliverable is detector/describe model accuracy (mAP, recall, FP rate); the LLM decision
A/B test is demoted to a secondary "applied demo". Code comments reference these docs by
section (e.g. "v2.md §3", "v3 §7").

## Central experiment: describe OFF vs ON

The core hypothesis is that adding a Florence-2 region *description* + parsed *state signal*
to the LLM prompt improves decision accuracy. This is an A/B comparison threaded through the
whole pipeline via the `describe` boolean:

- `describe=False` → **OFF / control**: detector → category → LLM (label + confidence only).
- `describe=True` → **ON / treatment**: also runs Florence-2 per box and adds two lines
  (`description`, `state`) to each object's LLM prompt entry.

**Fairness invariant** (enforced in [pipeline/llm.py](pipeline/llm.py)): OFF and ON must
differ *only* by those two prompt lines. System instruction, output schema, model, and
`temperature=0` are identical on both arms. Do not let other differences leak between arms.

## Two entry points

1. **[baseline.py](baseline.py)** — the original v1 monolith. Self-contained: detect →
   annotate → CSV → build prompt → call OpenAI. Run with `python baseline.py` (hardcoded to
   `test_images/test1.jpeg`). It duplicates `OBJECT_CLASSES` and the category map rather than
   importing them. Treat it as a frozen reference, not the active codebase.

2. **[app/](app/) + [pipeline/](pipeline/) + [models/](models/)** — the v2/v3 modular system.
   This is where active work happens.

## Architecture (modular path)

```
app/main.py          FastAPI: POST /analyze, GET /health. Loads detector + Florence-2 once
                     at startup (lifespan), stores on app.state.
pipeline/orchestrator.py  run_pipeline(): EXIF-correct → detect → (describe) → category
                     → LLM → annotate. The single source of truth for the request flow.
pipeline/category.py classify_item(label) → (category_key, korean_name, recommendation).
                     6 fixed categories. Returns "unknown"/"미분류" for unmapped labels.
pipeline/attributes.py  parse_state_signal(desc): keyword-match a Florence-2 caption to one
                     of trash_signal / needs_wash / tidy_target / in_use (or None).
pipeline/llm.py      build_prompt(objects, describe) + ask_llm(). Returns None (not error)
                     when OPENAI_API_KEY is absent — pipeline still completes without decisions.
models/detector.py   Detector wrappers behind get_detector(name). See contract below.
models/describer.py  Florence2Describer.describe_region(pil_img, bbox) via <MORE_DETAILED_CAPTION>.
app/schemas.py       Pydantic response models (DetectedObject, Guide, AnalyzeResponse).
app/config.py        Env-var config + auto_device() (cuda > mps > cpu).
```

### Detector contract

Every detector class returns `list[dict]` where each dict is
`{"label": str, "confidence": float, "bbox": [x1, y1, x2, y2]}` (pixel xyxy). Detectors are
registered in `_REGISTRY` in [models/detector.py](models/detector.py) and constructed via
`get_detector(name, **kwargs)`. Current names: `yolo-world` (default, `yolov8s-world.pt`),
`yoloe`. Both are open-vocabulary and set their class list from `OBJECT_CLASSES`. To add a
detector, implement a class with `.detect(image_path)` returning that contract and add it to
`_REGISTRY`. The orchestrator and all eval scripts depend only on this contract.

### Graceful degradation

Two pieces are optional and fail soft: Florence-2 (if it won't load, `describe=ON` is
silently disabled — see startup log) and the OpenAI key (if absent, objects/categories are
still returned, just without decisions). Don't turn these into hard failures.

## Eval subsystem ([eval/](eval/) — fully separate from runtime)

The eval code never imports the FastAPI app; it imports only `models.detector` and
`eval.metrics`/`eval.stats`. Each script inserts the repo root into `sys.path`, so run them
as `python eval/<script>.py` from the repo root.

- **[eval/metrics.py](eval/metrics.py)** — library: `compute_map` (mAP@0.5, per-class,
  small-object), `compute_map_range` (mAP@0.5:0.95), `compute_cost_weighted_acc`. The cost
  matrix weights *irreversible* errors highest (predicting discard on a GT=keep object = cost 4).
- **[eval/compare_detectors.py](eval/compare_detectors.py)** — primary CV metric: compares
  detectors zero-shot. With `--gt` gives mAP/recall/precision/small-object; without it does
  qualitative (annotated-image) comparison only.
- **[eval/eval_coco.py](eval/eval_coco.py)** — COCO-2017 val zero-shot mAP via FiftyOne +
  pycocotools. Maps our labels → COCO-80 via `OUR_TO_COCO`. Requires `pip install fiftyone`.
- **[eval/audit.py](eval/audit.py)** — describe-quality audit: stratified-sample a JSONL
  trace into a human-scoring todo, then score with Wilson CIs. Oversamples `trash_signal`
  (the decision-flipping stratum).
- **[eval/make_todo.py](eval/make_todo.py)** — labeling workflow: detector output →
  `.todo.json` draft, human fills `decision`/`_fp`/`missed_objects`, then `--finalize`
  converts `.todo.json` → GT `.json`.
- **[eval/stats.py](eval/stats.py)** — Wilson CI, Cohen's kappa (target κ≥0.7), proportion z-test.

### Ground-truth format

GT lives in [eval/ground_truth/](eval/ground_truth/) as `<image_stem>.json`:
`{"image_id", "objects": [{"label", "bbox": [x1,y1,x2,y2], "decision"}]}`. `decision` is one
of discard/relocate/keep, or **`"invalid"`** to mark a detector false-positive — `invalid`
objects are excluded from detection-metric computation everywhere. See
[eval/labeling_guide.md](eval/labeling_guide.md) and `eval/ground_truth/EXAMPLE_FORMAT.json`.

## Common commands

```bash
pip install -r requirements.txt          # fiftyone for eval_coco.py is NOT in requirements

# Run the API (loads models at startup)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
#   POST /analyze  form fields: image (file), describe (bool), detector (str)
#   GET  /health   reports detector/device/llm status

python baseline.py                        # v1 monolith on test_images/test1.jpeg

# Detector comparison (the primary CV metric)
python eval/compare_detectors.py --images eval/data/dev --gt eval/ground_truth \
    --detectors yolo-world yoloe --output outputs/detector_comparison
python eval/eval_coco.py --detector yolo-world --max-samples 200   # smoke test

# Labeling: draft → (human fills decisions) → finalize
python eval/make_todo.py --images eval/data/dev --output eval/ground_truth
python eval/make_todo.py --finalize --gt eval/ground_truth

# Describe audit (after producing a describe=ON JSONL trace)
python eval/audit.py sample --trace <trace>.jsonl --n 120
python eval/audit.py score  --todo eval/audit_todo.json
```

There is **no test suite, linter, or build step** configured for the Python side.

### iOS app ([ios/OrganizeDetector/](ios/OrganizeDetector/))

6 Swift files, iOS 17+, no external dependencies (PhotosUI + SwiftUI only):

| File | Role |
|---|---|
| `OrganizeDetectorApp.swift` | `@main` entry, NavigationStack root |
| `Models.swift` | `AnalyzeResponse`, `DetectedObject`, `Guide` — Codable + SwiftUI display helpers |
| `APIService.swift` | Multipart POST `/analyze`, EXIF normalization before upload |
| `HomeView.swift` | PhotosPicker, server URL + describe toggle (AppStorage), analyze → sheet |
| `ResultView.swift` | Image + overlay, object chips, detail card, GuideView, GuideSection |
| `BBoxOverlayView.swift` | Canvas bbox drawing + transparent tap targets |

**Xcode setup** (not tracked in git — create manually):
1. New Project → iOS App, Product Name `OrganizeDetector`, Swift, SwiftUI
2. Minimum Deployments: **iOS 17.0**
3. Info / target settings: add `NSPhotoLibraryUsageDescription`
4. App Transport Security → Allow Arbitrary Loads = YES (demo only; scope to Tailscale IP for release)

## Configuration (environment variables)

`MODEL_PATH` (yolov8s-world.pt), `DETECTOR` (yolo-world), `CONF` (0.08 — deliberately low to
catch small/cluttered objects), `IMGSZ` (960), `STATIC_DIR` (outputs/static), `OPENAI_MODEL`,
`OPENAI_API_KEY`. Device is auto-detected (cuda > mps > cpu).

## Gotchas

- **`OBJECT_CLASSES` and the category map are duplicated** in [baseline.py](baseline.py) and
  in [models/detector.py](models/detector.py) / [pipeline/category.py](pipeline/category.py).
  When editing the active path, change the latter; baseline.py is the frozen v1 copy.
- **`OPENAI_MODEL` defaults disagree**: `gpt-5.4-mini` in baseline.py vs `gpt-4o-mini` in
  app/config.py and pipeline/llm.py. The modular path is authoritative.
- **Model weights are tracked via Git LFS** (`*.pt` in [.gitattributes](.gitattributes)).
  `yolov8s-world.pt` and `weights/clip/ViT-B-32.pt` show as modified in git status because
  YOLO-World may re-download/re-touch them; avoid committing weight churn.
- The orchestrator writes a temp file before detection because detectors take a *file path*,
  not an array. It cleans the temp file up in a `finally`.
