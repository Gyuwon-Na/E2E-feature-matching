# Evaluation kit for homography / dense matching results

This folder is designed to be dropped into your project as `evaluate/`.
It gives you:

- a single evaluation runner for **alignment error**, **MACE**, **AEPE**, **coverage**, **PCK**
- **rotation-bin analysis**
- **stage comparison** such as `after_transformer -> final`
- a **safe default baseline suite** that prefers models with stable loading paths
- a **generic adapter** for your own `.pth` model without rewriting the evaluator
- a **verify step** so you can catch broken imports/checkpoints before wasting time on a long run

---

## 1. Folder layout

```text
evaluate/
├── README.md
├── requirements.txt
├── run_suite.py
├── verify_env.py
├── prepare_hf_cache.py
├── configs/
├── example_data/
└── evalkit/
```

Your project weights can stay under:

```text
../checkpoints/
```

relative to this `evaluate/` folder.

---

## 2. Quick start

### 2.1 Smoke test the whole pipeline

This runs a synthetic GT-based matcher to verify that the metrics/reporting code works.

```bash
cd evaluate
pip install -r requirements.txt
python run_suite.py --config configs/suite_smoke_test.yaml
```

Outputs will be written to:

```text
../evaluate_outputs/smoke_test/
```

### 2.2 Prepare stable Hugging Face baselines

This downloads and stores the HF models under `../checkpoints/hf/` so later runs do not depend on fresh downloads.

```bash
cd evaluate
python prepare_hf_cache.py --output-root ../checkpoints/hf
```

### 2.3 Check whether everything loads

```bash
cd evaluate
python verify_env.py --config configs/suite_mdpi_safe_template.yaml --load-models
```

This writes a report under:

```text
../evaluate_outputs/mdpi_safe/verify_env/
```

### 2.4 Run the stable baseline suite

```bash
cd evaluate
python run_suite.py --config configs/suite_mdpi_safe_template.yaml
```

---

## 3. What each metric means here

### 3.1 Alignment error (`alignment_error_px`)
Global registration error.

- If the model outputs a homography directly, that homography is used.
- If not, the evaluator fits a homography from predicted correspondences.
- It then warps the **entire source image grid** and computes the mean pixel error against GT correspondences on the valid region.

This is the metric for “image-wide alignment quality”.

### 3.2 MACE (`mace_px`)
Mean corner reprojection error.

- predicted homography vs GT homography
- evaluated on the 4 source-image corners

This is the standard homography accuracy view.

### 3.3 AEPE (`aepe_px`)
Average endpoint error of the model’s **direct correspondences**.

- dense models: uses the predicted dense field directly
- sparse models: rasterizes sparse matches onto the source pixel grid

This reflects local correspondence quality, not the homography fit.

### 3.4 Coverage (`coverage`)
Matching density.

```text
coverage = number of predicted matched source pixels / total number of source pixels
```

This follows your requested definition.

### 3.5 PCK (`pck@1px`, `pck@3px`, `pck@5px`)
Fraction of direct correspondences whose endpoint error is within the threshold.

In this evaluator, PCK is computed on the pixels where the model actually predicts a match.
Coverage is reported separately so density and accuracy are not mixed.

### 3.6 Rotation-bin mean error
The evaluator groups samples by absolute rotation angle and reports the mean of a chosen metric per bin.
Default:

- metric: `alignment_error_px`
- bins: `[0, 5), [5, 15), [15, 30), [30, 60), [60, 90), [90, 180)`

### 3.7 Refinement gain
The evaluator can compare stage outputs such as:

```yaml
stage_pairs:
  - [after_transformer, final]
```

It then writes:

- `stage_deltas.csv`
- `stage_delta_summary.csv`

For error metrics, positive delta means **the later stage reduced error**.
For coverage/PCK, positive delta means **the later stage improved density/accuracy**.

---

## 4. Output files

After a run, the output directory contains:

- `per_sample.csv` — all per-sample metrics for all models/stages
- `summary.csv` — per-model mean/median/std
- `leaderboard.csv` — final-stage ranking
- `rotation_bins.csv` — mean metric by rotation bin
- `stage_deltas.csv` — per-sample stage improvements
- `stage_delta_summary.csv` — mean stage improvements
- `model_reports.csv` — model-level runtime status
- `errors.csv` — any load/inference failures
- `resolved_config.json` — fully resolved config used for the run

---

## 5. Dataset manifest format

The evaluator reads a JSONL manifest.
One line = one sample.

Minimum form:

```json
{
  "sample_id": "sample_001",
  "image0": "path/to/image0.png",
  "image1": "path/to/image1.png",
  "gt": {
    "homography_0to1": "path/to/H.npy",
    "flow01": "path/to/flow01.npy",
    "valid_mask": "path/to/valid_mask.npy",
    "rotation_deg": 10.0
  }
}
```

Supported GT fields:

- `homography_0to1`: inline 3x3 or path to `.npy/.npz/.json`
- `corners0`, `corners1`: if you want GT homography derived from corners
- `flow01`: dense GT correspondence field `[H, W, 2]` in **absolute target pixel coordinates**
- `valid_mask`: `[H, W]` boolean mask
- `rotation_deg`: explicit rotation label for binning

If `flow01` is missing but `homography_0to1` exists, the evaluator synthesizes GT flow from the homography.

---

## 6. Stable default baseline suite

`configs/suite_mdpi_safe_template.yaml` contains four baselines chosen for ease of loading:

- `loftr_kornia`
- `efficientloftr` (HF)
- `lightglue_superpoint` (HF)
- `superglue_outdoor` (HF)

Why these four?

- they are strong related-work baselines for matching
- they have simpler and more stable public loading paths than older repo-specific research code
- two of them load from `transformers`, one from `kornia`, which reduces checkpoint friction

If you want optional denser baselines, see:

```text
configs/suite_dense_optional_template.yaml
```

That file includes:

- `roma_outdoor`
- `dkm_base`

These are intentionally **not** in the default safe suite because their install paths tend to drift more often.

---

## 7. Using your own model

Edit:

```text
configs/user_model_template.yaml
```

The key section is `output_spec`.
You map your model outputs to the evaluator’s unified schema.

### Example: direct dense prediction

If your model returns:

```python
{
    "flow01": ...,          # [H,W,2] absolute target coordinates
    "valid_mask": ...,      # [H,W]
    "confidence": ...       # [H,W]
}
```

then this works:

```yaml
output_spec:
  default:
    flow: flow01
    flow_mode: absolute
    valid_mask: valid_mask
    confidence: confidence
```

### Example: displacement field

If your model returns pixel displacement instead:

```yaml
output_spec:
  default:
    flow: flow01
    flow_mode: displacement
    valid_mask: valid_mask
    confidence: confidence
```

### Example: intermediate transformer stage + final refinement

If your output looks like:

```python
{
    "flow01": final_flow,
    "valid_mask": final_mask,
    "intermediate": {
        "flow01": stage_flow,
        "valid_mask": stage_mask
    }
}
```

then:

```yaml
output_spec:
  default:
    flow: flow01
    flow_mode: absolute
    valid_mask: valid_mask
  stages:
    after_transformer:
      flow: intermediate.flow01
      flow_mode: absolute
      valid_mask: intermediate.valid_mask
```

Now the evaluator can compute the stage delta automatically.

---

## 8. Notes on broken baseline loading

The usual failure points are:

- missing package version
- checkpoint path mismatch
- wrong `state_dict` key
- `module.` prefix mismatch from DDP training
- older research repos pinning outdated Python/CUDA environments

This kit addresses those in two ways:

1. `verify_env.py` checks imports and can attempt actual model loading.
2. `user_torch_module` supports:
   - `checkpoint_key`
   - `strict_load`
   - `strip_prefixes`

so you can recover from common `.pth` loading mismatches without editing source code.

---

## 9. Recommended workflow for your paper results section

1. Run `suite_smoke_test.yaml` first.
2. Prepare HF baselines with `prepare_hf_cache.py`.
3. Run `verify_env.py --load-models`.
4. Replace `dataset.manifest` with your real dataset manifest.
5. Add your own model via `user_model_template.yaml`.
6. Run `run_suite.py`.
7. Use:
   - `summary.csv`
   - `leaderboard.csv`
   - `rotation_bins.csv`
   - `stage_delta_summary.csv`

for the Results section tables and analysis.
