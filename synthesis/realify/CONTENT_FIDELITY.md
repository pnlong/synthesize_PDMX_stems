# Content fidelity gate for SA3 realify

Optional post-SA3 safeguard that compares **reference (Fluidsynth) vs realified** audio using **onset alignment** in reference-active regions. When enabled, realify backs off `init_noise_level` on failure and falls back to the unrealified reference if fidelity cannot be achieved.

Complements [`SILENCE.md`](SILENCE.md), which only removes hallucinations during **reference-silent rests**.

## Motivation

Higher `init_noise_level` improves timbral realism but increases **active-region content drift** (extra notes, changed rhythm). Manual listening on probe stems does not always generalize to diverse data.

The content fidelity gate automates a **rhythmic** check: if realified onsets diverge too much from the reference in played sections, lower noise and retry; if still failing at the floor, copy the reference stem unchanged.

## Algorithm (v1)

1. Convert reference and realified stems to mono float32.
2. Build a **reference-active mask** (samples above `REALIFY_SILENCE_THRESHOLD_DB`, optionally dilated).
3. Detect onsets with `librosa.onset.onset_detect` (`backtrack=True`) on both signals.
4. Keep onsets whose sample index falls inside the active mask.
5. **Match onsets** greedily within a tolerance window (default **50 ms**).
6. Compute **F1** from matched / extra / missing counts.
7. Pass if `score >= REALIFY_CONTENT_FIDELITY_THRESHOLD` (default **0.85**, tune manually).

### Adaptive realify loop

For non-chunked stems when `REALIFY_CONTENT_FIDELITY_ENFORCE=True`:

```
noise = preset.init_noise_level
repeat up to MAX_ATTEMPTS:
    generate at noise → silence enforce → score
    if pass: write realified stem
    noise -= NOISE_STEP (default 0.10)
if exhausted: write reference stem (no realify)
```

Chunked long stems (> model buffer): **single pass only** (retry would be too expensive); score is logged but output is not rejected.

## Parameters

Defined in [`shared/config.py`](../../shared/config.py):

| Constant | Default | Purpose |
|----------|---------|---------|
| `REALIFY_CONTENT_FIDELITY_ENFORCE` | `False` | Master switch (off until calibrated) |
| `REALIFY_CONTENT_FIDELITY_THRESHOLD` | `0.85` | Minimum F1 to pass |
| `REALIFY_CONTENT_FIDELITY_NOISE_STEP` | `0.10` | Noise backoff step (phase-1 grid spacing) |
| `REALIFY_CONTENT_FIDELITY_MIN_NOISE` | `0.25` | Floor before reference passthrough |
| `REALIFY_CONTENT_FIDELITY_MAX_ATTEMPTS` | `4` | Cap SA3 retries per stem |
| `REALIFY_CONTENT_FIDELITY_ONSET_TOLERANCE_MS` | `50.0` | Onset matching window |
| `REALIFY_CONTENT_FIDELITY_ACTIVE_MARGIN_MS` | `100.0` | Active-region dilation (reserved for tuning) |

CLI:

```bash
python -m synthesis.realify.realify --content-fidelity-enforce ...
python -m synthesis.synthesize --render-mode basic --realify --content-fidelity-enforce
```

Disable explicitly:

```bash
python -m synthesis.realify.realify --no-content-fidelity-enforce ...
```

## Calibration workflow

Before enabling in production:

1. **Score existing sweep renders:**

```bash
uv run python -m experiments.preset_sweep.score_content_fidelity \
  --sweep-dir experiments/preset_sweep/output/phase1b_noise_audit \
  --responses experiments/preset_sweep/output/phase1b_noise_audit/responses/responses_....json
```

2. Inspect `content_fidelity_scores.csv` and optional `content_fidelity_correlation.csv`.
3. Use printed **suggested threshold** (grid search vs human content ≥ 4.5) as a starting point.
4. Listen to **borderline** cases (scores near threshold ± 0.05) on diverse stems.
5. Lock `REALIFY_CONTENT_FIDELITY_THRESHOLD` in `shared/config.py` and enable the flag only after validation.

## Limitations (v1)

| Detects well | Weak / missed |
|--------------|----------------|
| Extra note onsets, ghost hits, rhythmic additions | Pitch substitutions without new onsets |
| Missing note onsets | Timbral drift with same onsets |
| | Percussion false positives (consider skipping drums initially) |

Future v2 may add chroma correlation in active frames; v1 scope is intentionally onset-only.

## Implementation

| File | Role |
|------|------|
| [`content_fidelity.py`](content_fidelity.py) | Scoring + matching |
| [`realify.py`](realify.py) | Backoff loop in `realify_stem()` |
| [`experiments/preset_sweep/score_content_fidelity.py`](../../experiments/preset_sweep/score_content_fidelity.py) | Offline calibration |
| [`tests/test_content_fidelity.py`](tests/test_content_fidelity.py) | Unit tests |

Unit tests: [`tests/test_content_fidelity.py`](tests/test_content_fidelity.py)
