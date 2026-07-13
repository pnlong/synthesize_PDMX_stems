# SA3 preset tuning — step-by-step runbook

Follow these steps in order. Each phase has a **blinded listening test** (you only see Sample A/B/C, not noise levels or prompt names). Winners chain into the next phase.

**Prerequisites**

```bash
# A1 basic ablation probe stems exist
ls synthesis/ablations_output/basic/data/

# GPU with enough VRAM for SA3 medium (see synthesis/realify/README.md)
# Optional: smaller MP3 outputs for listening
export SWEEP_MP3="--mp3"
```

---

## Phase 1 — init_noise_level (fixed prompt)

Compare **5 noise levels** with `prompt_variant: current`, `steps: 8`, `cfg_scale: 1.0`. Pick the best **per category** (piano, strings, wind, …).

### 1.1 Render

```bash
uv run python -m experiments.preset_sweep.sweep \
  --phase phase1_noise $SWEEP_MP3
```

~5 variants × 24 stems = **120** SA3 forwards.  
Output: `experiments/preset_sweep/output/phase1_noise/`

### 1.2 Blinded listening test

```bash
uv run python -m experiments.listening.serve --sweep preset \
  --preset-sweep-dir experiments/preset_sweep/output/phase1_noise
```

Open [http://127.0.0.1:8766/test?type=preset](http://127.0.0.1:8766/test?type=preset)

- Reference = A1 raw (unrealified basic stem)
- Rate each blinded sample: **content** + **realism**
- Phase 1 tip: content scores should spread with noise level; find the realism peak that still passes the content gate
- Export JSON when all stems are complete (Save to server or download)

### 1.3 Record winners

```bash
uv run python -m experiments.preset_sweep.record_winners \
  --phase phase1_noise \
  --responses experiments/preset_sweep/output/phase1_noise/responses/responses_YYYYMMDDTHHMMSSZ.json
```

This writes per-category `variant_id` (e.g. `noise0.45`, `noise0.55`) into [`winners.yaml`](winners.yaml).

Optional: inspect aggregate report:

```bash
uv run python -m experiments.listening.aggregate \
  --sweep preset \
  --sweep-dir experiments/preset_sweep/output/phase1_noise \
  --responses experiments/preset_sweep/output/phase1_noise/responses/responses_....json \
  --output experiments/preset_sweep/results_notes.md
```

---

## Phase 2 — prompt_variant (on phase-1 winners)

Compare **3 prompt formats** using each category's winning noise level from phase 1.

### 2.1 Render

Requires `winners.yaml` phase 1 `completed: true`.

```bash
uv run python -m experiments.preset_sweep.sweep \
  --phase phase2_prompts $SWEEP_MP3
```

~3 variants × 24 stems = **72** SA3 forwards.  
Output: `experiments/preset_sweep/output/phase2_prompts/`

### 2.2 Listen → record

```bash
uv run python -m experiments.listening.serve --sweep preset \
  --preset-sweep-dir experiments/preset_sweep/output/phase2_prompts

uv run python -m experiments.preset_sweep.record_winners \
  --phase phase2_prompts \
  --responses experiments/preset_sweep/output/phase2_prompts/responses/responses_....json
```

Phase 2 tip: apply the **content gate** strictly — pick highest realism among variants that preserve melody/rhythm/timing.

---

## Phase 3 — Diffusion budget (optional)

Compare **3 steps × cfg_scale profiles** using each category's locked noise + prompt from phases 1–2.

Skip this phase if phases 1–2 already sound good; lock will use default `steps: 8`, `cfg_scale: 1.0`.

### 3.1 Render

Requires phase 1 and 2 winners.

```bash
uv run python -m experiments.preset_sweep.sweep \
  --phase phase3_diffusion $SWEEP_MP3
```

~3 variants × 24 stems = **72** SA3 forwards.  
Output: `experiments/preset_sweep/output/phase3_diffusion/`

### 3.2 Listen → record

```bash
uv run python -m experiments.listening.serve --sweep preset \
  --preset-sweep-dir experiments/preset_sweep/output/phase3_diffusion

uv run python -m experiments.preset_sweep.record_winners \
  --phase phase3_diffusion \
  --responses experiments/preset_sweep/output/phase3_diffusion/responses/responses_....json
```

---

## Phase 4 — Lock production config

When phases 1 and 2 are `completed: true` in `winners.yaml` (phase 3 optional):

```bash
uv run python -m experiments.preset_sweep.lock
```

Writes:

- [`winners_locked.yaml`](winners_locked.yaml) — audit trail of per-category presets
- [`synthesis/realify/presets/categories.yaml`](../../synthesis/realify/presets/categories.yaml) — production overrides merged into `categories:`

Example locked entry:

```yaml
categories:
  piano:
    init_noise_level: 0.45
    prompt_variant: minimal
    steps: 8
    cfg_scale: 1.0
```

To preview without touching `categories.yaml`:

```bash
uv run python -m experiments.preset_sweep.lock --skip-categories-yaml
```

---

## Phase 5 — Validate & run A2 ablation

### 5.1 Sanity check (probe stems)

Re-listen on 2–3 held-out stems not in the probe set, or spot-check a few probe stems with the locked config.

### 5.2 Full A2 ablation

```bash
uv run python -m synthesis.synthesize --render-mode basic --realify
```

Output: `dev/ablations/basic_realify/`

### 5.3 Ablation listening comparison

```bash
uv run python -m synthesis.listening.serve
```

Compare A1 (basic) vs A2 (basic_realify) on port **8765**.

---

---

## Quick reference

| Step | Command |
|------|---------|
| Render phase N | `uv run python -m experiments.preset_sweep.sweep --phase <phase>` |
| Listen | `uv run python -m experiments.listening.serve --sweep preset --preset-sweep-dir <phase_dir>` |
| Record winners | `uv run python -m experiments.preset_sweep.record_winners --phase <phase> --responses <json>` |
| Lock production | `uv run python -m experiments.preset_sweep.lock` |
| A2 ablation | `uv run python -m synthesis.synthesize --render-mode basic --realify` |

| Phase | `--phase` value | Variants | Needs prior winners |
|-------|-----------------|----------|---------------------|
| 1 Noise | `phase1_noise` | 5 | — |
| 2 Prompts | `phase2_prompts` | 3 | phase 1 |
| 3 Diffusion | `phase3_diffusion` | 3 | phase 1 + 2 (optional before lock) |

## Files

| File | Purpose |
|------|---------|
| [`grids/`](grids/) | Variant definitions per phase |
| [`winners.yaml`](winners.yaml) | Your per-phase decisions (updated by `record_winners`) |
| [`winners_locked.yaml`](winners_locked.yaml) | Locked production config (written by `lock`) |
| [`preset_grid.yaml`](preset_grid.yaml) | Optional flat factorial for validation |
| [`results_notes.md`](results_notes.md) | Subjective notes template |

## Troubleshooting

- **Phase 2/3 sweep refuses to start** — run `record_winners` for the prior phase first; check `winners.yaml` shows `completed: true`.
- **Missing probe stem** — ensure A1 basic ablation exists for all `probe_stems.yaml` song IDs.
- **GPU OOM** — reduce `--realify-batch-size` or run one stem at a time with `--limit-stems 1`.
- **Listening UI shows "Audio not available"** — sweep didn't finish; check manifest.csv `out_path` files exist.
