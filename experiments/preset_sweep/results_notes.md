# Preset Sweep Results

Record subjective winners after each phased listening test (`experiments.listening.serve`). Methodology: [`../TUNING.md`](../TUNING.md). **Runbook:** [`GUIDE.md`](GUIDE.md).

**Phased parameters:** phase 1 `init_noise_level` → phase 2 `prompt_variant` → optional phase 3 `steps`/`cfg_scale`. **Lock target:** `synthesis/realify/presets/categories.yaml`.

## Phase 1 — init_noise_level

| Category | Winner variant_id | init_noise_level | Notes |
|----------|-------------------|------------------|-------|
| piano | | | |
| drums | | | |
| strings | | | |
| wind | | | |
| voice | | | |
| mallet | | | |
| organ | | | |
| polyphonic | | | |

## Phase 2 — prompt_variant

| Category | Winner variant_id | prompt_variant | Notes |
|----------|-------------------|----------------|-------|
| piano | | | |
| drums | | | |
| strings | | | |
| wind | | | |
| voice | | | |
| mallet | | | |
| organ | | | |
| polyphonic | | | |

## Phase 3 — diffusion (optional)

| Category | Winner variant_id | steps | cfg_scale | Notes |
|----------|-------------------|-------|-----------|-------|
| piano | | | | |
| drums | | | | |
| strings | | | | |
| wind | | | | |
| voice | | | | |
| mallet | | | | |
| organ | | | |
| polyphonic | | | | |

## Next steps

1. Re-listen on 3–5 held-out stems (not in probe set)
2. `uv run python -m experiments.preset_sweep.lock`
3. Run full A2 ablation: `uv run python -m synthesis.synthesize --render-mode basic --realify`
