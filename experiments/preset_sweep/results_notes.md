# Preset Sweep Results

Record subjective winners after listening via the preset-sweep viewer.

## Per instrument class

| Category | Winner variant_id | init_noise_level | prompt_variant | Notes |
|----------|-------------------|------------------|----------------|-------|
| piano | | | | |
| drums | | | | |
| strings | | | | |
| wind | | | | |
| voice | | | | |
| mallet | | | | |
| organ | | | | |
| polyphonic | | | | |

## Next steps

1. Re-listen on 3–5 held-out stems (not in probe set)
2. Update `synthesis/realify/presets/categories.yaml` with winners
3. Run full A2 ablation: `uv run python -m synthesis.synthesize --render-mode basic --realify`
