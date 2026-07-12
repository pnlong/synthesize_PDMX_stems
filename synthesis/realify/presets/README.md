# Realify presets

Per-category SA3 parameters for production realify (`synthesis.realify.realify`).

## Layout

| File | Purpose |
|------|---------|
| [`categories.yaml`](categories.yaml) | `default` preset, per-category overrides, and `routing` rules |

## Defaults

All stems start from `default` (currently `init_noise_level: 0.45`, `prompt_variant: current`).

Category overrides in `categories:` are merged on top when a stem matches a `routing` rule.

## Routing

Rules are evaluated in order; first match wins. Each rule can use:

- `is_drum: true` — drum tracks
- `name_keywords: [...]` — substring match on stem `name` (case-insensitive)

Unmatched stems use `default` only.

## Tuning workflow

1. Run [`experiments/preset_sweep/`](../../../experiments/preset_sweep/)
2. Listen via `uv run python -m synthesis.listening.serve --preset-sweep`
3. Add overrides under `categories:` for categories that need different settings

Example:

```yaml
categories:
  drums:
    init_noise_level: 0.55
    prompt_variant: minimal
  voice:
    init_noise_level: 0.35
    prompt_variant: preservation
```
