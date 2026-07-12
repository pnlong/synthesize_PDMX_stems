# Experiments

Offline parameter sweeps and tuning runs. Experiment **code and config** live here; large **audio outputs** live on deepfreeze with in-repo symlinks.

## Layout

```
experiments/
├── README.md
└── preset_sweep/          # SA3 init_noise_level + prompt tuning
    ├── README.md
    ├── probe_stems.yaml
    ├── preset_grid.yaml
    ├── sweep.py
    ├── results_notes.md
    └── output/            # symlink → {OUTPUT_DIR}/dev/experiments/preset_sweep/
```

## Setup

After clone, create dev artifact symlinks (includes preset sweep output):

```bash
uv run python -m shared.setup_symlinks
```

## Preset sweep

See [`preset_sweep/README.md`](preset_sweep/README.md).
