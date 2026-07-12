# Ablation listening viewer

Localhost web UI for browsing the 100-song ablation sample and comparing mixtures and stems across conditions (A1–B2).

## Quick start

```bash
uv run python -m synthesis.listening.serve
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

The server reads ablation outputs from the in-repo symlink [`../ablations_output/`](../ablations_output/) by default (or `{OUTPUT_DIR}/dev/ablations/` on deepfreeze).

## Options

```bash
uv run python -m synthesis.listening.serve --port 8765
uv run python -m synthesis.listening.serve --ablations-dir /path/to/dev/ablations
uv run python -m synthesis.listening.serve --host 127.0.0.1 --port 9000
```

## Conditions

| ID | Label | Directory |
|----|-------|-----------|
| A1 | `basic` | Raw Fluidsynth synthesis |
| A2 | `basic_realify` | SA3-realified stems |
| B1 | `slakh` | Slakh-style patch randomization |
| B2 | `slakh_realify` | Slakh + SA3 realify |

Conditions without generated audio show as **Not generated** or **Audio missing**.

## UI

- **Sidebar:** searchable song list (title, artist, genres)
- **Main panel:** mixture and per-stem grids with four columns (A1–B2)
- **Realify columns:** expandable SA3 caption prompts
- **Prev / Next:** step through the filtered song list

Independent `<audio>` players per cell (no synced playback).

## Preset sweep viewer

Compare A1 raw stems against SA3 preset variants from `experiments/preset_sweep/`:

```bash
uv run python -m synthesis.listening.serve --preset-sweep
```

Open [http://127.0.0.1:8765/preset-sweep](http://127.0.0.1:8765/preset-sweep).

The preset-sweep API is also enabled automatically when `manifest.csv` exists under `experiments/preset_sweep/output/`.

## API (ablation)

| Endpoint | Description |
|----------|-------------|
| `GET /api/conditions` | All four conditions and availability |
| `GET /api/songs` | Song list with metadata |
| `GET /api/songs/{song_id}` | Full detail with stem/mixture URLs |
| `GET /audio/{condition}/{song_id}/{filename}` | Stream audio file |

## API (preset sweep)

| Endpoint | Description |
|----------|-------------|
| `GET /api/preset-sweep/meta` | Variant list and availability |
| `GET /api/preset-sweep/stems` | Probe stem list |
| `GET /api/preset-sweep/stems/{stem_id}` | A1 reference + variant audio URLs |
| `GET /audio/preset-sweep/reference/{stem_id}/{filename}` | Raw A1 stem |
| `GET /audio/preset-sweep/variant/{variant_id}/{song_id}/{filename}` | Sweep variant stem |

`song_id` is the relative path under `data/` (e.g. `7/19/QmPfjDmFbF97N6T6ge4PFFiTQ9VxAFsqLPArCRoLuaTGb1`).

## Implementation

- [`catalog.py`](catalog.py) — loads `data.csv`, `stems.csv`; derives captions in memory; resolves cross-condition paths
- [`serve.py`](serve.py) — stdlib `http.server` (no extra dependencies)
- [`static/`](static/) — vanilla HTML/CSS/JS frontend
