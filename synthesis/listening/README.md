# Ablation listening viewer

Localhost web UI for browsing the 100-song ablation sample and comparing mixtures and stems across conditions (A1–B2).

Sweep tuning (patch/preset) uses a separate server: [`experiments/listening/`](../../experiments/listening/) on port **8766**.

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

## Sweep listening tests

Structured evaluation for patch and preset sweeps:

```bash
uv run python -m experiments.listening.serve
```

Open [http://127.0.0.1:8766](http://127.0.0.1:8766).

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/conditions` | All four conditions and availability |
| `GET /api/songs` | Song list with metadata |
| `GET /api/songs/{song_id}` | Full detail with stem/mixture URLs |
| `GET /audio/{condition}/{song_id}/{filename}` | Stream audio file |

`song_id` is the relative path under `data/` (e.g. `7/19/QmPfjDmFbF97N6T6ge4PFFiTQ9VxAFsqLPArCRoLuaTGb1`).

## Implementation

- [`catalog.py`](catalog.py) — loads `data.csv`, `stems.csv`; derives captions in memory; resolves cross-condition paths
- [`serve.py`](serve.py) — stdlib `http.server` (no extra dependencies)
- [`static/`](static/) — vanilla HTML/CSS/JS frontend
