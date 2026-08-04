# Ablation listening viewer

Localhost web UI for comparing stems across eight ablation conditions (A1–CB2). Prefer the **10s clips** tree produced by `make_clips` (served automatically when present).

Sweep tuning (patch/preset) uses a separate server: [`experiments/listening/`](../../experiments/listening/) on port **8766**.

## Quick start

```bash
# After all 8 ablations exist:
uv run python -m synthesis.listening.make_clips --clip-seconds 10
uv run python -m synthesis.listening.serve
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

The server prefers `{OUTPUT_DIR}/dev/ablations/clips/` when that tree has condition CSVs; otherwise it reads full ablations under [`../ablations_output/`](../ablations_output/).

By default (`LISTENING_PREFER_SUMMABLE = True` in `shared/config.py`), each logical condition (`basic`, …) **requires** the sibling `{condition}_summable` tree from `synthesis.mix --no-overwrite` (no silent fallback to raw). If any of the eight are missing, the catalog/server errors. Set that constant to `False` to audition raw synthesis stems instead.

```bash
# Mix all 8 ablations into *_summable (keeps raw untouched; no mixture files)
bash synthesis/mix_ablations.sh
# or: JOBS=20 bash synthesis/mix_ablations.sh
```

## Options

```bash
uv run python -m synthesis.listening.serve --port 8765
uv run python -m synthesis.listening.serve --ablations-dir /path/to/dev/ablations/clips
uv run python -m synthesis.listening.serve --host 127.0.0.1 --port 9000
```

## Conditions

| ID | Directory | Notes |
|----|-----------|-------|
| A1 | `basic` | Raw Fluidsynth |
| A2 | `basic_realify` | SA3 on A1 |
| B1 | `slakh` | Slakh recipes |
| B2 | `slakh_realify` | SA3 on B1 |
| CA1 | `ddsp_basic` | Neural DDSP + copies from A1 |
| CA2 | `ddsp_basic_realify` | SA3 neural + copies from A2 |
| CB1 | `ddsp_slakh` | Neural DDSP + copies from B1 |
| CB2 | `ddsp_slakh_realify` | SA3 neural + copies from B2 |

Conditions without generated audio show as **Not generated** or **Audio missing**.

## UI

- **Sidebar:** searchable song list + listening-category filters
- **Main panel:** per-stem grids across conditions (mixtures hidden for clip trees)
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
| `GET /api/conditions` | All eight conditions and availability |
| `GET /api/categories` | Listening categories present in the sample |
| `GET /api/songs` | Song list with metadata |
| `GET /api/songs/{song_id}` | Full detail with stem URLs |
| `GET /audio/{condition}/{song_id}/{filename}` | Stream audio file |

`song_id` is the relative path under `data/` (e.g. `7/19/QmPfjDmFbF97N6T6ge4PFFiTQ9VxAFsqLPArCRoLuaTGb1`).

## Implementation

- [`catalog.py`](catalog.py) — loads `data.csv`, `stems.csv`; derives captions in memory; resolves cross-condition paths
- [`make_clips.py`](make_clips.py) — aligned 10s clips (windows from A1)
- [`serve.py`](serve.py) — stdlib `http.server` (no extra dependencies)
- [`static/`](static/) — vanilla HTML/CSS/JS frontend
