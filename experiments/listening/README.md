# Sweep Listening Test

Structured blinded evaluation for patch and preset sweeps. Separate from the ablation validation viewer (`synthesis.listening.serve` on port 8765).

**Purpose:** identify the best patch-pool variant or SA3 preset variant **per instrument category** (strings, brass, piano, …). Probe stems carry a `category` tag; aggregation reports one winner per category, not one winner overall.

## Quick start

```bash
uv run python -m experiments.listening.serve
```

Open [http://127.0.0.1:8766](http://127.0.0.1:8766) and pick a sweep, or go directly:

- Preset: [http://127.0.0.1:8766/test?type=preset](http://127.0.0.1:8766/test?type=preset)
- Patch: [http://127.0.0.1:8766/test?type=patch](http://127.0.0.1:8766/test?type=patch)

## Protocol

Per probe stem:

1. **Reference** anchor always visible (A1 raw for preset; basic synthesis for patch)
2. **Blinded samples** (A, B, C, …) in randomized order
3. Rate each sample on **content** (1–5) and **realism** (1–5)
4. Complete all stems, then export JSON

Progress is saved in two places:

- **Browser** — each star click writes to `localStorage`
- **Server** — each completed stem (on **Next stem**) writes to `responses/responses_in_progress.json` under the sweep output dir

On reload, the UI merges browser and server progress and resumes at the first incomplete stem. On **Finish**, a timestamped `responses_*.json` is also written for `record_winners` / `aggregate`.

## Options

```bash
uv run python -m experiments.listening.serve --sweep preset --port 8766
uv run python -m experiments.listening.serve --sweep patch
uv run python -m experiments.listening.serve --preset-sweep-dir /path/to/output
```

## Aggregate results

```bash
uv run python -m experiments.listening.aggregate \
  --sweep preset \
  --responses responses_preset_2026-07-12.json \
  --output experiments/preset_sweep/results_notes.md
```

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/{preset\|patch}/meta` | Rubric, stem order, manifest id |
| `GET /api/{preset\|patch}/stems/{stem_id}` | Reference + blinded samples |
| `GET /api/{preset\|patch}/responses/session` | In-progress responses checkpoint |
| `POST /api/{preset\|patch}/responses` | Save responses (`checkpoint: true` → in-progress file) |
| `GET /audio/{preset\|patch}/reference/{stem_id}/{filename}` | Reference audio |
| `GET /audio/{preset\|patch}/variant/{variant_id}/{song_id}/{filename}` | Variant audio |

## Implementation

- [`catalog.py`](catalog.py) — manifest + audio resolution for both sweep types
- [`session.py`](session.py) — blinded ordering and rubrics
- [`aggregate.py`](aggregate.py) — per-category winner selection
- [`serve.py`](serve.py) — stdlib HTTP server (port 8766)
- [`static/`](static/) — listening test UI
