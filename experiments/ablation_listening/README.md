# Ablation listening test (Test 1)

Formal listening test comparing **all 8 ablation conditions** (A1–CB2), with **content adherence** and **realism** on the **same page** per excerpt, broken down **per instrument category**.

## Protocol

| | Non-realified | Realified |
|---|---|---|
| **A** `basic` | A1 | A2 |
| **B** `slakh` | B1 | B2 |
| **CA** `ddsp_basic` | CA1 | CA2 |
| **CB** `ddsp_slakh` | CB1 | CB2 |

- **One page = one clip** (~20 stem trials by default)
- **Visible Reference** = A1 (`basic`) — play-only
- **Shared loop player** (webMUSHRA-style): one waveform, looping 10s clip; Play/Pause on Reference and each blind sample continues from the same playhead
- **Blind samples** = unique conditions for that excerpt (shuffled; up to 8), each with **Content** and **Realism** sliders (0–100). One blind sample may match the Reference.
- **Donor-copy dedup:** when `route_stem` chooses soundfont, DDSP↔donor duplicates are omitted from the page. Aggregation auto-assigns the donor’s scores so factorial tables still cover all 8. See `equivalences` in [`trial_manifest.yaml`](trial_manifest.yaml).
- Default trials: **2 stems × 10 listening categories** = 20 excerpts
  - Categories: piano, drums, strings, wind, voice, mallet, organ, guitar, brass, polyphonic
- Clip windows require **≥70% active material** (short-hop RMS) and prefer the densest 10s stretch on the reference stem, so silence-then-burst clips are skipped.
Use `--stems-per-category 1` when preparing clips for a shorter test.

## Setup

### 1. Finish all 8 ablation renders

All condition dirs under the ablations root must have matching stem audio (see `synthesis/RENDERING_NOTES.md`).

### 2. Prepare trial clips

```bash
uv run python -m experiments.ablation_listening.prepare_clips -j 8
# shorter: --stems-per-category 1
```

Writes 10s clips to [`output/clips/`](output/clips/) and [`trial_manifest.yaml`](trial_manifest.yaml).

Refresh equivalences without re-cutting:

```bash
uv run python -c "
from experiments.ablation_listening.prepare_clips import annotate_manifest_equivalences
from experiments.ablation_listening.paths import DEFAULT_MANIFEST
annotate_manifest_equivalences(DEFAULT_MANIFEST)
"
```

### 3. Serve

```bash
uv run python -m experiments.ablation_listening.serve --host 0.0.0.0 --port 8767
ngrok http 8767
```

Open: `http://127.0.0.1:8767/test`

Participant invite draft: [`PARTICIPANT_EMAIL.md`](PARTICIPANT_EMAIL.md)

Responses: `experiments/ablation_listening/output/responses/`

Each browser gets a random listener ID (stored in `localStorage`) so concurrent participants do not collide. The same browser can close and reopen mid-test to resume.

### 4. Aggregate results

```bash
uv run python -m experiments.ablation_listening.aggregate \
  --output experiments/ablation_listening/output/results_notes.md \
  --plots-dir
```

By default this reads `output/responses/` and **only** finished exports (`responses_<listener>_<timestamp>.json`). In-progress checkpoints (`responses_in_progress_*.json`) are ignored.

`--plots-dir` writes bar charts to `output/plots/` (PDF, transparent background):
- `overview.pdf` — overall content / realism / combined (x-axis A/B/CA/CB; left=synthetic, right=realified; ★ = winner)
- `overview_by_category.pdf` — all categories in one figure
- `by_category/<category>.pdf` — three-panel content|realism|combined per category
- `category_winners.csv` — per-category winner on
  $\mathrm{combined}=(\mathrm{content}/100)\times\mathrm{realism}$ (+ margin vs 2nd / DDSP donor)

Plots only:

```bash
uv run python -m experiments.ablation_listening.plot_results
```

## Layout

| File | Role |
|------|------|
| [`conditions.py`](conditions.py) | 8 condition IDs, scales |
| [`equivalence.py`](equivalence.py) | `route_stem` donor-copy detection + score expand |
| [`prepare_clips.py`](prepare_clips.py) | Select per-category stems + extract 10s clips |
| [`catalog.py`](catalog.py) / [`session.py`](session.py) | Trial API + blinded ordering |
| [`serve.py`](serve.py) | Dual-slider listening UI server |
| [`static/`](static/) | Intro + test pages |
| [`aggregate.py`](aggregate.py) | Parse response JSON → condition / factorial tables |

Informal browsing (no scoring): `uv run python -m synthesis.listening.serve` (port 8765).
