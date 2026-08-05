# Ablation listening test (Test 1) — webMUSHRA

Formal **MUSHRA** comparing **all 8 ablation conditions** (A1–CB2), with separate scores for **content adherence** and **realism**, broken down **per instrument category**.

## Protocol

| | Non-realified | Realified |
|---|---|---|
| **A** `basic` | A1 | A2 |
| **B** `slakh` | B1 | B2 |
| **CA** `ddsp_basic` | CA1 | CA2 |
| **CB** `ddsp_slakh` | CB1 | CB2 |

- **Reference** button = **A1** (`basic`)
- **Blind conditions** = unique A1–CB2 for that excerpt (shuffled; names hidden); one slot matches the Reference when A1 is present
- **Donor-copy dedup:** when ``route_stem`` chooses soundfont for a trial stem, that means the DDSP conditions are donor copies (`ddsp_basic`↔`basic`, `ddsp_slakh`↔`slakh`, and the realify pairs). Those duplicates are **omitted** from the page. Aggregation **auto-assigns** the donor’s score to the omitted condition so factorial tables still cover all 8. Equivalences are stored per trial in `trial_manifest.yaml` under `equivalences`.
- Each stem excerpt is rated **twice** (two pages, same audio):
  - **Content adherence** — melody / rhythm / timing vs Reference
  - **Realism** — natural instrument timbre / artifacts vs Reference
- **Stem trials only** by default: **2 stems × 10 listening categories** = 20 excerpts → **40 rating pages**
  - Categories: piano, drums, strings, wind, voice, mallet, organ, guitar, brass, polyphonic
- Aggregation reports means overall, as a **4×2 factorial**, and **per category × condition × scale**

Use `--stems-per-category 1` for a shorter (~20 page) test if needed.

## Setup

### 1. Clone webMUSHRA (once)

```bash
git clone https://github.com/audiolabs/webMUSHRA.git third_party/webMUSHRA
```

Requires **PHP** for the built-in server and result export.

### 2. Finish all 8 ablation renders

All condition dirs under the ablations root must have matching stem audio for selected songs (see `synthesis/RENDERING_NOTES.md`). Clips prefer stems; mixtures are optional (`--include-mixtures`).

### 3. Prepare trial clips

```bash
uv run python -m experiments.ablation_listening.prepare_clips -j 8
# shorter: --stems-per-category 1
```

Writes 10s clips to [`output/clips/`](output/clips/) and [`trial_manifest.yaml`](trial_manifest.yaml) (including per-trial `equivalences` for donor copies).

To refresh equivalences on an existing manifest without re-cutting:

```bash
uv run python -c "
from experiments.ablation_listening.prepare_clips import annotate_manifest_equivalences
from experiments.ablation_listening.paths import DEFAULT_MANIFEST
annotate_manifest_equivalences(DEFAULT_MANIFEST)
"
```

### 4. Export WAV + generate webMUSHRA config

```bash
uv run python -m experiments.ablation_listening.generate_webmushra
```

- Converts clips → WAV under `third_party/webMUSHRA/stimuli/spdmx_ablation/`
- Writes `third_party/webMUSHRA/configs/spdmx_ablation.yaml` (dual-scale pages: `stem_<cat>_<nn>__content` / `__realism`; omits equivalent duplicates)

### 5. Serve (local or ngrok)

```bash
uv run python -m experiments.ablation_listening.serve_webmushra --host 0.0.0.0 --port 8767
ngrok http 8767
```

Open: `http://127.0.0.1:8767/?config=spdmx_ablation.yaml`

Participant invite draft: [`PARTICIPANT_EMAIL.md`](PARTICIPANT_EMAIL.md)

Results CSV: `third_party/webMUSHRA/results/spdmx_ablation/mushra.csv`

### 6. Aggregate results

```bash
uv run python -m experiments.ablation_listening.aggregate_webmushra \
  --output experiments/ablation_listening/output/results_notes_webmushra.md
```

Reads `trial_manifest.yaml` equivalences by default and expands omitted DDSP conditions before computing means. Produces overall condition means, 4×2 factorial tables, and **per-category** content/realism tables.

## Layout

| File | Role |
|------|------|
| [`conditions.py`](conditions.py) | 8 condition IDs, scales, page-id helpers |
| [`equivalence.py`](equivalence.py) | ``route_stem``-based donor-copy detection + unique stimuli |
| [`prepare_clips.py`](prepare_clips.py) | Select per-category stems + extract 10s clips |
| [`generate_webmushra.py`](generate_webmushra.py) | WAV export + YAML config |
| [`serve_webmushra.py`](serve_webmushra.py) | PHP dev server wrapper |
| [`aggregate_webmushra.py`](aggregate_webmushra.py) | Parse `mushra.csv` → category × scale |
| [`webmushra.py`](webmushra.py) | Config/stimulus helpers |

Informal browsing (no scoring): `uv run python -m synthesis.listening.serve` (port 8765).

Legacy custom UI (`serve.py`) still supports dual sliders but is secondary to webMUSHRA.
