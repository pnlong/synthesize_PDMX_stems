# Ablation listening test (Test 1) — webMUSHRA

Formal **MUSHRA** listening test comparing **A1 / A2 / B1 / B2** ablation clips (~20 min; 4 mixture + 6 stem trials).

## Protocol (hidden reference)

Each trial uses ITU-R BS.1534-style **MUSHRA** (0–100 Basic Audio Quality):

- **Reference** button = **A1** (`basic` Fluidsynth synthesis)
- **Four blind conditions** = A1, A2, B1, B2 (shuffled; condition names hidden)
- One blind slot is **identical** to the Reference (hidden reference); rate how close each blind condition sounds to the Reference

## Setup

### 1. Clone webMUSHRA (once)

```bash
git clone https://github.com/audiolabs/webMUSHRA.git third_party/webMUSHRA
```

Requires **PHP** for the built-in server and result export.

### 2. Prepare ablation clips

```bash
uv run python -m experiments.ablation_listening.prepare_clips
```

Writes 10s MP3 clips to [`output/clips/`](output/clips/) and [`trial_manifest.yaml`](trial_manifest.yaml).

Requires all four ablation dirs on deepfreeze with matching audio for selected songs. Finish A2 realify if needed:

```bash
uv run python -m synthesis.synthesize --render-mode basic --realify --mp3
```

### 3. Export WAV + generate webMUSHRA config

```bash
uv run python -m experiments.ablation_listening.generate_webmushra
```

- Converts clips → WAV under `third_party/webMUSHRA/stimuli/spdmx_ablation/`
- Writes `third_party/webMUSHRA/configs/spdmx_ablation.yaml`

### 4. Serve (local or ngrok)

```bash
uv run python -m experiments.ablation_listening.serve_webmushra --host 0.0.0.0 --port 8767
ngrok http 8767
```

Open: `http://127.0.0.1:8767/?config=spdmx_ablation.yaml`

Results CSV: `third_party/webMUSHRA/results/spdmx_ablation/mushra.csv`

### 5. Aggregate results

```bash
uv run python -m experiments.ablation_listening.aggregate_webmushra \
  --output experiments/ablation_listening/output/results_notes_webmushra.md
```

## Legacy custom UI

The earlier custom slider UI (`serve.py` on port 8767) is still in [`static/`](static/) but **webMUSHRA is the primary test**. Do not run both servers on the same port.

## Layout

| File | Role |
|------|------|
| [`prepare_clips.py`](prepare_clips.py) | Select trials + extract 10s clips from ablations |
| [`generate_webmushra.py`](generate_webmushra.py) | WAV export + YAML config |
| [`serve_webmushra.py`](serve_webmushra.py) | PHP dev server wrapper |
| [`aggregate_webmushra.py`](aggregate_webmushra.py) | Parse `mushra.csv` |
| [`webmushra.py`](webmushra.py) | Config/stimulus helpers |

Informal browsing (no scoring): `uv run python -m synthesis.listening.serve` (port 8765).
