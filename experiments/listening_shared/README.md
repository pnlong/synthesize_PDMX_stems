# Shared listening test infrastructure

Reusable 0–100 slider UI and scale helpers for formal listening tests under `experiments/ablation_listening/` and `experiments/model_listening/`.

## Rating scale

Listeners rate **content** and **realism** on a continuous **0–100** scale:

| Band | Likert | Content | Realism |
|------|--------|---------|---------|
| 0–20 | 1 | Very different | Very synthetic |
| 20–40 | 2 | Different | Synthetic |
| 40–60 | 3 | Mostly same | Mixed |
| 60–80 | 4 | Same | Realistic |
| 80–100 | 5 | Identical | Very realistic |

**Interaction:** click a band region to snap to its midpoint (10, 30, 50, 70, 90); drag or use the number input for any precise 0–100 value.

## ngrok hosting

**Ablation test (webMUSHRA, recommended):**

```bash
uv run python -m experiments.ablation_listening.generate_webmushra
uv run python -m experiments.ablation_listening.serve_webmushra --host 0.0.0.0 --port 8767
ngrok http 8767
```

Open `http://127.0.0.1:8767/?config=spdmx_ablation.yaml`. Requires PHP. Results: `third_party/webMUSHRA/results/spdmx_ablation/mushra.csv`.

**Legacy custom slider UI:**

```bash
uv run python -m experiments.ablation_listening.serve --host 0.0.0.0 --port 8767
ngrok http 8767
```

Share the `https://….ngrok-free.app` URL with listeners. webMUSHRA saves CSV server-side; the legacy UI saves JSON under `output/responses/`.

**Remote listening tips:**
- Use wired headphones when possible
- Quiet room; set volume once on the calibration clip
- ngrok free URLs change on restart — use a reserved domain for multi-day collection

## Files

| File | Role |
|------|------|
| [`scale.py`](scale.py) | Band labels, thresholds, Likert mapping |
| [`static/slider.js`](static/slider.js) | 0–100 slider widget |
| [`static/slider.css`](static/slider.css) | Slider styles |
