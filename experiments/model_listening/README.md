# Model listening test (Test 2)

Compare **Stable Audio Open** generation outputs from models trained on different datasets (e.g. SLAKH2100 vs winning sPDMX variant).

## Status

Scaffold only — populate after Test 1 picks a render mode and SAO training runs complete.

## Steps (after Test 1)

1. Train SAO from scratch on **SLAKH2100** (baseline).
2. Train SAO on **sPDMX** rendered with the Test 1 winner (e.g. B2).
3. Generate held-out prompts with both checkpoints.
4. Fill in [`trial_manifest.yaml`](trial_manifest.yaml) with model output paths.
5. Run clip prep (reuse ablation pattern or point manifest at full generations).
6. Serve and collect ratings with the same 0–100 UI as Test 1.

## Serve (once manifest exists)

```bash
uv run python -m experiments.model_listening.serve --host 0.0.0.0 --port 8768
ngrok http 8768
```

## Suggested design

| Arm | Training data | Question |
|-----|---------------|----------|
| Baseline | SLAKH2100 | Reference SAO behavior |
| sPDMX | Winner from Test 1 | Does better data → better generations? |
| Optional control | Runner-up ablation | Realify-only or slakh-only ablation |

Use the same **content + realism** rubrics. Test 2 likely uses fewer trials (model outputs on fixed held-out text prompts) than Test 1.

## Files

Same layout as [`../ablation_listening/`](../ablation_listening/): `catalog.py`, `serve.py`, `aggregate.py`, `static/`, `output/responses/`.

Shared UI: [`../listening_shared/`](../listening_shared/).
