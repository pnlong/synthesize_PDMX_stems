# Silence enforcement for SA3 realify

Post-processing pass that removes SA3 **hallucinated content during rests** in stem-separated sPDMX. Applied automatically after each SA3 realify forward pass (pass 2), before stems are written to disk.

## Motivation

Stable Audio 3 timbre transfer (`init_audio` + text caption) often **invents notes or noise** in regions where the Fluidsynth reference stem (A1/basic) is silent—common in sparse MIDI with rests between phrases. This is separate from content-preservation tuning via `init_noise_level` and prompt variants.

The reference stem is trustworthy: it is rendered without neural models, so silent regions are genuinely silent. The realified stem shares the same performance and timing; only timbre differs.

## Algorithm

Processing runs in **four steps** on the full reference and realified waveforms (after SA3 chunk stitch for long stems):

```
1. Scan      — sliding windows over reference and realified at identical positions
2. Margin    — dilate reference-active regions to protect reverb/decay tails
3. Zero      — hard-force hallucination samples to 0
4. Crossfade — linear ramps at force_silent boundaries only
```

### Hallucination criterion

For each overlapping window on the realified stem, if the window is **not silent**, mark only the **reference-silent samples** inside that window as hallucination candidates. This avoids missing rests at window boundaries where partial reference activity would otherwise suppress the whole window.

Sample-level reference activity (with margin dilation) defines the protected tail zone.

| Reference | Realified | Action |
|-----------|-----------|--------|
| Silent | Not silent | Force to zero (unless inside margin) |
| Silent | Silent | No change |
| Active | Anything | Keep realified |

Overlap consensus: if **any** window covering a sample qualifies, that sample is a hallucination candidate.

### Active margin

Reference-active windows are dilated by `REALIFY_SILENCE_ACTIVE_MARGIN_MS` (default 300 ms) on each side. Samples inside this protected zone are never force-zeroed, even if the reference energy falls below threshold—preserving realified reverb tails that extend past quiet Fluidsynth decay.

### Hard zero and boundary crossfade

- **Interior** of force_silent regions → pure `0.0`
- **Boundaries** between force_silent and active → short linear crossfade (`REALIFY_SILENCE_FADE_MS`, default 15 ms) from realified toward zero (entering silence) or zero toward realified (leaving silence)

Crossfade does not apply across the full silent interior.

## Parameters

Defined in [`shared/config.py`](../../shared/config.py):

| Constant | Default | Purpose |
|----------|---------|---------|
| `REALIFY_SILENCE_ENFORCE` | `True` | Master switch |
| `REALIFY_SILENCE_CHUNK_MS` | `500.0` | Analysis window length |
| `REALIFY_SILENCE_OVERLAP_RATIO` | `0.5` | Window overlap fraction |
| `REALIFY_SILENCE_THRESHOLD_DB` | `-60.0` | Peak below this = silent |
| `REALIFY_SILENCE_ACTIVE_MARGIN_MS` | `300.0` | Tail protection after reference activity |
| `REALIFY_SILENCE_FADE_MS` | `15.0` | Boundary crossfade length |

Disable for A/B comparison:

```bash
python -m synthesis.realify.realify --no-silence-enforce ...
python -m synthesis.synthesize --render-mode basic --realify --no-silence-enforce
```

## Integration

Implemented in [`silence.py`](silence.py), called from [`realify.py`](realify.py):

- `realify_stem()` — after single forward or post-stitch on long stems
- `realify_stems_batch()` — per stem after batch trim

Silence detection windows are independent of SA3's ~114s generation chunking in [`chunking.py`](chunking.py).

## Validation

Unit tests: [`tests/test_silence.py`](tests/test_silence.py)

Listening audit probe stems: [`experiments/probe_stems_silence_audit.yaml`](../../experiments/probe_stems_silence_audit.yaml)

Suggested listening checks:

1. **Rests silent?** — no invented material in reference-silent gaps
2. **Tails natural?** — note decay/reverb not clipped at phrase endings
3. **Content preserved?** — active regions unchanged vs realified without enforcement

Auto-metric: peak amplitude in force_silent regions should be ≈ 0.

## Suggested paper language

> After SA3 timbre transfer, we apply a reference-gated silence enforcement step to each stem. We scan overlapping windows over the Fluidsynth reference and realified outputs; wherever the reference is silent but the realified stem contains energy, we classify the region as a hallucination and force those samples to zero. An active-region margin (300 ms) after reference note activity preserves realified decay tails that extend beyond the quiet reference envelope. Short linear crossfades (15 ms) at silence boundaries prevent discontinuity artifacts. This post-hoc step requires no MIDI parsing and uses the non-realified reference as ground truth for rest locations.

## Design rationale

- **Reference-based detection** — Fluidsynth silent regions are ground truth; no sidecar metadata or MIDI replay needed
- **Comparison not blanket gating** — only zero regions where SA3 added content the reference lacks
- **Margin before crossfade** — tail protection defines where zeroing may occur; crossfade smooths the transition into forced-silent regions
- **Hard zero interior** — directly removes invented content; crossfade is boundary-only
