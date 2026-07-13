# Sweep Listening Test Results

Source: `experiments/patch_sweep/output/phase1_soundfonts/responses/responses_20260713T162053Z.json`
Sweep: patch

## Per instrument class

| Category | Winner variant_id | soundfont_id | fx_profile | pool_id | mean_content | mean_realism | Notes |
|----------|-------------------|--------------|------------|---------|--------------|--------------|-------|
| drums | arachno | arachno | dry |  | 5.0 | 4.33 | |
| mallet | airfont_380 | airfont_380 | dry |  | 5.0 | 4.0 | |
| organ | arachno | arachno | dry |  | 5.0 | 4.67 | |
| piano | airfont_380 | airfont_380 | dry |  | 5.0 | 4.33 | |
| polyphonic | generaluser | generaluser | dry |  | 5.0 | 4.33 | |
| strings | sgm_v2 | sgm_v2 | dry |  | 5.0 | 4.0 | |
| voice | generaluser | generaluser | dry |  | 5.0 | 4.0 | |
| wind | generaluser | generaluser | dry |  | 5.0 | 4.33 | |

## Config suggestions

```
# Suggested production config

# drums: variant arachno, soundfont=arachno, fx=dry
# mallet: variant airfont_380, soundfont=airfont_380, fx=dry
# organ: variant arachno, soundfont=arachno, fx=dry
# piano: variant airfont_380, soundfont=airfont_380, fx=dry
# polyphonic: variant generaluser, soundfont=generaluser, fx=dry
# strings: variant sgm_v2, soundfont=sgm_v2, fx=dry
# voice: variant generaluser, soundfont=generaluser, fx=dry
# wind: variant generaluser, soundfont=generaluser, fx=dry

# After all phases: uv run python -m experiments.patch_sweep.lock
# Or merge per-category winners into experiments/patch_sweep/winners.yaml
```
