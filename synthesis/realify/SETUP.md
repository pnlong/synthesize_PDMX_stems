# Environment setup (uv + Stable Audio 3)

Use [uv](https://docs.astral.sh/uv/) for both spdmx and SA3. We recommend **Python 3.10** (see `.python-version` at repo root).

## System dependencies

**Synthesis** (fluidsynth MIDI rendering):

```bash
sudo apt install fluidsynth   # or equivalent on your OS
```

Soundfont path is set in `shared/config.py` (`SOUNDFONT_PATH`).

**Realify** (SA3 medium): NVIDIA GPU with CUDA, plus flash-attention (see below).

---

## 1. spdmx environment

From the repo root:

```bash
uv sync --group dev
```

This creates `.venv/` with synthesis, analysis, and test dependencies.

Verify:

```bash
uv run python -c "import mido, synthesis.audio; print('spdmx ok')"
uv run PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest -q
```

---

## 2. Stable Audio 3 submodule

```bash
git submodule update --init synthesis/realify/stable-audio-3
```

The submodule must contain upstream `pyproject.toml` (not just this stub README). If empty, run:

```bash
git submodule add https://github.com/Stability-AI/stable-audio-3.git synthesis/realify/stable-audio-3
```

---

## 3. Install SA3 into the spdmx venv

Still from repo root:

```bash
uv sync --extra realify --group dev
```

This installs `stable-audio-3` from the submodule path (editable).

**CUDA / PyTorch:** On Linux x86_64, `pyproject.toml` pulls `torch`/`torchaudio` 2.7.1 from the PyTorch cu126 index. For a different CUDA version, see the [SA3 README](https://github.com/Stability-AI/stable-audio-3/blob/main/README.md#cuda-version).

---

## 4. Flash Attention (required for SA3 medium)

SA3 **medium** (our default from song-length analysis) needs flash-attention. Install a wheel matching your CUDA, PyTorch, and Python versions — see [SA3 flash-attention docs](https://github.com/Stability-AI/stable-audio-3/blob/main/README.md#flash-attention).

Example (CUDA 12.6, torch 2.7, Python 3.10):

```bash
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu126torch2.7-cp310-cp310-linux_x86_64.whl
```

Verify:

```bash
uv run python -c "import flash_attn; print('flash-attn ok')"
```

Use `uv sync --inexact --extra realify --group dev` after adding flash-attn so `uv sync` does not remove it.

---

## 5. Hugging Face model access

SA3 weights download from Hugging Face on first run. Log in if needed:

```bash
uv pip install huggingface-hub
huggingface-cli login
```

Models (see song-length report — we use **medium**):

- [stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium)

---

## 6. Smoke test realify

After A1 stems exist (or any dataset dir with `stem_*.flac` + `captions.csv`):

```bash
uv run python -m synthesis.synthesize --render-mode basic --realify --realify-limit 2
```

Or standalone:

```bash
uv run python -m synthesis.realify.captions.generate --dataset_dir .../dev/ablations/basic
uv run python -m synthesis.realify.realify \
  --source-dir .../dev/ablations/basic \
  --output-dir .../dev/ablations/basic_realify \
  --limit 2
```

Preset exploration notebook: `synthesis/realify/tests/explore_presets.ipynb`

---

## Optional: SA3-only venv in submodule

Upstream SA3 also supports a standalone env:

```bash
cd synthesis/realify/stable-audio-3
uv sync
```

Use this for Gradio UI (`uv run python run_gradio.py --model medium`) or upstream tests. For spdmx realify, prefer the **single root venv** with `--extra realify` so `python -m synthesis.synthesize --realify` works from the repo root.
