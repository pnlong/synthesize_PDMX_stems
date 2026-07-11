# Stable Audio 3 submodule

Git submodule for [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3).

## First-time setup

From the **spdmx repo root**:

```bash
git submodule update --init synthesis/realify/stable-audio-3
```

If the submodule is not registered yet:

```bash
git submodule add https://github.com/Stability-AI/stable-audio-3.git synthesis/realify/stable-audio-3
```

Then follow [`SETUP.md`](../SETUP.md) for `uv`, model weights, and flash-attention.

Upstream docs: [stable-audio-3 README](https://github.com/Stability-AI/stable-audio-3/blob/main/README.md)
