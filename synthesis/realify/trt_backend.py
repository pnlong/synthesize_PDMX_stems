"""TensorRT backend for SA3 audio-to-audio realification."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from shared.config import SAMPLE_RATE

_TRT_SCRIPT_DIR = (
    Path(__file__).resolve().parent / "stable-audio-3" / "optimized" / "tensorRT" / "scripts"
)

_MODEL_TO_TRT = {
    "medium": ("medium", "same-l"),
    "small-music": ("sm-music", "same-s"),
}


class TrtRealifyError(RuntimeError):
    """TensorRT realify backend is unavailable or failed."""


def trt_script_dir() -> Path:
    return _TRT_SCRIPT_DIR


def trt_available() -> bool:
    return _TRT_SCRIPT_DIR.is_dir() and (_TRT_SCRIPT_DIR / "sa3_trt_core.py").is_file()


def _configure_trt_import_path() -> None:
    script_dir = str(_TRT_SCRIPT_DIR)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)


@lru_cache(maxsize=1)
def _load_trt_modules():
    if not trt_available():
        raise TrtRealifyError(
            f"TensorRT scripts not found under {trt_script_dir()}. "
            "See synthesis/realify/stable-audio-3/optimized/tensorRT/README.md."
        )
    _configure_trt_import_path()
    try:
        import sa3_trt_core as core  # type: ignore[import-not-found]
    except ImportError as exc:
        raise TrtRealifyError(
            "Failed to import sa3_trt_core. Install TensorRT deps per "
            "synthesis/realify/stable-audio-3/optimized/tensorRT/README.md."
        ) from exc
    return core


class TrtRealifySession:
    """Persistent TensorRT session for audio-to-audio generation."""

    @property
    def model_config(self) -> dict:
        core = _load_trt_modules()
        max_samples = 4096 * core.SAMPLES_PER_LATENT
        return {"sample_size": max_samples}

    def __init__(
        self,
        *,
        model_name: str = "medium",
        steps: int = 8,
        precision: str = "fp16mixed",
        models_dir: str | None = None,
        quiet: bool = True,
    ):
        if model_name not in _MODEL_TO_TRT:
            raise TrtRealifyError(f"Unsupported TRT model mapping for {model_name!r}")
        self.model_name = model_name
        self.dit_name, self.decoder_name = _MODEL_TO_TRT[model_name]
        self.steps = steps
        self.precision = precision
        self.quiet = quiet
        self.core = _load_trt_modules()
        self._runners: dict | None = None
        self._tokenizer = None
        self._dist_shift = None
        self._dit: object | None = None
        if models_dir is not None:
            self.core.MODELS_DIR = Path(models_dir).resolve()
            self.core.ARCH_DIR = self.core.MODELS_DIR / self.core.ARCH

    def _ensure_loaded(self) -> None:
        if self._runners is not None:
            return

        core = self.core
        core._import_heavy()
        needed = core.get_engine_files(
            self.dit_name,
            self.decoder_name,
            self.precision,
            with_encoder=True,
        )
        core._ensure_files(needed)

        import concurrent.futures
        import runtime as rt

        rt.MODELS_DIR = str(core.MODELS_DIR)
        rt.ARCH_DIR = str(core.ARCH_DIR)
        state = rt.load()
        self._tokenizer = state["tokenizer"]
        self._dist_shift = state["dist_shift"]

        engine_specs = {
            "t5": core.T5GEMMA_PATH,
            "dit": core.get_dit_engine_path(self.dit_name, self.precision),
            "dec": core.get_decoder_engine_path(self.decoder_name, self.precision),
            "enc": core.ENCODER_PATHS[self.decoder_name],
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(engine_specs)) as ex:
            futs = {name: ex.submit(core.TRTRunner, path) for name, path in engine_specs.items()}
            self._runners = {name: fut.result() for name, fut in futs.items()}
        self._dit = core.DiTRunner(self._runners["dit"])

    @staticmethod
    def _waveform_to_encoder_input(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
        wf = waveform.detach().cpu().float()
        if wf.ndim == 1:
            wf = wf.unsqueeze(0)
        if wf.shape[0] == 1:
            wf = wf.repeat(2, 1)
        audio = wf.T.contiguous().numpy().astype(np.float32)
        if audio.shape[0] >= target_samples:
            audio = audio[:target_samples]
        else:
            pad = target_samples - audio.shape[0]
            audio = np.pad(audio, ((0, pad), (0, 0)), mode="constant")
        return torch.from_numpy(audio).unsqueeze(0).cuda()

    @staticmethod
    def _pcm_to_waveform(pcm: np.ndarray) -> torch.Tensor:
        """Return (channels, samples) float32 tensor in [-1, 1]."""
        if pcm.ndim != 2:
            raise ValueError(f"Expected PCM shape (samples, channels), got {pcm.shape}")
        scaled = pcm.astype(np.float32) / 32767.0
        stereo = torch.from_numpy(scaled.T.copy())
        return stereo

    def generate(
        self,
        *,
        waveform: torch.Tensor,
        prompt: str,
        duration_seconds: float,
        init_noise_level: float,
        seed: int,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        if cfg_scale != 1.0:
            raise TrtRealifyError("TensorRT backend currently supports cfg_scale=1.0 only.")

        self._ensure_loaded()
        core = self.core
        assert self._runners is not None
        assert self._dit is not None

        seconds = max(duration_seconds, 0.25)
        T_lat = int(np.ceil(seconds * SAMPLE_RATE / core.SAMPLES_PER_LATENT))
        T_lat = max(1, min(T_lat, 4096))
        target_samples = T_lat * core.SAMPLES_PER_LATENT
        requested_samples = int(round(seconds * SAMPLE_RATE))

        embeds, mask = core.t5gemma_encode(
            self._runners["t5"],
            self._tokenizer,
            prompt,
        )
        audio_t = self._waveform_to_encoder_input(waveform, target_samples)
        init_latents = core.encoder_encode(self._runners["enc"], audio_t)

        sigma_max = float(init_noise_level)
        sigmas = core.build_pingpong_schedule(
            self.steps,
            sigma_max=sigma_max,
            dist_shift=self._dist_shift,
            latent_len=T_lat,
        )

        g = torch.Generator(device="cuda")
        g.manual_seed(int(seed))
        pure_noise = torch.randn(
            1,
            core.IO_CHANNELS,
            T_lat,
            device="cuda",
            dtype=torch.float32,
            generator=g,
        )
        noise = init_latents * (1.0 - sigma_max) + pure_noise * sigma_max
        local_add_cond = torch.zeros((1, 257, T_lat), device="cuda", dtype=torch.float32)

        def model_fn(x, t):
            return self._dit.step(x, t, embeds, mask, seconds, local_add_cond)

        latents = core.sample_flow_pingpong(
            model_fn,
            noise,
            sigmas,
            seed=int(seed) + 1,
            paste_back=None,
            on_step=None,
        )
        audio = core.decoder_decode(self._runners["dec"], latents)

        if audio.dtype == torch.int32:
            pcm_gpu = audio[0, :requested_samples].clamp(-32767, 32767).to(torch.int16)
            pcm = pcm_gpu.contiguous().cpu().numpy()
        else:
            audio_gpu = audio[0, ..., :requested_samples]
            pcm = (audio_gpu.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).T.contiguous().cpu().numpy()

        return self._pcm_to_waveform(pcm)


_SESSIONS: dict[tuple[str, int, str], TrtRealifySession] = {}


def get_trt_session(*, model_name: str, steps: int, precision: str = "fp16mixed") -> TrtRealifySession:
    key = (model_name, steps, precision)
    if key not in _SESSIONS:
        _SESSIONS[key] = TrtRealifySession(
            model_name=model_name,
            steps=steps,
            precision=precision,
            quiet=True,
        )
    return _SESSIONS[key]
