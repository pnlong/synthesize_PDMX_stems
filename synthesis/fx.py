"""Light post-fluidsynth FX profiles for Slakh-style rendering."""

from __future__ import annotations

from dataclasses import dataclass

import torch

FX_PROFILE_DRY = "dry"
FX_PROFILE_DEFAULT = "default"
FX_PROFILE_LIGHT = "light"
FX_PROFILE_WARM = "warm"

FX_PROFILES = frozenset({
    FX_PROFILE_DRY,
    FX_PROFILE_DEFAULT,
    FX_PROFILE_LIGHT,
    FX_PROFILE_WARM,
})


@dataclass(frozen=True)
class FluidsynthFxSettings:
    reverb_active: bool | None = None
    chorus_active: bool | None = None
    reverb_level: float | None = None
    reverb_room_size: float | None = None
    post_eq: str | None = None


_FLUIDSYNTH_FX: dict[str, FluidsynthFxSettings] = {
    FX_PROFILE_DRY: FluidsynthFxSettings(reverb_active=False, chorus_active=False),
    FX_PROFILE_DEFAULT: FluidsynthFxSettings(),
    FX_PROFILE_LIGHT: FluidsynthFxSettings(
        reverb_active=True,
        chorus_active=False,
        reverb_level=0.2,
        reverb_room_size=0.25,
    ),
    FX_PROFILE_WARM: FluidsynthFxSettings(
        reverb_active=True,
        chorus_active=False,
        reverb_level=0.25,
        reverb_room_size=0.3,
        post_eq="warm",
    ),
}


def fluidsynth_fx_args(fx_profile: str | None) -> list[str]:
    """Extra fluidsynth CLI args for a named FX profile."""
    if fx_profile is None or fx_profile == FX_PROFILE_DEFAULT:
        return []

    settings = _FLUIDSYNTH_FX.get(fx_profile)
    if settings is None:
        raise ValueError(f"Unknown FX profile: {fx_profile}")

    args: list[str] = []
    if settings.reverb_active is False:
        args.extend(["-R", "0"])
    if settings.chorus_active is False:
        args.extend(["-C", "0"])
    if settings.reverb_level is not None:
        args.extend(["-o", f"synth.reverb.level={settings.reverb_level}"])
    if settings.reverb_room_size is not None:
        args.extend(["-o", f"synth.reverb.room-size={settings.reverb_room_size}"])
    return args


def apply_post_fx(waveform: torch.Tensor, fx_profile: str | None) -> torch.Tensor:
    """Apply optional post-render EQ in the float domain."""
    if fx_profile is None or fx_profile == FX_PROFILE_DEFAULT:
        return waveform

    settings = _FLUIDSYNTH_FX.get(fx_profile)
    if settings is None or settings.post_eq is None:
        return waveform

    if settings.post_eq == "warm":
        return _apply_warm_eq(waveform)
    raise ValueError(f"Unknown post EQ for profile: {fx_profile}")


def _apply_warm_eq(waveform: torch.Tensor) -> torch.Tensor:
    """Gentle warmth: slight low shelf boost, slight high roll-off."""
    import torchaudio

    audio = waveform.detach().cpu().to(torch.float32)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio[0]

    sample_rate = 44100
    low = torchaudio.functional.highpass_biquad(audio, sample_rate, 80.0)
    lows = audio - low
    highs = low
    warmed = lows * 1.08 + highs * 0.92
    return warmed.to(torch.float32)
