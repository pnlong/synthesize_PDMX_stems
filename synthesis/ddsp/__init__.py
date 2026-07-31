"""Neural DDSP synthesis backends (MIDI-DDSP + DDSP-Piano) for ablation B3.

Heavy TF/Torch wrappers are imported lazily so ``python -m synthesis.ddsp.worker``
can run inside the isolated TF venv without requiring PyTorch.
"""

from synthesis.ddsp.routing import (
    BACKEND_DDSP_PIANO,
    BACKEND_MIDI_DDSP,
    BACKEND_SOUNDFONT,
    StemRoute,
    is_monophonic_midi,
    route_stem,
)

__all__ = [
    "BACKEND_DDSP_PIANO",
    "BACKEND_MIDI_DDSP",
    "BACKEND_SOUNDFONT",
    "StemRoute",
    "is_monophonic_midi",
    "route_stem",
    "synthesize_stem_ddsp_piano",
    "synthesize_stem_midi_ddsp",
    "synthesize_stem_neural",
]


def __getattr__(name: str):
    if name in (
        "synthesize_stem_ddsp_piano",
        "synthesize_stem_midi_ddsp",
        "synthesize_stem_neural",
    ):
        from synthesis.ddsp import synthesize as _synthesize

        return getattr(_synthesize, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
