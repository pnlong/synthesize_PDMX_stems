"""Spot-listen helper for DDSP-Piano before committing B3 piano routing.

Synthesizes a short piano MIDI with the isolated TF worker and writes WAV
under a local output dir for manual listening.

  SPDMX_DDSP_PYTHON=.venv-ddsp/bin/python \\
    uv run python -m synthesis.ddsp.spot_listen_piano --out /tmp/ddsp_piano_spot.wav

Citations (from upstream README BibTeX):

  @article{renault2023ddsp_piano,
    title={DDSP-Piano: A Neural Sound Synthesizer Informed by Instrument Knowledge},
    author={Renault, Lenny and Mignot, Rémi and Roebel, Axel},
    journal={Journal of the Audio Engineering Society},
    volume={71}, number={9}, pages={552--565}, year={2023}, month={September}
  }

  @inproceedings{renault2022diffpiano,
    title={Differentiable Piano Model for MIDI-to-Audio Performance Synthesis},
    author={Renault, Lenny and Mignot, Rémi and Roebel, Axel},
    booktitle={Proceedings of the 25th International Conference on Digital Audio Effects},
    year={2022}
  }

Training data: MAESTRO (Yamaha Disklavier performances at the International
Piano-e-Competition). License: Apache-2.0.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mido

from synthesis.ddsp.env import DdspEnvError, assert_ddsp_env_ready
from synthesis.ddsp.synthesize import synthesize_stem_ddsp_piano


def _write_fixture_piano_mid(path: Path) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.Message("program_change", program=0, time=0))
    # C major triad (polyphony) then a scale.
    for i, note in enumerate((60, 64, 67)):
        track.append(mido.Message("note_on", note=note, velocity=80, time=0 if i else 0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    track.append(mido.Message("note_off", note=64, velocity=0, time=0))
    track.append(mido.Message("note_off", note=67, velocity=0, time=0))
    for i, note in enumerate(range(60, 72)):
        track.append(mido.Message("note_on", note=note, velocity=70, time=120 if i else 120))
        track.append(mido.Message("note_off", note=note, velocity=0, time=120))
    midi.tracks.append(track)
    midi.save(str(path))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("ddsp_piano_spot.wav"))
    parser.add_argument("--midi", type=Path, default=None, help="Optional input MIDI.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        assert_ddsp_env_ready()
    except DdspEnvError as exc:
        print(f"SKIP spot-listen (env not ready): {exc}")
        print(__doc__)
        return 2

    with tempfile.TemporaryDirectory(prefix="spot_piano_") as tmp:
        mid = args.midi or Path(tmp) / "piano.mid"
        if args.midi is None:
            _write_fixture_piano_mid(mid)
        waveform = synthesize_stem_ddsp_piano(mid)
        import soundfile as sf
        from synthesis.audio import to_mono_numpy
        from shared.config import SAMPLE_RATE

        args.out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(args.out), to_mono_numpy(waveform), SAMPLE_RATE)
        print(
            f"Wrote {args.out} "
            f"(shape={tuple(waveform.shape)}, sr={SAMPLE_RATE}). "
            "Listen before enabling large B3 piano batches."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
