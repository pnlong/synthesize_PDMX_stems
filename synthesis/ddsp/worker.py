"""Worker entrypoint executed inside the isolated TF/DDSP venv.

Usage (from repo root, with SPDMX_DDSP_PYTHON or .venv-ddsp):

  .venv-ddsp/bin/python -m synthesis.ddsp.worker midi_ddsp \\
      --midi stem.mid --instrument violin --out out.wav --weights-dir ...

  .venv-ddsp/bin/python -m synthesis.ddsp.worker ddsp_piano \\
      --midi stem.mid --out out.wav --ckpt ... --gin ... --piano-root ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _write_wav(path: Path, audio, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        # Prefer mono file for pipeline simplicity.
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=-1) if arr.shape[-1] <= 8 else arr.reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), arr, sample_rate, subtype="FLOAT")


def _run_midi_ddsp(args: argparse.Namespace) -> dict:
    import numpy as np

    from midi_ddsp.data_handling.instrument_name_utils import INST_NAME_TO_ID_DICT
    from midi_ddsp.midi_ddsp_synthesize import load_pretrained_model
    from midi_ddsp.utils.midi_synthesis_utils import synthesize_mono_midi

    load_kwargs: dict = {}
    weights = Path(args.weights_dir) if args.weights_dir else None
    if weights and weights.is_dir():
        synth_ckpt = weights / "synthesis_generator" / "50000"
        expr_ckpt = weights / "expression_generator" / "5000"
        nested = weights / "midi_ddsp_model_weights_urmp_9_10"
        if nested.is_dir():
            synth_ckpt = nested / "synthesis_generator" / "50000"
            expr_ckpt = nested / "expression_generator" / "5000"
        if synth_ckpt.exists():
            load_kwargs["synthesis_generator_path"] = str(synth_ckpt)
        if expr_ckpt.exists():
            load_kwargs["expression_generator_path"] = str(expr_ckpt)

    synthesis_generator, expression_generator = load_pretrained_model(**load_kwargs)

    instrument_id = INST_NAME_TO_ID_DICT[args.instrument]
    midi_audio, *_rest = synthesize_mono_midi(
        synthesis_generator,
        expression_generator,
        args.midi,
        instrument_id,
        output_dir=None,
    )
    audio = np.asarray(midi_audio)
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    audio = np.squeeze(audio)
    sample_rate = 16000
    _write_wav(Path(args.out), audio, sample_rate)
    return {
        "ok": True,
        "backend": "midi_ddsp",
        "instrument": args.instrument,
        "sample_rate": sample_rate,
        "out": args.out,
    }


def _run_ddsp_piano(args: argparse.Namespace) -> dict:
    """Run DDSP-Piano inference in fixed-duration chunks (matches gin training)."""
    import os
    import sys

    import gin
    import numpy as np
    import tensorflow as tf
    from absl import logging
    from ddsp.training import trainers, train_util
    from ddsp.training.models import get_model
    from soundfile import write as sf_write

    piano_root = Path(args.piano_root)
    if not piano_root.is_dir():
        raise FileNotFoundError(
            f"DDSP-Piano root not found: {piano_root}. "
            "Clone https://github.com/lrenault/ddsp-piano into "
            "synthesis/ddsp/third_party/ddsp-piano"
        )

    sys.path.insert(0, str(piano_root))
    os.chdir(piano_root)

    from ddsp_piano.data_pipeline import get_dummy_data
    from ddsp_piano.utils.io_utils import load_midi_as_conditioning

    warm_up = float(getattr(args, "warm_up", 0.5))
    # Longer chunks = fewer CPU forwards. Override with --chunk-sec / env.
    chunk_sec = float(
        getattr(args, "chunk_sec", None)
        or os.environ.get("SPDMX_DDSP_PIANO_CHUNK_SEC", "12")
    )
    overlap_sec = float(
        getattr(args, "overlap_sec", None)
        or os.environ.get("SPDMX_DDSP_PIANO_CHUNK_OVERLAP_SEC", "2.0")
    )
    if overlap_sec < 0:
        raise ValueError("overlap_sec must be >= 0")
    if overlap_sec >= chunk_sec:
        raise ValueError(
            f"overlap_sec ({overlap_sec}) must be smaller than chunk_sec ({chunk_sec})"
        )

    from synthesis.ddsp.chunking import (
        frames_to_samples,
        plan_chunk_frame_spans,
        stitch_audio_chunks,
    )

    full = load_midi_as_conditioning(
        args.midi,
        duration=None,
        warm_up_duration=0.0,
    )
    cond = np.asarray(full["conditioning"])
    pedal = np.asarray(full["pedal"])
    frame_rate = 250
    chunk_frames = int(chunk_sec * frame_rate)
    overlap_frames = int(overlap_sec * frame_rate)
    warm_frames = int(warm_up * frame_rate)
    n_frames = cond.shape[1]
    frame_spans = plan_chunk_frame_spans(n_frames, chunk_frames, overlap_frames)

    gin.parse_config_file(str(args.gin))
    gin.bind_parameter("%inference", True)
    gin.bind_parameter("%duration", chunk_sec + warm_up)

    strategy = train_util.get_strategy()
    with strategy.scope():
        model = get_model()
        trainer = trainers.Trainer(model=model, strategy=strategy)
        trainer.build(
            get_dummy_data(
                batch_size=1,
                duration=chunk_sec + warm_up,
                sample_rate=model.sample_rate,
            )
        )
        trainer.restore(str(args.ckpt))

    audio_chunks: list[np.ndarray] = []
    sample_spans: list[tuple[int, int]] = []
    for start, end in frame_spans:
        cond_chunk = cond[:, start:end]
        pedal_chunk = pedal[:, start:end]
        # Left-pad warm-up silence so recurrent state settles.
        if warm_frames > 0:
            cond_chunk = np.pad(
                cond_chunk,
                ((0, 0), (warm_frames, 0), (0, 0), (0, 0)),
                mode="constant",
            )
            pedal_chunk = np.pad(
                pedal_chunk,
                ((0, 0), (warm_frames, 0), (0, 0)),
                mode="constant",
            )
        # Right-pad to fixed model length.
        need = int((chunk_sec + warm_up) * frame_rate) - cond_chunk.shape[1]
        if need > 0:
            cond_chunk = np.pad(
                cond_chunk, ((0, 0), (0, need), (0, 0), (0, 0)), mode="constant"
            )
            pedal_chunk = np.pad(
                pedal_chunk, ((0, 0), (0, need), (0, 0)), mode="constant"
            )
        inputs = {
            "conditioning": tf.convert_to_tensor(cond_chunk, dtype=tf.float32),
            "pedal": tf.convert_to_tensor(pedal_chunk, dtype=tf.float32),
            "piano_model": tf.convert_to_tensor([[int(args.piano_type)]]),
        }
        outs = model(inputs)
        chunk_audio = outs["audio_synth"][0, int(warm_up * model.sample_rate) :].numpy()
        # Keep only the real (non-right-pad) audio for this MIDI span.
        start_sample = frames_to_samples(start, frame_rate, model.sample_rate)
        end_sample = frames_to_samples(end, frame_rate, model.sample_rate)
        keep = end_sample - start_sample
        audio_chunks.append(chunk_audio[:keep])
        sample_spans.append((start_sample, end_sample))

    overlap_samples = frames_to_samples(overlap_frames, frame_rate, model.sample_rate)
    audio = stitch_audio_chunks(audio_chunks, sample_spans, overlap_samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf_write(str(out_path), data=np.asarray(audio, dtype=np.float32), samplerate=model.sample_rate)
    logging.info("DDSP-Piano wrote %s", out_path)
    return {
        "ok": True,
        "backend": "ddsp_piano",
        "piano_type": int(args.piano_type),
        "sample_rate": int(model.sample_rate),
        "out": str(out_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synthesis.ddsp.worker")
    sub = parser.add_subparsers(dest="command", required=True)

    p_midi = sub.add_parser("midi_ddsp")
    p_midi.add_argument("--midi", required=True)
    p_midi.add_argument("--instrument", required=True)
    p_midi.add_argument("--out", required=True)
    p_midi.add_argument("--weights-dir", default="")

    p_piano = sub.add_parser("ddsp_piano")
    p_piano.add_argument("--midi", required=True)
    p_piano.add_argument("--out", required=True)
    p_piano.add_argument("--piano-type", type=int, default=0)
    p_piano.add_argument("--ckpt", required=True)
    p_piano.add_argument("--gin", required=True)
    p_piano.add_argument("--piano-root", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "midi_ddsp":
            status = _run_midi_ddsp(args)
        elif args.command == "ddsp_piano":
            status = _run_ddsp_piano(args)
        else:
            raise SystemExit(f"unknown command {args.command}")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        print(f"worker error: {exc}", file=sys.stderr, flush=True)
        return 1

    print(json.dumps(status), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
