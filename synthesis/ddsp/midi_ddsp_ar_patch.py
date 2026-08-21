"""Speed up MIDI-DDSP autoregressive decode with ``tf.while_loop`` + ``tf.function``.

Upstream MIDI-DDSP runs a Python ``for`` loop over every frame (250 Hz), which
keeps GPU util near zero. This monkeypatches the two AR entry points used at
inference.

Disable with ``SPDMX_DDSP_AR_PATCH=0``.
"""

from __future__ import annotations

import os
import sys

_APPLIED = False


def ar_patch_enabled() -> bool:
    return os.environ.get("SPDMX_DDSP_AR_PATCH", "1") != "0"


def apply_midi_ddsp_ar_patch() -> bool:
    """Monkeypatch MIDI-DDSP AR methods. Safe to call repeatedly."""
    global _APPLIED
    if _APPLIED:
        return True
    if not ar_patch_enabled():
        return False

    import tensorflow as tf
    from midi_ddsp.modules.cond_rnn import TwoLayerCondAutoregRNN
    from midi_ddsp.modules.expression_generator import ExpressionGenerator

    def _stack_ta(ta: tf.TensorArray) -> tf.Tensor:
        return tf.transpose(ta.stack(), [1, 0, 2])

    def _get_fn_cache(layer) -> dict:
        cache = getattr(layer, "_spdmx_ar_fn_cache", None)
        if cache is None:
            cache = {}
            layer._spdmx_ar_fn_cache = cache
        return cache

    def cond_rnn_autoregressive(self, cond, training=False, display_progressbar=False):
        del display_progressbar
        # encode_cond includes BiRNN — keep it outside the cached AR step graph so
        # the while_loop graph stays smaller / more reusable across lengths.
        cond_encode = self.encode_cond(cond, training=False)
        return _cond_rnn_ar_steps(self, cond_encode)

    def _cond_rnn_ar_steps(layer, cond_encode):
        cache = _get_fn_cache(layer)
        fn = cache.get("cond_rnn")
        if fn is None:
            nhid = int(layer.nhid)
            n_out = int(layer.n_out)
            cls_name = type(layer).__name__
            if cls_name == "MidiToF0LDAutoregDecoder":
                from midi_ddsp.modules.synth_params_decoder import (
                    ABSOLUTE_LD_BINS,
                    RELATIVE_F0_BINS,
                )

                keys = (
                    "f0_midi_dv_logits",
                    "f0_midi_dv_onehot",
                    "ld_logits",
                    "ld_onehot",
                    "curr_out",
                )
                widths = {
                    "f0_midi_dv_logits": RELATIVE_F0_BINS,
                    "f0_midi_dv_onehot": RELATIVE_F0_BINS,
                    "ld_logits": ABSOLUTE_LD_BINS,
                    "ld_onehot": ABSOLUTE_LD_BINS,
                    "curr_out": n_out,
                }
            else:
                keys = ("f0_midi_dv_logits", "f0_midi_dv_onehot", "curr_out")
                widths = {k: n_out for k in keys}

            @tf.function(reduce_retracing=True)
            def _fn(cond_encode_t):
                batch_size = tf.shape(cond_encode_t)[0]
                length = tf.shape(cond_encode_t)[1]
                prev_out = tf.zeros([batch_size, 1, n_out], dtype=tf.float32)
                state1 = tf.zeros([batch_size, nhid], dtype=tf.float32)
                state2 = tf.zeros([batch_size, nhid], dtype=tf.float32)
                tas0 = tuple(
                    tf.TensorArray(
                        dtype=tf.float32,
                        size=length,
                        dynamic_size=False,
                        clear_after_read=False,
                        element_shape=tf.TensorShape([None, widths[k]]),
                    )
                    for k in keys
                )

                def cond_fn(i, *_rest):
                    return i < length

                def body_fn(i, prev_out_i, state1_i, state2_i, *ta_list):
                    curr_cond = cond_encode_t[:, i : i + 1, :]
                    prev_enc = layer.encode_out(prev_out_i, training=False)
                    curr_raw, (ns1, ns2) = layer._one_step(
                        curr_cond, prev_enc, (state1_i, state2_i), training=False
                    )
                    curr_out = layer.sample_out(curr_raw, training=False)
                    new_tas = tuple(
                        ta.write(i, tf.squeeze(curr_out[k], axis=1))
                        for ta, k in zip(ta_list, keys)
                    )
                    next_prev = curr_out["curr_out"]
                    # Force [batch, 1, n_out] for while_loop shape stability.
                    next_prev = tf.reshape(next_prev, [batch_size, 1, n_out])
                    return (i + 1, next_prev, ns1, ns2) + new_tas

                results = tf.while_loop(
                    cond_fn,
                    body_fn,
                    loop_vars=(
                        tf.constant(0, dtype=tf.int32),
                        prev_out,
                        state1,
                        state2,
                    )
                    + tas0,
                    shape_invariants=(
                        tf.TensorShape([]),
                        tf.TensorShape([None, 1, n_out]),
                        tf.TensorShape([None, nhid]),
                        tf.TensorShape([None, nhid]),
                    )
                    + tuple(tf.TensorShape(None) for _ in keys),
                    parallel_iterations=1,
                    maximum_iterations=length,
                )
                return {k: _stack_ta(ta) for k, ta in zip(keys, results[4:])}

            cache["cond_rnn"] = _fn
            cache["cond_rnn_keys"] = keys
            fn = _fn
        return fn(cond_encode)

    def expression_autoregressive(self, cond, training=False):
        del training
        note_pitch = cond["note_pitch"][..., tf.newaxis]
        cond_enc = self.encode_cond(cond, training=False)
        return _expression_ar_steps(self, cond_enc, note_pitch)

    def _expression_ar_steps(layer, cond_enc, note_pitch):
        cache = _get_fn_cache(layer)
        fn = cache.get("expression")
        if fn is None:
            nhid = int(layer.nhid)
            n_out = int(layer.n_out)

            def _sample_out(out, pitch_t):
                raw = out["raw_output"]
                sampled = raw * tf.cast(pitch_t != 0, tf.float32)
                return {"raw_output": raw, "output": sampled}

            @tf.function(reduce_retracing=True)
            def _fn(cond_enc_t, note_pitch_t):
                batch_size = tf.shape(cond_enc_t)[0]
                length = tf.shape(cond_enc_t)[1]
                prev_out = tf.zeros([batch_size, 1, n_out], dtype=tf.float32)
                state1 = tf.zeros([batch_size, nhid], dtype=tf.float32)
                state2 = tf.zeros([batch_size, nhid], dtype=tf.float32)
                keys = ("raw_output", "output")
                tas0 = tuple(
                    tf.TensorArray(
                        dtype=tf.float32,
                        size=length,
                        dynamic_size=False,
                        clear_after_read=False,
                        element_shape=tf.TensorShape([None, n_out]),
                    )
                    for _ in keys
                )

                def cond_fn(i, *_rest):
                    return i < length

                def body_fn(i, prev_out_i, state1_i, state2_i, *ta_list):
                    curr_cond = cond_enc_t[:, i : i + 1, :]
                    prev_enc = layer.encode_out(prev_out_i)
                    curr_raw, (ns1, ns2) = layer._one_step(
                        curr_cond, prev_enc, (state1_i, state2_i), training=False
                    )
                    curr_out = _sample_out(curr_raw, note_pitch_t[:, i : i + 1, :])
                    new_tas = tuple(
                        ta.write(i, tf.squeeze(curr_out[k], axis=1))
                        for ta, k in zip(ta_list, keys)
                    )
                    next_prev = tf.reshape(curr_out["output"], [batch_size, 1, n_out])
                    return (i + 1, next_prev, ns1, ns2) + new_tas

                results = tf.while_loop(
                    cond_fn,
                    body_fn,
                    loop_vars=(
                        tf.constant(0, dtype=tf.int32),
                        prev_out,
                        state1,
                        state2,
                    )
                    + tas0,
                    shape_invariants=(
                        tf.TensorShape([]),
                        tf.TensorShape([None, 1, n_out]),
                        tf.TensorShape([None, nhid]),
                        tf.TensorShape([None, nhid]),
                    )
                    + tuple(tf.TensorShape(None) for _ in keys),
                    parallel_iterations=1,
                    maximum_iterations=length,
                )
                return {k: _stack_ta(ta) for k, ta in zip(keys, results[4:])}

            cache["expression"] = _fn
            fn = _fn
        return fn(cond_enc, note_pitch)

    TwoLayerCondAutoregRNN.autoregressive = cond_rnn_autoregressive
    ExpressionGenerator.autoregressive = expression_autoregressive
    _APPLIED = True
    sys.stderr.write(
        "[ddsp] MIDI-DDSP autoregressive tf.function+while_loop patch applied\n"
    )
    sys.stderr.flush()
    return True
