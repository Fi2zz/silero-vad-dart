#!/usr/bin/env python3
"""Generate reference test data from Python Silero VAD for Dart comparison.

This script runs the Python (numpy/onnxruntime) implementation of Silero VAD
to generate reference outputs that the Dart implementation will be compared
against in CI.

Usage:
    python reference_generator.py <model.onnx> <test.wav> <output_dir>

Outputs:
    output_dir/
        audio_info.json       - Audio file metadata
        probs.json            - Speech probabilities per window
        timestamps.json       - Speech timestamps (samples)
        timestamps_seconds.json - Speech timestamps (seconds)
        vad_iterator.json     - VADIterator streaming events
"""

import json
import math
import struct
import sys
import wave

import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOW_SIZE = 512
CONTEXT_SIZE = 64
SR = 16000


# ---------------------------------------------------------------------------
# WAV reading (pure Python, no torchaudio)
# ---------------------------------------------------------------------------
def read_wav(path):
    """Read a WAV file and return mono float32 samples."""
    with wave.open(path, "rb") as f:
        nchannels = f.getnchannels()
        sampwidth = f.getsampwidth()
        framerate = f.getframerate()
        nframes = f.getnframes()
        raw = f.readframes(nframes)
        if sampwidth == 2:
            fmt = f"<{nframes * nchannels}h"
            samples = np.array(struct.unpack(fmt, raw), dtype=np.float32) / 32768.0
        else:
            raise ValueError(f"Unsupported sampwidth: {sampwidth}")
        if nchannels == 2:
            samples = (samples[0::2] + samples[1::2]) / 2
        return samples, framerate


# ---------------------------------------------------------------------------
# audio_forward (reference implementation matching original Python)
# ---------------------------------------------------------------------------
def audio_forward(session, audio, sr=16000):
    """Process entire audio and return speech probability per window."""
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(CONTEXT_SIZE, dtype=np.float32)

    num_windows = math.ceil(len(audio) / WINDOW_SIZE)
    padded = np.zeros(num_windows * WINDOW_SIZE, dtype=np.float32)
    padded[: len(audio)] = audio

    probs = []
    for i in range(num_windows):
        window = padded[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE]

        concat = np.concatenate([context, window]).reshape(1, -1).astype(np.float32)
        outputs = session.run(
            None,
            {"input": concat, "state": state, "sr": np.array(SR, dtype=np.int64)},
        )

        prob = float(outputs[0][0][0])
        state = outputs[1]
        probs.append(prob)
        context = window[-CONTEXT_SIZE:].copy()

    return probs


# ---------------------------------------------------------------------------
# get_speech_timestamps (reference implementation)
# ---------------------------------------------------------------------------
def get_speech_ts(probs, threshold=0.5, min_speech_ms=250, min_silence_ms=100, speech_pad_ms=30):
    neg_thresh = max(threshold - 0.15, 0.01)
    min_speech_samples = SR * min_speech_ms // 1000
    min_silence_samples = SR * min_silence_ms // 1000
    speech_pad_samples = SR * speech_pad_ms // 1000

    speeches = []
    triggered = False
    temp_end = 0

    for i, prob in enumerate(probs):
        ws = i * WINDOW_SIZE
        we = ws + WINDOW_SIZE

        if not triggered:
            if prob >= threshold:
                speeches.append({"start": ws, "end": 0})
                triggered = True
                temp_end = 0
        else:
            if prob < neg_thresh:
                if temp_end == 0:
                    temp_end = ws
                if we - temp_end >= min_silence_samples:
                    speeches[-1]["end"] = temp_end
                    triggered = False
                    temp_end = 0
            else:
                temp_end = 0

    if triggered and speeches:
        speeches[-1]["end"] = len(probs) * WINDOW_SIZE

    filtered = []
    for s in speeches:
        dur = s["end"] - s["start"]
        if dur < min_speech_samples:
            continue
        ps = max(0, s["start"] - speech_pad_samples)
        pe = min(len(probs) * WINDOW_SIZE, s["end"] + speech_pad_samples)
        if filtered and ps <= filtered[-1]["end"]:
            filtered[-1]["end"] = pe
        else:
            filtered.append({"start": ps, "end": pe})

    return filtered


# ---------------------------------------------------------------------------
# VADIterator (reference implementation)
# ---------------------------------------------------------------------------
def run_vad_iterator(session, audio):
    """Run VADIterator on audio windows and return events."""
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(CONTEXT_SIZE, dtype=np.float32)
    threshold = 0.5
    neg_thresh = max(threshold - 0.15, 0.01)
    min_silence_samples = SR * 100 // 1000  # 100ms
    speech_pad_samples = SR * 30 // 1000    # 30ms
    triggered = False
    temp_end = 0
    current_sample = 0
    events = []

    num_windows = len(audio) // WINDOW_SIZE

    for i in range(num_windows):
        window = audio[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE]

        concat = np.concatenate([context, window]).reshape(1, -1).astype(np.float32)
        outputs = session.run(
            None,
            {"input": concat, "state": state, "sr": np.array(SR, dtype=np.int64)},
        )
        prob = float(outputs[0][0][0])
        state = outputs[1]
        context = window[-CONTEXT_SIZE:].copy()

        chunk_start = current_sample
        chunk_end = current_sample + WINDOW_SIZE
        current_sample = chunk_end

        if not triggered:
            if prob >= threshold:
                triggered = True
                start_pos = max(0, chunk_start - speech_pad_samples)
                events.append({"window": i, "type": "start", "pos": int(start_pos)})
                temp_end = 0
        else:
            if prob < neg_thresh:
                if temp_end == 0:
                    temp_end = chunk_start
                if chunk_end - temp_end >= min_silence_samples:
                    end_pos = temp_end + speech_pad_samples
                    events.append({"window": i, "type": "end", "pos": int(end_pos)})
                    triggered = False
                    temp_end = 0
            else:
                temp_end = 0

    return events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <model.onnx> <test.wav> <output_dir>")
        sys.exit(1)

    model_path = sys.argv[1]
    wav_path = sys.argv[2]
    output_dir = sys.argv[3]

    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    print(f"Reading audio: {wav_path}")
    audio, sr = read_wav(wav_path)

    # 1. Audio info
    info = {
        "sample_rate": sr,
        "length": int(len(audio)),
        "duration": round(len(audio) / sr, 6),
        "dtype": "float32",
    }
    with open(f"{output_dir}/audio_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Audio: {info['length']} samples, {info['duration']}s")

    # 2. audio_forward probabilities
    print("Running audio_forward...")
    probs = audio_forward(session, audio)
    with open(f"{output_dir}/probs.json", "w") as f:
        json.dump(probs, f, indent=2)
    print(f"  -> {len(probs)} windows")

    # 3. get_speech_timestamps (samples)
    print("Running get_speech_timestamps (samples)...")
    ts = get_speech_ts(probs)
    with open(f"{output_dir}/timestamps.json", "w") as f:
        json.dump(ts, f, indent=2)
    print(f"  -> {len(ts)} segments")

    # 4. get_speech_timestamps (seconds)
    print("Running get_speech_timestamps (seconds)...")
    ts_sec = [{"start": round(s["start"] / SR, 6), "end": round(s["end"] / SR, 6)} for s in ts]
    with open(f"{output_dir}/timestamps_seconds.json", "w") as f:
        json.dump(ts_sec, f, indent=2)
    print(f"  -> {len(ts_sec)} segments")

    # 5. VADIterator events
    print("Running VADIterator...")
    events = run_vad_iterator(session, audio)
    with open(f"{output_dir}/vad_iterator.json", "w") as f:
        json.dump(events, f, indent=2)
    print(f"  -> {len(events)} events")

    print(f"\nReference data written to {output_dir}/")


if __name__ == "__main__":
    main()
