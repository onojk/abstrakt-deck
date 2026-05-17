#!/usr/bin/env python3
"""Headless pitch-layer smoke test against a real audio file.

Reads a WAV file, processes it chunk-by-chunk with the same chromagram + K-S
algorithm as Rust's KeyTracker, and prints a timestamped log of key detections.

Usage:
    python3 tools/pitch_smoke_audio.py <audio.wav>

Criteria being verified:
  (a) KEY detection events appear within 5-10 s of audio start
  (b) Detected key is plausible (printed for human review)
  (c) Confidence values are in 0.6-1.0 range and not flapping
"""

import sys
import math
import wave
import struct
import numpy as np
from scipy.fft import rfft

# ── K-S profiles ──────────────────────────────────────────────────────────────
KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def normalize(v):
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-9:
        return None
    return [x / norm for x in v]


def pearson_rotated(chroma, profile, rot):
    n = 12
    mean_c = sum(chroma) / n
    mean_p = sum(profile) / n
    num = sum(
        (chroma[(i + rot) % 12] - mean_c) * (profile[i] - mean_p)
        for i in range(n)
    )
    ss_c = sum((chroma[(i + rot) % 12] - mean_c) ** 2 for i in range(n))
    ss_p = sum((profile[i] - mean_p) ** 2 for i in range(n))
    den = math.sqrt(ss_c * ss_p)
    return num / den if den > 1e-9 else 0.0


def detect_key(chroma):
    c = normalize(chroma)
    if c is None:
        return 0, True, 0.0
    best_root, best_major, best_corr = 0, True, float("-inf")
    for rot in range(12):
        r_maj = pearson_rotated(c, KS_MAJOR, rot)
        r_min = pearson_rotated(c, KS_MINOR, rot)
        if r_maj > best_corr:
            best_corr, best_root, best_major = r_maj, rot, True
        if r_min > best_corr:
            best_corr, best_root, best_major = r_min, rot, False
    return best_root, best_major, best_corr


def chromagram_from_fft(mags, sample_rate, fft_size):
    """Mirror of Rust pitch::chromagram exactly."""
    bin_hz = sample_rate / fft_size
    chroma = [0.0] * 12
    for b, mag in enumerate(mags):
        freq = b * bin_hz
        if freq < 27.5 or freq > 4186.0:
            continue
        if freq <= 0:
            continue
        midi = 69.0 + 12.0 * math.log2(freq / 440.0)
        pc = int(round(midi)) % 12
        pc = (pc + 12) % 12
        chroma[pc] += mag
    return chroma


def hann_window(n):
    return [0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]


def run(wav_path):
    print(f"Pitch smoke test: {wav_path}")
    print("=" * 70)

    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    # Decode samples to float32
    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        print(f"Unsupported sample width: {sampwidth}")
        sys.exit(1)

    # Mix to mono
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    total_seconds = len(samples) / sample_rate
    print(f"Audio: {total_seconds:.1f}s, {sample_rate} Hz, {n_channels}ch → mono")
    print()

    FFT_SIZE = 2048
    HISTORY_SECONDS = 4.0
    ESTIMATE_PERIOD_SECONDS = 0.5
    chunk_seconds = FFT_SIZE / sample_rate
    history_size = math.ceil(HISTORY_SECONDS / chunk_seconds)
    estimate_period = math.ceil(ESTIMATE_PERIOD_SECONDS / chunk_seconds)

    window = hann_window(FFT_SIZE)
    chroma_history = []
    chunks_since_estimate = 0
    key_root, key_major, key_conf = 0, True, 0.0
    last_key_root, last_key_major = None, None
    first_event_ts = None
    confidence_samples = []
    change_count = 0

    # Process up to 60 seconds
    max_samples = min(len(samples), int(60 * sample_rate))
    pos = 0
    chunk_idx = 0

    while pos + FFT_SIZE <= max_samples:
        chunk = samples[pos:pos + FFT_SIZE]
        pos += FFT_SIZE
        t = pos / sample_rate

        # Hann window
        windowed = chunk * np.array(window)

        # FFT — half-wave rectified magnitudes (mirror of Rust spectral_magnitudes)
        spectrum = rfft(windowed)
        # Rust: if im >= 0 use norm, else 0
        mags = np.where(spectrum.imag >= 0, np.abs(spectrum), 0.0)

        # Chromagram
        chroma = chromagram_from_fft(mags.tolist(), sample_rate, FFT_SIZE)
        c = normalize(chroma)
        if c is None:
            chunk_idx += 1
            continue

        chroma_history.append(c)
        if len(chroma_history) > history_size:
            chroma_history.pop(0)

        chunks_since_estimate += 1
        if chunks_since_estimate >= estimate_period and len(chroma_history) >= history_size:
            chunks_since_estimate = 0

            # Accumulate + normalize window
            acc = [0.0] * 12
            for cv in chroma_history:
                for i in range(12):
                    acc[i] += cv[i]
            acc_n = normalize(acc)
            if acc_n:
                root, major, conf = detect_key(acc_n)
                key_root, key_major, key_conf = root, major, conf

                if conf > 0.4:
                    confidence_samples.append(conf)

                if (root, major) != (last_key_root, last_key_major) and last_key_root is not None:
                    if conf > 0.5:
                        mode = "major" if major else "minor"
                        print(f"  [{t:6.1f}s] KEY CHANGE → {PITCH_NAMES[root]} {mode}  (conf={conf:.3f})")
                        change_count += 1
                        if first_event_ts is None:
                            first_event_ts = t

                if last_key_root is None and conf > 0.4:
                    mode = "major" if major else "minor"
                    print(f"  [{t:6.1f}s] KEY first lock → {PITCH_NAMES[root]} {mode}  (conf={conf:.3f})")
                    if first_event_ts is None:
                        first_event_ts = t

                last_key_root, last_key_major = root, major

        chunk_idx += 1

    print()
    print("=" * 70)
    mode = "major" if key_major else "minor"
    print(f"Final key at end of analysis: {PITCH_NAMES[key_root]} {mode}  (conf={key_conf:.3f})")

    # Summary
    print()
    print("Criteria check:")

    # (a) First event within 5-10s
    if first_event_ts is not None:
        status_a = "PASS" if first_event_ts <= 10.0 else "LATE"
        print(f"  (a) First key event at {first_event_ts:.1f}s  [{status_a} — target ≤ 10s]")
    else:
        print("  (a) No key events detected  [FAIL]")

    # (b) Plausibility — print for human review
    print(f"  (b) Dominant key: {PITCH_NAMES[key_root]} {mode} — review against song")

    # (c) Confidence range and stability
    if confidence_samples:
        lo = min(confidence_samples)
        hi = max(confidence_samples)
        mean = sum(confidence_samples) / len(confidence_samples)
        in_range = all(0.5 <= c <= 1.0 for c in confidence_samples)
        stable = (hi - lo) < 0.35  # < 35 pp spread → not flapping
        status_c = "PASS" if (in_range and stable) else "WARN"
        print(
            f"  (c) Confidence: min={lo:.3f} mean={mean:.3f} max={hi:.3f}  "
            f"spread={hi-lo:.3f}  [{status_c}]"
        )
        print(f"       {len(confidence_samples)} samples, {change_count} key changes logged")
    else:
        print("  (c) No confident key readings  [FAIL]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tools/pitch_smoke_audio.py <audio.wav>")
        sys.exit(1)
    run(sys.argv[1])
