#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone audio generator for examination.

Generates the WAV audio tracks for square and sawtooth waveforms
without creating any video files. This lets you inspect the raw
audio waveforms for distortion or other artifacts.

Usage:
    python3 generate_audio_only.py

Outputs:
    renders/square_wave_audio.wav
    renders/sawtooth_wave_audio.wav
"""

import os
#import sys

import numpy as np
import scipy.io.wavfile as wavfile

# ---------------------------------------------------------------------------
# Constants (must match the generator)
# ---------------------------------------------------------------------------
HOLD_FRAMES = 90
ANIM_FRAMES = 300
NUM_FRAMES = HOLD_FRAMES + ANIM_FRAMES
NUM_TERMS_MIN = 1
NUM_TERMS_MAX = 32
FPS = 30
CARRIER_FREQ = 440.0
SAMPLING_RATE = 44100
TWO_CYCLES = 4 * np.pi
RENDER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders")
os.makedirs(RENDER_DIR, exist_ok=True)

# -3 dB gain reduction
DB_REDUCTION = 3.0
AMP_FACTOR = 10.0 ** (-DB_REDUCTION / 20.0)  # ~0.7079


# ---------------------------------------------------------------------------
# Waveform functions
# ---------------------------------------------------------------------------
def square_wave(x: np.ndarray, n_terms: int) -> np.ndarray:
    wave = np.zeros_like(x, dtype=np.float64)
    for i in range(1, n_terms + 1):
        k = 2 * i - 1
        wave += np.sin(k * x) / k
    return (4.0 / np.pi) * wave


def sawtooth_wave(x: np.ndarray, n_terms: int) -> np.ndarray:
    wave = np.zeros_like(x, dtype=np.float64)
    for i in range(1, n_terms + 1):
        wave += (-1.0) ** (i + 1) * np.sin(i * x) / i
    return (2.0 / np.pi) * wave


# ---------------------------------------------------------------------------
# Blended waveform
# ---------------------------------------------------------------------------
def blended_waveform(x: np.ndarray, n_frac: float, waveform_fn) -> np.ndarray:
    n_lo = int(np.floor(n_frac))
    n_hi = int(np.ceil(n_frac))
    frac = n_frac - n_lo

    if n_lo == n_hi or n_hi > NUM_TERMS_MAX:
        return waveform_fn(x, n_hi)

    wave_lo = waveform_fn(x, n_lo)
    wave_hi = waveform_fn(x, n_hi)
    return wave_lo + frac * (wave_hi - wave_lo)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
def build_n_terms_schedule(num_frames: int = NUM_FRAMES,
                           hold: int = HOLD_FRAMES) -> list[float]:
    schedule = []
    for _ in range(hold):
        schedule.append(float(NUM_TERMS_MIN))
    anim_frames = num_frames - hold
    for i in range(anim_frames):
        fraction = i / (anim_frames - 1) if anim_frames > 1 else 0.0
        n_frac = NUM_TERMS_MIN + fraction * (NUM_TERMS_MAX - NUM_TERMS_MIN)
        schedule.append(n_frac)
    return schedule


# ---------------------------------------------------------------------------
# Audio generation
# ---------------------------------------------------------------------------
def generate_audio(waveform_fn, n_terms_per_frame: list[float]) -> np.ndarray:
    """Generate audio by repeating the waveform at CARRIER_FREQ Hz.

    The waveform at the current n is computed over one period (2π),
    then played out at CARRIER_FREQ times per second.  The waveform
    *is* the audio — no carrier sine modulation.
    """
    samples_per_frame = SAMPLING_RATE // FPS
    total_samples = NUM_FRAMES * samples_per_frame
    samples_per_cycle = SAMPLING_RATE / CARRIER_FREQ
    audio = np.zeros(total_samples, dtype=np.float64)

    for frame_idx in range(NUM_FRAMES):
        n_frac = n_terms_per_frame[frame_idx]
        frame_start = frame_idx * samples_per_frame
        frame_samples = samples_per_frame

        # Waveform over one period at 4x oversampling for smooth interp.
        x_period = np.linspace(0, 2 * np.pi, int(samples_per_cycle) * 4)
        wave = blended_waveform(x_period, n_frac, waveform_fn)
        wave = wave / 1.3

        # Phase advance per audio sample: waveform repeats at CARRIER_FREQ.
        phase_per_sample = 2.0 * np.pi * CARRIER_FREQ / SAMPLING_RATE
        global_phase = frame_start * phase_per_sample
        frame_phase = (global_phase + np.arange(frame_samples, dtype=np.float64) * phase_per_sample) % (2.0 * np.pi)

        audio[frame_start : frame_start + frame_samples] = (
            np.interp(frame_phase, x_period, wave,
                      left=wave[0], right=wave[-1])
        )

    # Apply -3 dB gain reduction
    audio *= AMP_FACTOR

    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val

    return audio


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    n_terms_schedule = build_n_terms_schedule(NUM_FRAMES, HOLD_FRAMES)

    waveforms = [
        (square_wave, "Square"),
        (sawtooth_wave, "Sawtooth"),
    ]

    for waveform_fn, wave_type in waveforms:
        print(f"Generating {wave_type} audio ...")
        audio = generate_audio(waveform_fn, n_terms_schedule)

        out_path = os.path.join(
            RENDER_DIR,
            f"{wave_type.lower().replace(' ', '_')}_wave_audio.wav"
        )

        wavfile.write(out_path, SAMPLING_RATE,
                      (audio * 32767.0).astype(np.int16))
        print(f"  -> {out_path}  ({audio.shape[0]} samples, "
              f"{audio.shape[0] / SAMPLING_RATE:.1f} s)")

    print("Done.")


if __name__ == "__main__":
    main()
