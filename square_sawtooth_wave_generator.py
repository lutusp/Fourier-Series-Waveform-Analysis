#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Square and Sawtooth Waveform Video Generator

Generates two animated videos showing the gradual Fourier series evolution
of a square wave and a sawtooth wave, each with an accompanying sound track
that uses the evolving waveform to modulate a 440 Hz carrier signal.

Each video is 13 seconds long (390 frames at 30 fps). The first 3 seconds
(90 frames) hold n = 1 so the viewer can see the starting waveform. Then
the series animates from n = 1 to n = 32 over the remaining 300 frames.
Each frame renders two full cycles of the current waveform using matplotlib.

For each frame we compute a fractional term count (linearly interpolated
between 1 and 32 across the animation portion), then blend the waveforms
at floor(n_frac) and ceil(n_frac) so the audio waveform shape matches the
blended waveform shown on screen.
"""

import os
import sys
import subprocess
import tempfile
import shutil

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HOLD_FRAMES = 90  # 3 seconds of n=1 at start (90 / 30 = 3 s)
ANIM_FRAMES = 300  # frames over which the series animates
END_HOLD_FRAMES = 90  # 3 seconds of n=128 at the end (90 / 30 = 3 s)
NUM_FRAMES = HOLD_FRAMES + ANIM_FRAMES + END_HOLD_FRAMES  # total = 480 frames (16 s)
NUM_TERMS_MIN = 1
NUM_TERMS_MAX = 128
FPS = 30
CARRIER_FREQ = 110.0  # Hz
SAMPLING_RATE = 44100  # Hz (audio)
TWO_CYCLES = 4 * np.pi  # domain: [0, 4pi] = two full cycles of fundamental
DB_REDUCTION = 6.0  # dB gain reduction for audio
AMP_FACTOR = 10.0 ** (-DB_REDUCTION / 20.0)  # ≈ 0.5012 for 6 dB

# Output directory
RENDER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders")
os.makedirs(RENDER_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Waveform functions (Fourier series)
# ---------------------------------------------------------------------------
def square_wave(x: np.ndarray, n_terms: int) -> np.ndarray:
    """Generate a square wave approximation using the Fourier series.

    The square wave is the sum of odd harmonics:
        (4/pi) * sum_{k odd} sin(k*x) / k

    Parameters
    ----------
    x : np.ndarray
        Domain points.
    n_terms : int
        Number of terms in the series (counts odd harmonics only).

    Returns
    -------
    np.ndarray
        Square wave approximation values.
    """
    wave = np.zeros_like(x, dtype=np.float64)
    for i in range(1, n_terms + 1):
        k = 2 * i - 1  # odd harmonic index: 1, 3, 5, ...
        wave += np.sin(k * x) / k
    return (4.0 / np.pi) * wave


def sawtooth_wave(x: np.ndarray, n_terms: int) -> np.ndarray:
    """Generate a sawtooth wave approximation using the Fourier series.

    The sawtooth wave is the sum of all harmonics with 1/k decay:
        (2/pi) * sum_{k=1}^{N} (-1)^{k+1} * sin(k*x) / k

    Parameters
    ----------
    x : np.ndarray
        Domain points.
    n_terms : int
        Number of terms in the series.

    Returns
    -------
    np.ndarray
        Sawtooth wave approximation values.
    """
    wave = np.zeros_like(x, dtype=np.float64)
    for i in range(1, n_terms + 1):
        wave += (-1.0) ** (i + 1) * np.sin(i * x) / i
    return (2.0 / np.pi) * wave


# ---------------------------------------------------------------------------
# Blended waveform (fractional term interpolation)
# ---------------------------------------------------------------------------
def blended_waveform(
    x: np.ndarray,
    n_frac: float,
    waveform_fn,
) -> np.ndarray:
    """Compute a waveform blended between floor(n) and ceil(n) terms.

    When n_frac is an integer, returns the waveform for that exact term
    count.  Otherwise, linearly interpolates between the waveform at
    floor(n_frac) and ceil(n_frac) using the fractional remainder.

    Parameters
    ----------
    x : np.ndarray
        Domain points.
    n_frac : float
        Fractional number of terms (e.g. 15.7).
    waveform_fn : callable
        One of square_wave or sawtooth_wave.

    Returns
    -------
    np.ndarray
        Blended waveform values.
    """
    n_lo = int(np.floor(n_frac))
    n_hi = int(np.ceil(n_frac))
    frac = n_frac - n_lo

    if n_lo == n_hi or n_hi > NUM_TERMS_MAX:
        # At the boundary or exact integer, just use n_hi (which == n_lo)
        return waveform_fn(x, n_hi)

    wave_lo = waveform_fn(x, n_lo)
    wave_hi = waveform_fn(x, n_hi)
    return wave_lo + frac * (wave_hi - wave_lo)


# ---------------------------------------------------------------------------
# Audio synthesis
# ---------------------------------------------------------------------------
def generate_audio(
    waveform_fn,
    n_terms_per_frame: list[float],
    num_frames: int = NUM_FRAMES,
    fps: int = FPS,
    carrier_freq: float = CARRIER_FREQ,
    sampling_rate: int = SAMPLING_RATE,
) -> np.ndarray:
    """Synthesize an audio track from the waveform itself.

    The Fourier-series waveform at the current n is computed over one
    period (2π), repeated at the audio sample rate so it covers exactly
    one cycle of the carrier frequency.  The resulting samples are
    repeated ``carrier_freq`` times per second — that is, the waveform
    *is* the audio, played out at the target frequency.

    Parameters
    ----------
    waveform_fn : callable
        One of square_wave or sawtooth_wave.
    n_terms_per_frame : list[float]
        Fractional n-terms for each frame.
    num_frames : int
        Total number of frames.
    fps : int
        Frames per second.
    carrier_freq : float
        Frequency at which the waveform repeats (Hz).
    sampling_rate : int
        Audio sample rate in Hz.

    Returns
    -------
    np.ndarray
        Audio samples (float64).
    """
    samples_per_frame = sampling_rate // fps
    total_samples = num_frames * samples_per_frame
    audio = np.zeros(total_samples, dtype=np.float64)

    # Exact number of audio samples per waveform cycle.
    # 44100 / 440 = 100.227…, so we use the exact float value and
    # generate the full duration as one continuous phase sweep.
    samples_per_cycle = sampling_rate / carrier_freq

    for frame_idx in range(num_frames):
        n_frac = n_terms_per_frame[frame_idx]
        frame_start = frame_idx * samples_per_frame
        frame_samples = samples_per_frame

        # Compute blended waveform over one period (2π) at a
        # resolution that matches the audio sample density.
        # We use enough points so interpolation is smooth.
        x_period = np.linspace(0, 2 * np.pi, int(samples_per_cycle) * 4)
        wave = blended_waveform(x_period, n_frac, waveform_fn)
        wave = wave / 1.3

        # Phase advance per audio sample: waveform repeats at carrier_freq
        # Hz.  Phase increment = 2π × carrier_freq / sampling_rate.
        phase_per_sample = 2.0 * np.pi * carrier_freq / sampling_rate
        global_phase = frame_start * phase_per_sample
        frame_phase = (
            global_phase + np.arange(frame_samples, dtype=np.float64) * phase_per_sample
        ) % (2.0 * np.pi)

        # Interpolate waveform onto the audio sample grid.
        audio[frame_start : frame_start + frame_samples] = np.interp(
            frame_phase, x_period, wave, left=wave[0], right=wave[-1]
        )

    # Normalize to prevent clipping before applying gain reduction
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val

    # Apply gain reduction
    audio *= AMP_FACTOR  # 10^(-DB_REDUCTION/20)

    return audio


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------
def render_frame(
    x: np.ndarray,
    wave: np.ndarray,
    n_terms: float,
    wave_type: str,
    output_path: str,
) -> None:
    """Render a single frame of the waveform animation.

    Parameters
    ----------
    x : np.ndarray
        Domain points.
    wave : np.ndarray
        Waveform values at this frame (already blended).
    n_terms : float
        Current fractional number of Fourier terms (for title).
    wave_type : str
        "Square" or "Sawtooth" for display.
    output_path : str
        Where to save the PNG file.
    """
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)

        # Plot the current blended waveform — x in radians → ms
        time_ms = x / (2.0 * np.pi) * (1000.0 / CARRIER_FREQ)
        ax.plot(
            time_ms, wave, color="#0080ff", linewidth=1.0, label=f"n = {n_terms:.1f}"
        )

        # Add grid and labels
        ax.set_xlabel("time (ms)", fontsize=24)
        ax.set_ylabel("Amplitude", fontsize=24)
        ax.set_title(
            f"{wave_type} Wave — Fourier Series Approximation, {int(CARRIER_FREQ)}Hz",
            fontsize=28,
        )
        # 1 second = 1000 ms, 1/CARRIER_FREQ = period; show two cycles (2 periods) in ms
        period_ms = 1000.0 / CARRIER_FREQ
        two_periods_ms = 2.0 * period_ms
        ax.set_xlim(0, two_periods_ms)
        ax.set_xticks(np.linspace(0, two_periods_ms, 5))
        ax.set_xticklabels([f"{v:.2f}" for v in np.linspace(0, two_periods_ms, 5)])
        ax.set_ylim(-1.4, 1.4)
        ax.tick_params(labelsize=22)
        ax.legend(loc="upper right", fontsize=22)
        ax.grid(True, alpha=0.4, color="#00FF00")
        ax.axhline(0, color="white", linewidth=1.0)

        plt.tight_layout()
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        fig.clf()


# ---------------------------------------------------------------------------
# Build fractional schedule: hold n=1 for HOLD_FRAMES, then animate
# ---------------------------------------------------------------------------
def build_n_terms_schedule(
    num_frames: int = NUM_FRAMES,
    hold: int = HOLD_FRAMES,
    end_hold: int = END_HOLD_FRAMES,
) -> list[float]:
    """Create a schedule of fractional n-terms for each frame.

    The first ``hold`` frames stay at n = 1.  The middle frames
    linearly interpolate from 1.0 to NUM_TERMS_MAX.  The last
    ``end_hold`` frames hold at n = NUM_TERMS_MAX.

    Parameters
    ----------
    num_frames : int
        Total number of frames (hold + animation + end hold).
    hold : int
        Number of frames to hold at n = 1.
    end_hold : int
        Number of frames to hold at n = NUM_TERMS_MAX.

    Returns
    -------
    list[float]
        Fractional n-terms value for each frame.
    """
    schedule = []
    # Hold phase: n = 1.0 for the first `hold` frames
    for _ in range(hold):
        schedule.append(float(NUM_TERMS_MIN))
    # Animation phase: quadratic ramp.  At fraction t within the
    # animation, n = 1 + (NUM_TERMS_MAX - 1) * t²  so the term count
    # accelerates from 1 up to NUM_TERMS_MAX.
    anim_frames = num_frames - hold - end_hold
    for i in range(anim_frames):
        fraction = i / (anim_frames - 1) if anim_frames > 1 else 0.0
        n_frac = NUM_TERMS_MIN + (NUM_TERMS_MAX - NUM_TERMS_MIN) * fraction * fraction
        schedule.append(n_frac)
    # End hold phase: n = NUM_TERMS_MAX for the last `end_hold` frames
    for _ in range(end_hold):
        schedule.append(float(NUM_TERMS_MAX))
    return schedule


# ---------------------------------------------------------------------------
# Video composition via ffmpeg
# ---------------------------------------------------------------------------
def images_to_video(
    image_dir: str,
    output_path: str,
    fps: int = FPS,
) -> None:
    """Convert a sequence of PNG images into an MP4 video using ffmpeg.

    Parameters
    ----------
    image_dir : str
        Directory containing numbered PNG files (frame_00000.png, …).
    output_path : str
        Output MP4 file path.
    fps : int
        Frames per second.
    """
    # Find frame count and sort numerically
    png_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))
    num_images = len(png_files)
    if num_images == 0:
        raise ValueError(f"No PNG images found in {image_dir}")

    # Determine the zero-padding width from the first filename
    first_file = png_files[0]
    digits = len(first_file.split("_")[-1].replace(".png", ""))

    # ffmpeg -i pattern: frame_%05d.png  (5-digit zero-padded counter)
    pattern = f"frame_%0{digits}d.png"
    input_path = os.path.join(image_dir, pattern)

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=1920:1080",
        "-loglevel",
        "error",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (rc={result.returncode}):\n{result.stderr}")


def video_and_audio_to_final(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> None:
    """Merge a video with an audio track using ffmpeg.

    Parameters
    ----------
    video_path : str
        Path to the silent MP4 video.
    audio_path : str
        Path to the WAV audio file.
    output_path : str
        Final output MP4 with audio.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Frequency spectrum rendering
# ---------------------------------------------------------------------------
def compute_frequency_spectrum(
    waveform_fn,
    n_terms: float,
    fundamental_freq: float = CARRIER_FREQ,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the frequency spectrum (amplitude vs harmonic index) for a waveform.

    Uses a moving-average window to blend harmonics near the cutoff
    (n_terms) so the spectrum fades in gradually rather than jumping
    when the integer term count changes.

    Parameters
    ----------
    waveform_fn : callable
        One of square_wave or sawtooth_wave.
    n_terms : float
        Fractional number of terms (e.g. 15.7). Harmonics at or below
        floor(n_terms) show their full amplitude. Harmonics between
        floor(n_terms) and ceil(n_terms) are blended linearly. Harmonics
        above ceil(n_terms) are zero.
    fundamental_freq : float
        The fundamental frequency in Hz (carrier_freq).

    Returns
    -------
    harmonics : np.ndarray
        Harmonic indices (1, 2, 3, …, ceil(n_terms)).
    amplitudes : np.ndarray
        Absolute amplitude of each harmonic component, blended near the cutoff.
    """
    n_lo = int(np.floor(n_terms))
    n_hi = int(np.ceil(n_terms))
    frac = n_terms - n_lo
    n_max = min(n_hi, NUM_TERMS_MAX)

    harmonics = np.arange(1, n_max + 1, dtype=np.float64)
    amplitudes = np.zeros(n_max, dtype=np.float64)

    # Full theoretical amplitudes for each harmonic
    full_amp = np.zeros(n_max, dtype=np.float64)
    for i in range(1, n_max + 1):
        k = i
        if waveform_fn == square_wave:
            if k % 2 == 1:  # odd harmonic
                full_amp[i - 1] = 4.0 / np.pi * (1.0 / k)
            else:
                full_amp[i - 1] = 0.0
        elif waveform_fn == sawtooth_wave:
            full_amp[i - 1] = 2.0 / np.pi * abs((-1.0) ** (i + 1) / i)

    # Apply windowing: harmonics up to n_lo get full weight,
    # harmonics between n_lo and n_hi get blended weight,
    # harmonics above n_hi are zero.
    for i in range(1, n_max + 1):
        k = i
        if k <= n_lo:
            amplitudes[i - 1] = full_amp[i - 1]
        elif k > n_hi:
            amplitudes[i - 1] = 0.0
        else:
            # Blend between 0 and full amplitude for the boundary harmonic
            amplitudes[i - 1] = full_amp[i - 1] * frac

    # Apply a running-average smoothing window of width 2 * n_lo + 1
    # around the cutoff to further soften the transition.
    if n_lo >= 1:
        window_width = min(2 * n_lo + 1, n_max)
        # Use a triangular (Bartlett) window centered on n_lo
        center = n_lo
        weights = np.zeros(n_max)
        for i in range(1, n_max + 1):
            dist = abs(i - center)
            if dist <= window_width // 2:
                weights[i - 1] = 1.0 - 2.0 * dist / (window_width + 1)
        # Normalize weights so the sum is preserved
        w_sum = np.sum(weights)
        if w_sum > 0:
            # weighted_amp = amplitudes * weights
            # Renormalize: replace only the tail region with weighted values
            tail_start = min(n_lo + 1, n_max)
            if tail_start < n_max:
                # Blend tail with neighbors
                smoothed = amplitudes.copy()
                for i in range(tail_start, n_max):
                    lo = max(tail_start - 1, 0)
                    hi = min(i + 1, n_max - 1)
                    smoothed[i] = (
                        amplitudes[lo] + 2 * amplitudes[i] + amplitudes[hi]
                    ) / 4.0
                amplitudes = smoothed

    freqs = harmonics * fundamental_freq
    return freqs, amplitudes


def render_frequency_frame(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    n_terms: float,
    wave_type: str,
    output_path: str,
) -> None:
    """Render a single frequency spectrum frame.

    Parameters
    ----------
    freqs : np.ndarray
        Frequencies of each harmonic in Hz.
    amplitudes : np.ndarray
        Amplitude of each harmonic.
    n_terms : float
        Current fractional number of Fourier terms (for title).
    wave_type : str
        "Square" or "Sawtooth" for display.
    output_path : str
        Where to save the PNG file.
    """
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)

        # Stem plot for discrete frequency components
        ax.stem(
            freqs,
            amplitudes,
            linefmt="#0080ff",
            markerfmt="o",
            basefmt="#888888",
            label=f"n = {n_terms:.1f}",
        )

        # Labels and title
        ax.set_xlabel("Frequency (Hz)", fontsize=24)
        ax.set_ylabel("Amplitude", fontsize=24)
        ax.set_title(
            f"{wave_type} Wave — Frequency Spectrum, {int(CARRIER_FREQ)}Hz Fundamental",
            fontsize=28,
        )
        ax.set_xlim(0, 4000)
        ax.set_ylim(-0.1, max(amplitudes) * 1.15)
        ax.tick_params(labelsize=22)
        ax.legend(loc="upper right", fontsize=22)
        ax.grid(True, alpha=0.4, color="#00FF00")

        plt.tight_layout()
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        fig.clf()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def generate_waveform_video(
    waveform_fn,
    wave_type: str,
) -> str:
    """Generate a complete waveform animation video with audio.

    Parameters
    ----------
    waveform_fn : callable
        One of square_wave or sawtooth_wave.
    wave_type : str
        Display name: "Square" or "Sawtooth".

    Returns
    -------
    str
        Path to the final output MP4 file.
    """
    # Fractional n-terms schedule: hold at 1.0, animate 1.0 → 32.0,
    # then hold at 32.0
    n_terms_schedule = build_n_terms_schedule(NUM_FRAMES, HOLD_FRAMES, END_HOLD_FRAMES)

    # Domain for two full cycles of the fundamental
    x = np.linspace(0, TWO_CYCLES, 2000)

    # Create a temporary directory for frame images
    tmp_dir = tempfile.mkdtemp(prefix=f"{wave_type.lower()}_frames_")

    try:
        # Render each frame using the blended waveform
        for frame_idx in range(NUM_FRAMES):
            n_frac = n_terms_schedule[frame_idx]
            wave = blended_waveform(x, n_frac, waveform_fn)
            frame_path = os.path.join(tmp_dir, f"frame_{frame_idx:05d}.png")
            render_frame(x, wave, n_frac, wave_type, frame_path)
            pct = (frame_idx + 1) / NUM_FRAMES * 100
            print(f"Frame {frame_idx + 1} of {NUM_FRAMES}, {pct:.1f}%")
            sys.stdout.flush()

        # Build intermediate video (silent)
        silent_video = os.path.join(tmp_dir, "silent.mp4")
        images_to_video(tmp_dir, silent_video, fps=FPS)

        # Generate audio track using blended waveforms
        audio = generate_audio(
            waveform_fn,
            n_terms_schedule,
            num_frames=NUM_FRAMES,
            fps=FPS,
            carrier_freq=CARRIER_FREQ,
            sampling_rate=SAMPLING_RATE,
        )

        # Write audio to temporary WAV file
        audio_path = os.path.join(tmp_dir, "audio.wav")
        import scipy.io.wavfile as wavfile

        wavfile.write(audio_path, SAMPLING_RATE, (audio * 32767.0).astype(np.int16))

        # Combine video + audio
        output_path = os.path.join(
            RENDER_DIR, f"{wave_type.lower().replace(' ', '_')}_wave_time.mp4"
        )
        video_and_audio_to_final(silent_video, audio_path, output_path)

        print(f"  -> {output_path}")
        return output_path

    finally:
        # Clean up temporary directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


def generate_frequency_spectrum_video(
    waveform_fn,
    wave_type: str,
) -> str:
    """Generate a frequency spectrum animation video for a waveform.

    The video shows the harmonic amplitudes as a stem plot, with n
    animating from 1 to NUM_TERMS_MAX. The x-axis is frequency in Hz.

    Parameters
    ----------
    waveform_fn : callable
        One of square_wave or sawtooth_wave.
    wave_type : str
        Display name: "Square" or "Sawtooth".

    Returns
    -------
    str
        Path to the output MP4 file.
    """
    n_terms_schedule = build_n_terms_schedule(NUM_FRAMES, HOLD_FRAMES, END_HOLD_FRAMES)

    tmp_dir = tempfile.mkdtemp(prefix=f"{wave_type.lower()}_freq_frames_")

    try:
        for frame_idx in range(NUM_FRAMES):
            n_frac = n_terms_schedule[frame_idx]
            freqs, amplitudes = compute_frequency_spectrum(waveform_fn, n_frac)
            frame_path = os.path.join(tmp_dir, f"frame_{frame_idx:05d}.png")
            render_frequency_frame(freqs, amplitudes, n_frac, wave_type, frame_path)
            pct = (frame_idx + 1) / NUM_FRAMES * 100
            print(f"Frame {frame_idx + 1} of {NUM_FRAMES}, {pct:.1f}%")
            sys.stdout.flush()

        # Build intermediate video (silent)
        silent_video = os.path.join(tmp_dir, "silent.mp4")
        images_to_video(tmp_dir, silent_video, fps=FPS)

        # Generate audio track using blended waveforms
        audio = generate_audio(
            waveform_fn,
            n_terms_schedule,
            num_frames=NUM_FRAMES,
            fps=FPS,
            carrier_freq=CARRIER_FREQ,
            sampling_rate=SAMPLING_RATE,
        )

        # Write audio to temporary WAV file
        audio_path = os.path.join(tmp_dir, "audio.wav")
        import scipy.io.wavfile as wavfile

        wavfile.write(audio_path, SAMPLING_RATE, (audio * 32767.0).astype(np.int16))

        # Combine video + audio
        output_path = os.path.join(
            RENDER_DIR, f"{wave_type.lower().replace(' ', '_')}_wave_frequency.mp4"
        )
        video_and_audio_to_final(silent_video, audio_path, output_path)

        print(f"  -> {output_path}")
        return output_path

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    Path(RENDER_DIR).mkdir(parents=True, exist_ok=True)
    """Generate square and sawtooth waveform videos with audio and frequency spectra.

    Creates four MP4 files in the renders/ directory:
      - square_wave_time.mp4       time-domain waveform with audio
      - sawtooth_wave_time.mp4     time-domain waveform with audio
      - square_wave_frequency.mp4  frequency-domain harmonic bar chart with audio
      - sawtooth_wave_frequency.mp4 frequency-domain harmonic bar chart with audio

    The time-domain videos are 16 seconds (480 frames). The Fourier series
    animates from n = 1 to n = 128 over 300 frames.
    The spectrum videos animate the same way, showing how harmonic amplitudes
    are revealed as more terms are added.
    """
    print("Generating square wave video...")
    generate_waveform_video(square_wave, "Square")

    print("Generating sawtooth wave video...")
    generate_waveform_video(sawtooth_wave, "Sawtooth")

    print("Generating square wave frequency spectrum...")
    generate_frequency_spectrum_video(square_wave, "Square")

    print("Generating sawtooth wave frequency spectrum...")
    generate_frequency_spectrum_video(sawtooth_wave, "Sawtooth")

    print("Done. Videos saved in renders/")


if __name__ == "__main__":
    main()
