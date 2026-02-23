# NTSC Composite Simulator

Simulate the NTSC composite video signal encoding and decoding pipeline. Process video or images through an accurate analog signal path to reproduce the characteristic artifacts of NTSC television — color bleeding, rainbow cross-color, dot crawl, and chroma/luma bandwidth limitations.

Available in both **Rust** (recommended) and **Python**. The Rust version offers significantly faster processing with multithreaded parallelism via rayon.

## Examples

| Source | NTSC Roundtrip | Degraded Signal |
|:---:|:---:|:---:|
| ![Source](examples/colorbars_source.png) | ![NTSC](examples/colorbars_ntsc.png) | ![Degraded](examples/colorbars_degraded.png) |
| Clean SMPTE color bars | After encode/decode roundtrip | With noise, ghosting, attenuation, and jitter |

## Features

- **Roundtrip** video through the full encode/decode pipeline in memory
- **Telecine** simulation with 3:2 pulldown (24p film → 480i interlaced)
- **Image** processing through the NTSC signal path
- **SMPTE color bars** test pattern generator
- **Physically-motivated signal degradation**: noise (snow), multipath ghosting, attenuation, horizontal jitter
- **Realistic multipath ghosting model**: multiple reflections, phase shift (polarity inversion), HF rolloff, sub-sample delay interpolation, dynamic amplitude modulation
- **Two comb filter modes**: horizontal 2-sample delay (default) and 1H line-delay
- Parallel frame processing (rayon in Rust, multiprocessing in Python)

## Building (Rust)

Requires Rust 1.70+ and [ffmpeg](https://ffmpeg.org/) on your PATH (for video I/O).

```bash
cd rust
cargo build --release
```

The binary is at `rust/target/release/ntsc-composite-simulator` (`.exe` on Windows).

## Usage

### Roundtrip video through NTSC

```bash
ntsc-composite-simulator roundtrip input.mp4 -o output.mp4
```

With telecine (3:2 pulldown, interlaced 480i output):

```bash
ntsc-composite-simulator roundtrip input.mp4 -o output.mp4 --telecine
```

### Process a single image

```bash
ntsc-composite-simulator image photo.png -o ntsc_photo.png
```

### Generate SMPTE color bars

```bash
ntsc-composite-simulator colorbars -o colorbars.png --save-source source.png
```

### Batch processing

Pass a directory as input and a directory as output:

```bash
ntsc-composite-simulator image ./input_frames/ -o ./output_frames/
ntsc-composite-simulator roundtrip ./videos/ -o ./processed/
```

### Simulate weak signal reception

```bash
# Subtle snow
ntsc-composite-simulator image photo.png -o noisy.png --noise 0.05

# Moderate degradation — all effects combined
ntsc-composite-simulator roundtrip input.mp4 -o degraded.mp4 \
  --noise 0.05 --ghost 0.15 --attenuation 0.1 --jitter 0.5
```

### Realistic multipath ghosting

The ghosting model simulates real multipath propagation with multiple reflections, phase shifts, HF rolloff, and optional dynamic amplitude variation.

```bash
# Single inverted ghost off a nearby building
ntsc-composite-simulator roundtrip input.mp4 -o output.mp4 \
  --ghost 0.25 --ghost-delay 1.5 --ghost-phase 180

# Multiple reflections with dynamic amplitude drift
ntsc-composite-simulator roundtrip input.mp4 -o output.mp4 \
  --ghost 0.25 --ghost-delay 1.5 --ghost-phase 180 --ghost-dynamic \
  --ghost-multi "0.10,4.2,0;0.05,8.0,180"

# Subtle, barely-visible ghosting
ntsc-composite-simulator roundtrip input.mp4 -o output.mp4 \
  --ghost 0.06 --ghost-delay 2.0 \
  --ghost-multi "0.03,4.5,0;0.015,8.0,180"

# Heavy distant-antenna reception with soft ghosts and snow
ntsc-composite-simulator roundtrip input.mp4 -o output.mp4 \
  --ghost 0.35 --ghost-delay 2.0 --ghost-phase 180 --ghost-dynamic \
  --ghost-multi "0.20,5.0,0;0.12,9.5,180;0.06,15.0,0" \
  --ghost-rolloff-mhz 2.5 --noise 0.04
```

The `--ghost-multi` flag accepts semicolon-separated ghosts, each as `amplitude,delay_us,phase_deg`:

| Parameter | Meaning |
|---|---|
| `amplitude` | Ghost strength 0–1 (relative to direct signal) |
| `delay_us` | Propagation delay in microseconds |
| `phase_deg` | Phase rotation in degrees (0 = in-phase, 180 = polarity inversion) |

## Options Reference

### General

| Flag | Commands | Description |
|---|---|---|
| `-o, --output` | all | Output file path |
| `--width` | roundtrip, image, colorbars | Output width (default: 640) |
| `--height` | roundtrip, image, colorbars | Output height (default: 480) |
| `--comb-1h` | roundtrip, image, colorbars | Use 1H line-delay comb filter instead of 2-sample delay |
| `--save-source` | colorbars | Save the source SMPTE pattern as PNG |

### Video encoding

| Flag | Commands | Description |
|---|---|---|
| `--telecine` | roundtrip | Enable 3:2 pulldown telecine |
| `--crf` | roundtrip | x264 CRF quality, 0=lossless, 51=worst (default: 17) |
| `--preset` | roundtrip | x264 encoding preset, e.g. `ultrafast`, `fast`, `slow` (default: fast) |
| `--lossless` | roundtrip | Lossless output (FFV1 for .mkv, x264 QP 0 for .mp4) |
| `--threads` | roundtrip | Number of parallel worker threads (default: all logical cores) |

### Signal degradation

| Flag | Commands | Description |
|---|---|---|
| `--noise` | roundtrip, image, colorbars | Snow amplitude (0.05 = subtle, 0.2 = heavy) |
| `--attenuation` | roundtrip, image, colorbars | Signal attenuation 0–1 (0 = none, 1 = flat at blanking) |
| `--jitter` | roundtrip, image, colorbars | Horizontal jitter std dev in subcarrier cycles |

### Ghosting (multipath)

| Flag | Commands | Description |
|---|---|---|
| `--ghost` | roundtrip, image, colorbars | Primary ghost amplitude 0–1 |
| `--ghost-delay` | roundtrip, image, colorbars | Primary ghost delay in µs (default: 2.0) |
| `--ghost-phase` | roundtrip, image, colorbars | Phase shift in degrees; 180 = polarity inversion (default: 0) |
| `--ghost-rolloff-mhz` | roundtrip, image, colorbars | HF rolloff cutoff in MHz (default: 3.0) — lower = softer ghost |
| `--ghost-dynamic` | roundtrip, image, colorbars | Enable slow amplitude modulation (environmental drift) |
| `--ghost-dynamic-rate` | roundtrip, image, colorbars | Dynamic modulation rate in Hz (default: 0.5) |
| `--ghost-multi` | roundtrip, image, colorbars | Additional ghosts as `"amp,delay,phase;..."` triples |

Run `ntsc-composite-simulator <command> -h` for full details.

## Python Version

The Python version supports the same core pipeline plus raw signal export (`encode`/`decode` commands, `.npy` and `.wav` output).

### Requirements

- Python 3, [ffmpeg](https://ffmpeg.org/) (optional)
- `pip install -r requirements.txt` (numpy, scipy, opencv-python, tqdm)

### Commands

```bash
python main.py roundtrip input.mp4 -o output.mp4
python main.py roundtrip input.mp4 -o output.mp4 --telecine
python main.py image photo.png -o ntsc_photo.png
python main.py colorbars -o colorbars.npy --save-source bars.png
python main.py encode input.mp4 -o signal.npy
python main.py decode signal.npy -o output.mp4 --width 640 --height 480
```

Additional Python-only flags: `--signal` (export composite as `.npy`), `--wav` (export as WAV).

Run `python main.py <command> -h` for details.

## How It Works

### Encoding

1. Convert RGB to YIQ color space
2. Bandwidth-limit luma (4.2 MHz), I (1.5 MHz), and Q (0.5 MHz) channels with FIR filters
3. Modulate I/Q onto the 3.58 MHz color subcarrier
4. Build the full 525-line NTSC frame structure with sync, blanking, and colorburst
5. Apply vestigial sideband filtering

### Decoding

1. Separate luma and chroma using a comb filter
2. Detect burst phase from the colorburst reference
3. Demodulate I and Q via product detection
4. Low-pass filter the recovered chroma channels
5. Convert YIQ back to RGB

### Ghosting Model

The multipath ghosting simulation models real-world signal reflections:

1. **Multiple reflections** — each ghost is an independent delayed copy of the direct signal, not derived from already-ghosted data
2. **Sub-sample delay** — linear interpolation between adjacent samples for precise delay positioning
3. **HF rolloff** — single-pole IIR lowpass per ghost simulates high-frequency absorption by reflective surfaces
4. **Phase shift** — cos(φ) gain rotation; exact at 0° (in-phase) and 180° (polarity inversion, the most common real-world cases)
5. **Dynamic amplitude** — sum-of-sinusoids LFO envelope simulates environmental variation (swaying foliage, passing vehicles)

### Signal Specifications

| Parameter | Value |
|---|---|
| Sample rate | 14,318,180 Hz (4 × F_SC) |
| Color subcarrier | 3,579,545 Hz |
| Frame rate | 29.97 fps |
| Lines per frame | 525 (480 visible) |
| Samples per line | 910 |

## Comb Filter Modes

- **2-sample delay** (default): Cancels chroma from luma using a horizontal delay. Can produce rainbow cross-color on sharp edges.
- **1H line-delay** (`--comb-1h`): Uses the adjacent line from the same field. Reduces rainbow artifacts but introduces hanging-dot patterns on horizontal color transitions as well as some line doubling.
