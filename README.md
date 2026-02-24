# NTSC Composite Simulator

Simulate the NTSC composite video signal encoding and decoding pipeline. Process video or images through an accurate analog signal path to reproduce the characteristic artifacts of NTSC television — color bleeding, rainbow cross-color, dot crawl, and chroma/luma bandwidth limitations.

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
- **VHS tape-path simulation**: luma bandwidth limiting, color-under chroma processing, tape dropouts, edge ringing, luminance-dependent noise, head-wear smearing, and causal rightward trailing
- **Range parameters**: all numeric effect arguments accept `min..max` ranges — a random value is sampled per file in batch mode for natural variation
- **Two comb filter modes**: horizontal 2-sample delay (default) and 1H line-delay
- Parallel frame processing via rayon

## Building

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

### Randomized ranges (batch variation)

All numeric effect arguments accept a `min..max` range. When processing a batch of files, each file gets a fresh random sample from that range, giving natural variation across the batch.

```bash
# Each image gets a different noise level between 0.02 and 0.15
ntsc-composite-simulator image ./frames/ -o ./out/ --noise 0.02..0.15

# Vary multiple effects at once
ntsc-composite-simulator roundtrip ./videos/ -o ./processed/ \
  --noise 0.03..0.10 --ghost 0.1..0.4 --ghost-delay 1.0..5.0 \
  --attenuation 0.05..0.2 --jitter 0.2..0.8

# VHS batch with randomized wear
ntsc-composite-simulator roundtrip ./videos/ -o ./processed/ \
  --vhs-luma-bw 1.6..3.0 --color-under-bw 250..450 \
  --luma-noise 0.02..0.08 --head-smear 0.3..0.7 --tape-trail 0.2..0.6
```

A single value (e.g. `--noise 0.05`) works exactly as before. When ranges are used, the sampled values are logged to stderr for reproducibility.

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

# Heavy distant-antenna reception with soft ghosts and snow
ntsc-composite-simulator roundtrip input.mp4 -o output.mp4 \
  --ghost 0.35 --ghost-delay 2.0 --ghost-phase 180 --ghost-dynamic \
  --ghost-multi "0.20,5.0,0;0.12,9.5,180;0.06,15.0,0" \
  --ghost-rolloff-mhz 2.5 --noise 0.04
```

The `--ghost-multi` flag accepts semicolon-separated ghosts, each as `amplitude,delay_us,phase_deg`. Each value can also be a `min..max` range (e.g. `--ghost-multi "0.1..0.3,2.0..5.0,0..180"`):

| Parameter | Meaning |
|---|---|
| `amplitude` | Ghost strength 0–1 (relative to direct signal) |
| `delay_us` | Propagation delay in microseconds |
| `phase_deg` | Phase rotation in degrees (0 = in-phase, 180 = polarity inversion) |

### VHS tape simulation

```bash
# Standard SP-quality VHS dub
ntsc-composite-simulator roundtrip input.mp4 -o output.mp4 \
  --vhs-luma-bw 3.0 --color-under-bw 400 --luma-noise 0.04 \
  --edge-ringing 1.0

# Worn EP-mode tape with dropouts, head smear, and trailing
ntsc-composite-simulator roundtrip input.mp4 -o output.mp4 \
  --vhs-luma-bw 1.6 --color-under-bw 300 --luma-noise 0.06 \
  --tape-dropout-rate 8 --head-smear 0.5 --tape-trail 0.4

# Heavily degraded multi-generation dub
ntsc-composite-simulator roundtrip input.mp4 -o output.mp4 \
  --vhs-luma-bw 1.6 --color-under-bw 250 --luma-noise 0.08 \
  --edge-ringing 1.5 --tape-dropout-rate 15 \
  --head-smear 0.7 --tape-trail 0.7 --noise 0.03
```

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

### Signal degradation (reception)

All numeric flags below accept either a single value (e.g. `0.05`) or a `min..max` range (e.g. `0.02..0.10`). With a range, each file in a batch gets a randomly sampled value.

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

### VHS tape-path effects

| Flag | Commands | Description |
|---|---|---|
| `--vhs-luma-bw` | roundtrip, image, colorbars | Luma bandwidth in MHz (SP ≈ 3.0, EP/SLP ≈ 1.6) |
| `--color-under-bw` | roundtrip, image, colorbars | Color-under chroma bandwidth in kHz (typical 300–500) |
| `--edge-ringing` | roundtrip, image, colorbars | Playback peaking / ringing gain (0.5–3.0) |
| `--luma-noise` | roundtrip, image, colorbars | Luminance-dependent tape noise amplitude (0.02–0.10) |
| `--tape-dropout-rate` | roundtrip, image, colorbars | Average dropouts per frame (2–20) |
| `--tape-dropout-len` | roundtrip, image, colorbars | Average dropout length in µs (default: 15.0) |
| `--head-smear` | roundtrip, image, colorbars | Worn-head symmetric luma blur (0.3–1.0); varies in bands across the frame |
| `--tape-trail` | roundtrip, image, colorbars | Causal rightward luma smear (0.3–1.0); the classic worn-tape trailing effect |

Run `ntsc-composite-simulator <command> -h` for full details.

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

### VHS Tape-Path Model

The VHS simulation models both the recording and playback signal paths:

1. **Luma bandwidth** — FIR low-pass limits luma resolution, matching the FM deviation limits of the tape format (SP vs EP/SLP)
2. **Color-under** — chroma is heterodyned down to 629 kHz, bandwidth-limited, and heterodyned back up, severely restricting color detail
3. **Edge ringing** — unsharp-mask peaking with intentional Gibbs-phenomenon overshoot simulates the sharpness-enhancement circuit in playback decks
4. **Luminance-dependent noise** — band-limited noise scaled by signal darkness (darker = noisier), matching real FM tape noise characteristics
5. **Tape dropouts** — random horizontal streaks from oxide damage; short ones flash white, longer ones trigger the dropout compensator (previous-line hold)
6. **Head-wear smear** — variable-width symmetric box blur on luma only, modulated in slow bands across the frame to simulate uneven head-to-tape contact
7. **Tape trailing** — causal one-pole IIR on luma only, applied as the final processing step so everything on the tape (noise, dropouts, detail) gets the characteristic rightward decay tail of worn playback electronics

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
