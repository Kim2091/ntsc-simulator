# NTSC Composite Simulator

Simulate the NTSC composite video signal encoding and decoding pipeline. Process video or images through an accurate analog signal path to reproduce the characteristic artifacts of NTSC television — color bleeding, rainbow cross-color, dot crawl, and chroma/luma bandwidth limitations.

## Examples

| Source | NTSC Roundtrip | Degraded Signal |
|:---:|:---:|:---:|
| ![Source](examples/colorbars_source.png) | ![NTSC](examples/colorbars_ntsc.png) | ![Degraded](examples/colorbars_degraded.png) |
| Clean SMPTE color bars | After encode/decode roundtrip | With noise, ghosting, attenuation, and jitter |

## Features

- **GUI** (`ntsc-gui`) with live before/after preview, presets, and batch processing
- **Roundtrip** video through the full encode/decode pipeline in memory
- **Telecine** simulation with 3:2 pulldown (24p film → 480i interlaced)
- **Image** processing through the NTSC signal path
- **SMPTE color bars** test pattern generator
- **Physically-motivated signal degradation**: noise (snow), multipath ghosting, attenuation, horizontal jitter
- **Realistic multipath ghosting model**: multiple reflections, phase shift (polarity inversion), HF rolloff, sub-sample delay interpolation, dynamic amplitude modulation
- **VHS tape-path simulation**: luma bandwidth limiting, color-under chroma processing, tape dropouts, edge ringing, luminance-dependent noise, head-wear smearing, causal rightward trailing, head-switching noise, top-of-frame chroma bands, and closed-caption data lines
- **Presets**: `--preset lq|mq|hq` sets the full VHS look at once; `legacy-lq|mq|hq` reproduce the original tape-path settings; custom presets load from JSON with `--preset-file`
- **Range parameters**: all numeric effect arguments accept `min..max` ranges — a random value is sampled per file in batch mode for natural variation
- **Two comb filter modes**: horizontal 2-sample delay (default) and 1H line-delay
- Parallel frame processing via rayon

## Building

Requires Rust 1.95+ (for the GUI; the CLI alone builds with older versions) and [ffmpeg](https://ffmpeg.org/) on your PATH (for video I/O). ffmpeg/ffprobe are only ever run as separate programs; they are not linked into or shipped with this project.

From the repository root:

```bash
cargo build --release
```

This builds two binaries in `target/release/` (`.exe` on Windows):

- `ntsc-composite-simulator` — the command-line tool
- `ntsc-gui` — the GUI with live preview (see [GUI](#gui))

To build only the command-line tool, without the GUI dependencies:

```bash
cargo build --release --no-default-features
```

The binaries target baseline x86-64, so a build runs on any x86-64 CPU. The hot loops are compiled twice, for SSE2 and AVX2, and the AVX2 version is picked at run time when the CPU has it; both give bit-identical output (set `NTSC_SIMD=sse2` to force the SSE2 version).

The crate is a library (`src/lib.rs`: encoder, decoder, effects, the shared parameter model in `params.rs`, presets, and the processing pipeline in `pipeline.rs`) with two thin front ends: the CLI in `src/main.rs` and the GUI in `src/bin/ntsc-gui/` (behind the default `gui` feature). Every effect parameter (flag name, label, help text, unit, slider range, default, group) is defined once in `src/params.rs`; the CLI flags and the GUI controls are both generated from that table.

## GUI

```bash
cargo run --release --bin ntsc-gui
# or open a file straight away
target/release/ntsc-gui photo.png
```

- **Settings panel** (left): pick a preset (VHS `lq`/`mq`/`hq`, DVD, VHS legacy, or your own) and a **Playback deck** profile next to it (the deck's values win where both set something), then fine-tune in collapsible groups that follow the signal chain: DVD, Tape, Colour, Noise & dropouts, Timing, Reception, Playback deck / capture, Input / decoder / output. Native DVD frames are shown at their display aspect. Every parameter has an on/off box, a slider with its unit, a tooltip explaining what it looks like, and a **range** toggle that turns the slider into min/max. Editing marks the preset as *modified*; **Reset group** and **Reset all** go back to the preset. **Save…** / **Load…** store presets as JSON, the same format as `--preset-file`.
- **Preview** (centre): opens with SMPTE colour bars. **Open…** or drag and drop an image, a video (scrub with the time slider, or play it), or a preset `.json`. Views: **Split** (drag the divider), **Side by side**, **After**. Zoom: **Fit**, **1×**, **2×**, **4×** (nearest neighbour), or scroll over the image to zoom from 0.25× to 16× about the cursor; drag to pan, double-click to centre. A frame smaller than the view stays centred, and a larger one cannot be dragged past its edges.
- **Playback** (videos): **Play** / **Pause** next to the time slider, or Space when no text field has focus, plays the processed video in the preview to judge the effects in motion (tape flicker, head-switch hiccups, noise, noise-reduction trails, temporal coherence). Frames go through the same library code as **Process…** and the CLI with their real frame numbers, so each shown frame matches the same frame of a `roundtrip` with the same settings and seed. Split, side by side and zoom keep working; the "before" side shows the source frame of the same moment. By default playback keeps to the source frame rate: frames that are late are not shown, and when no stage needs every frame the processing skips ahead to frames it can finish in time. The deck noise reduction needs every frame, so with it on playback slows down instead (the status bar says so). **Every frame** shows every frame, slower than real time if processing cannot keep up. The status bar shows the achieved frame rate and the dropped frames. **Loop** starts again at the end; scrubbing while playing continues from the new position; changed settings apply from the next frame (playback restarts there). Pausing keeps the shown frame as the preview, so **Save PNG…** and **Copy frame** give that frame. A start in the middle of a video warms the deck noise reduction up on the 8 frames before, as the still preview does, so with it on the first frames after such a start can differ slightly from a file processed from the beginning. The DVD path cannot be played (it encodes the whole clip first).
- **Live rendering** runs in the background, 150 ms after the last change, always on the newest settings. The status bar shows the render time and the **sampled values** that the ranges resolved to. **Seed** + **Reroll** change the random draws; **Frame** steps through frames to see per-frame effects and, with temporal coherence on, the slow drift of the tape state.
- **Toolbar**: **Copy CLI command** copies the exact equivalent `ntsc-composite-simulator` command line, **Save PNG…** saves the processed preview, **Copy frame** (or Ctrl+C) puts it on the clipboard, and **Process…** runs the current settings over a file or a whole folder (images, or video roundtrip: lossless by default, optional lossy CRF, encoder preset, telecine) with a progress bar and **Cancel**. Processing uses the same library code as the CLI, so the output matches the CLI with the same settings.

Optional start-up arguments: `--preset <name>`, `--deck <name>`, `--time <seconds>` (for videos), `--view split|side|after`, `--zoom fit|1|2|4`, `--process`, `--screenshot <png>` (save a window screenshot once the preview has rendered, then exit).

**Crash reports**: the GUI has no console, so panics and failed memory allocations are appended to `%LOCALAPPDATA%\ntsc-simulator\crash.log` (on other systems `~/.local/share/ntsc-simulator/crash.log`) with the thread, the message, the source location and a backtrace. A crash on the UI thread also shows a message box with the file's path. A panic in a background job (preview render, frame grab, playback, Process…) is logged and shown as an error in the status bar or the Process dialog instead of closing the program. Release builds keep line tables (`ntsc_gui.pdb` next to the `.exe` on Windows) so the backtrace names files and lines; keep the `.pdb` with the `.exe`.

## Usage

### Roundtrip video through NTSC

```bash
ntsc-composite-simulator roundtrip input.mp4 -o output.mkv
```

With telecine (3:2 pulldown, interlaced 480i output):

```bash
ntsc-composite-simulator roundtrip input.mp4 -o output.mkv --telecine
```

### Output quality and colour

Video output is **lossless and RGB-exact by default**: `.mkv` (the default `output.mkv`) is written as FFV1 with an 8-bit RGB pixel format, `.mp4`/`.mov` as x264 RGB at QP 0. Decoding the file back to RGB gives exactly the frames the simulator produced. The old default, x264 4:2:0 at CRF 17, halves the colour resolution and smooths away the fine noise and artefacts this tool generates, which matters when the output is used as training data. It is still available with `--lossy` (and `--crf`, `--preset`).

Input video is converted to RGB with the source's colour matrix and range: taken from the stream's tags when present, otherwise BT.709 for HD (720 lines or more) and BT.601 for SD, limited range except for full-range (JPEG) formats. Override with `--input-matrix 601|709` and `--input-range limited|full`. `extract-frame` saves one frame with the same conversion:

```bash
ntsc-composite-simulator extract-frame input.mp4 -o frame.png --time 12.5
```

Outputs are tagged as RGB with SMPTE 170M primaries and transfer (the NTSC decode), full range.

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
ntsc-composite-simulator roundtrip input.mp4 -o output.mkv \
  --ghost 0.25 --ghost-delay 1.5 --ghost-phase 180

# Multiple reflections with dynamic amplitude drift
ntsc-composite-simulator roundtrip input.mp4 -o output.mkv \
  --ghost 0.25 --ghost-delay 1.5 --ghost-phase 180 --ghost-dynamic \
  --ghost-multi "0.10,4.2,0;0.05,8.0,180"

# Heavy distant-antenna reception with soft ghosts and snow
ntsc-composite-simulator roundtrip input.mp4 -o output.mkv \
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
ntsc-composite-simulator roundtrip input.mp4 -o output.mkv \
  --vhs-luma-bw 3.0 --color-under-bw 400 --luma-noise 0.04 \
  --edge-ringing 1.0

# Worn EP-mode tape with dropouts, head smear, and trailing
ntsc-composite-simulator roundtrip input.mp4 -o output.mkv \
  --vhs-luma-bw 1.6 --color-under-bw 300 --luma-noise 0.06 \
  --tape-dropout-rate 8 --head-smear 0.5 --tape-trail 0.4

# Heavily degraded multi-generation dub
ntsc-composite-simulator roundtrip input.mp4 -o output.mkv \
  --vhs-luma-bw 1.6 --color-under-bw 250 --luma-noise 0.08 \
  --edge-ringing 1.5 --tape-dropout-rate 15 \
  --head-smear 0.7 --tape-trail 0.7 --noise 0.03

# Head-switching tear at the bottom, blue/purple bands and caption dashes at the top
ntsc-composite-simulator image photo.png -o vhs.png \
  --vhs-luma-bw 3.0 --color-under-bw 400 \
  --head-switch 0.7 --head-switch-lines 3 --top-band 0.5 --vbi-data 0.5

# The research-based FM tape model: emphasis and clipping, FM noise at 37 dB before the
# deck's noise canceller, counted dropouts per minute, head-contact bands, captured
# through a chroma trap as capture chips do for VCR input
ntsc-composite-simulator roundtrip input.mp4 -o output.mkv \
  --tape-model fm --vhs-luma-bw 3 --tape-snr 37 --dropouts 20 --rf-loss 0.3 \
  --doc luma --deck-nc 4 --deck-sharpness 0.4 --comb trap
```

#### The FM tape model

`--tape-model fm` replaces the original tape path (a lowpass on luma and colour) with the circuits of the VHS format, from the format standard and service literature:

- **Record**: the recording deck's luma/chroma split (a band filter of ± 0.6 MHz around the subcarrier; luma detail inside it is recorded as cross-colour), the colour-under filter, the record luma lowpass, optional nonlinear emphasis (`--nl-emphasis`), the main pre-emphasis (1.3 µs, ×5 shelf, each tape's record and playback networks within the format's ± 0.05 µs / ± 0.3 tolerance) and the white and dark clips (`--white-clip` 160 %, `--dark-clip` 40 % of the sync-to-white span). Large sharp edges overshoot the clips, so after de-emphasis they come out softened with a causal trail (a slow rise into white, a light trail after bright objects) while small edges come back intact; this replaces `--tape-trail`.
- **Tape**: FM noise in the emphasised domain (`--tape-snr`; after de-emphasis it rises with frequency, fine streaky grain about equal in dark and bright areas), dropouts counted per minute (`--dropouts`, lengths mostly 1–2 µs, none under 0.5 µs, white or black streaks with de-emphasis tails, spanning several fields on EP with `--dropout-span`), head-contact RF loss in drifting bands (`--rf-loss`, more noise and sparkles, slightly softer detail, weaker colour; replaces `--head-smear`), tracking noise bars (`--tracking-bars`), luma crosstalk beats (`--luma-crosstalk`), the EP interleave residue (`--ep-interleave`), black/white inversion after bright edges (`--bw-inversion`), a level step between the heads (`--head-level-step`) and dubs (`--generations`).
- **Playback**: the demodulator's lowpass (record + playback −3 dB at 0.66 × and −10 dB at 1.0 × `--vhs-luma-bw`), de-emphasis and nonlinear de-emphasis; dropout compensation per `--doc`.

`--tape-model rf` goes one step further and simulates the FM carrier itself (modulated at 3.4–4.4 MHz at 8 × fsc, record highpass and head/tape lowpass, RF noise, limiter and demodulator), so noise, sparkles, dropout streaks and crosstalk beats come out of the channel; it costs about four times as much per frame.

Every new stage is opt-in and draws its random numbers from its own stream, so turning one on never changes the others.

#### Presets

`--preset` (also accepted as `--vhs-preset`) fills in every flag you did not pass. Flags you pass explicitly always win, so a preset is a starting point:

```bash
# Worn EP-mode tape
ntsc-composite-simulator roundtrip input.mp4 -o output.mkv --preset lq

# Typical tape, but without the closed-caption dashes
ntsc-composite-simulator roundtrip input.mp4 -o output.mkv --preset mq --vbi-data 0

# The original tape-path-only look
ntsc-composite-simulator roundtrip input.mp4 -o output.mkv --preset legacy-mq
```

There are two families:

- `lq`, `mq`, `hq`: the full VHS look, built on the FM tape model and a typical consumer playback deck and capture card. `hq` is a clean SP recording, `mq` a typical rental tape with LP-like wear, `lq` a worn EP tape (some files are second-generation dubs).
- `legacy-lq`, `legacy-mq`, `legacy-hq`: exactly the original tape-path settings and nothing else. With the same `--seed` they give byte-identical output to earlier versions and to passing those seven flags by hand.

| Flag | `lq` (worn EP) | `mq` (rental, LP-like) | `hq` (SP) |
|---|---|---|---|
| `--tape-model` | fm | fm | fm |
| `--vhs-luma-bw` | 2.5 | 2.9 | 3 |
| `--color-under-bw` | 400..450 | 400..450 | 400 |
| `--nl-emphasis` | 1 | 0.5..1 | 0..0.5 |
| `--tape-snr` | 30..33 | 34..36 | 37..38 |
| `--dropouts` | 40..100 | 15..30 | 5..15 |
| `--dropout-span` | 2..4 | 1..2 | 1 |
| `--edge-ringing` (source) | 0.15..0.25 | 0.25..0.35 | 0.2..0.3 |
| `--rf-loss` | 0.6..0.9 | 0.3..0.5 | 0.1..0.2 |
| `--tracking-bars` | 0..0.3 | – | – |
| `--luma-crosstalk` | 0.4..0.7 | 0.1..0.3 | – |
| `--ep-interleave` | 0.2..0.8 | – | – |
| `--bw-inversion` | 0.3..0.6 | 0.1..0.2 | – |
| `--head-level-step` | 1..3 | 0.5..2 | 0..1 |
| `--generations` | 1..2 | 1 | 1 |
| `--head-switch` | 0.8..1.0 | 0.55..0.7 | 0.35..0.45 |
| `--head-switch-point` | 5.5..7.5 | 6..7 | 6..7 |
| `--flagging` | 0.4..0.6 | 0.25..0.4 | 0.15..0.25 |
| `--top-band` | 0.6..0.8 | 0.4..0.55 | 0.2..0.3 |
| `--vbi-data` | 0.6..0.8 | 0.4..0.55 | 0.25..0.35 |
| `--chroma-noise` | 0.025..0.035 | 0.015..0.025 | 0.008..0.015 |
| `--chroma-noise-bw` | 450 | 450 | 450 |
| `--chroma-crosstalk` | 0.6..0.9 | 0.3..0.5 | 0.15..0.3 |
| `--chroma-phase-noise` | 0 | 0 | 0 |
| `--chroma-vblur` | 1 | 1 | 1 |
| `--chroma-delay` | 2..3 | 1.5..2.5 | 1..2 |
| `--chroma-delay-v` | 1 | 0..1 | 0 |
| `--wobble` | 1..1.5 | 0.5..1 | 0.3..0.6 |
| `--temporal-coherence` | 1 | 1 | 1 |
| `--deck-nc` | 3..5 | 3..5 | 3..5 |
| `--deck-sharpness` | 0.3..0.5 | 0.3..0.5 | 0.3..0.5 |
| `--doc` | luma | luma | luma |
| `--comb` | trap | trap | trap |

Measured with `examples/vhs_measure.rs` (60 IRE p-p multiburst, flat 50 IRE field, unweighted luma S/N through the decoder): `hq` −3 dB at 2.2 MHz and −10 dB at 2.7 MHz on S-Video (2.0 / 2.5 MHz composite), S/N 45 dB; `mq` 2.2 / 2.8 MHz, 42 dB; `lq` 1.1–2.0 / 1.8–2.6 MHz (worn, varies with the tape's bands and dubbing), 37 dB. The tape channel alone (no deck noise canceller or sharpening) is −3 dB at 2.25 MHz and −10 dB at 3.0 MHz for `hq`, the 240 TVL of SP decks; deck S/N specs (43–45 dB SP, about 41 dB EP) are measured with the deck's noise canceller on.

| Flag | `legacy-lq` | `legacy-mq` | `legacy-hq` |
|---|---|---|---|
| `--vhs-luma-bw` | 1.6 | 3 | 3 |
| `--color-under-bw` | 400..450 | 400..450 | 300..400 |
| `--edge-ringing` | 0.3..0.4 | 0.25..0.35 | 0.2..0.3 |
| `--luma-noise` | 0.025..0.035 | 0.02..0.03 | 0.015..0.025 |
| `--tape-dropout-rate` | 1 | 0..1 | 0 |
| `--head-smear` | 0.2..0.3 | 0.15..0.2 | 0.1..0.15 |
| `--tape-trail` | 0.75..0.9 | 0.6..0.75 | 0.5..0.7 |

DVD presets select the DVD path (`--format dvd`); see [DVD-Video path](#dvd-video-path):

| Preset | Frames | Size | Avg / peak Mbit/s | Other |
|---|---|---|---|---|
| `dvd-hq` | 23.976p, soft pulldown | 720×480 | 7–8 / 9.5 | two-pass |
| `dvd-mq` | 23.976p, soft pulldown | 720×480 | 5–6 / 8–9 | |
| `dvd-mq-ee` | 23.976p, soft pulldown | 720×480 | 5–6 / 8–9 | edge enhancement 0.5–0.7, DNR 0.3–0.5 |
| `dvd-lq` | 23.976p, soft pulldown | 352×480 | 3–4 / 6–7 | low-rate matrices, pre-filter 0.3–0.5 |
| `dvd-interlaced` | 29.97i TFF | 720×480 | 5–6 / 8–9 | video-sourced |

#### Preset files

`--preset-file <path>` reads a JSON preset (the GUI saves the same format). Its values fill in flags not given explicitly, before `--preset`. Values are written as on the command line: a number, a `"min..max"` string, `true`/`false` for switches, or text; `seed` is optional.

```json
{
  "name": "My tape",
  "params": {
    "vhs-luma-bw": "1.6",
    "color-under-bw": "400..450",
    "chroma-delay": 2,
    "ghost-dynamic": true
  },
  "seed": 42
}
```

### Playback deck and capture chain

The deck and capture stages are off unless set; `--deck` picks a profile that combines with the tape preset (the deck's values win where both set something):

```bash
# Worn tape played on a high-end S-VHS deck with TBC and noise reduction
ntsc-composite-simulator roundtrip input.mp4 -o output.mkv --preset lq --deck deck-svhs-tbc
```

| Profile | Settings |
|---|---|
| `deck-basic` | typical consumer VCR: `--deck-nc 4 --deck-sharpness 0.4 --doc luma`, composite output, no TBC |
| `deck-svhs-tbc` | `--line-tbc 0.9 --frame-tbc 1 --deck-dnr 0.5 --comb 3d --video-out svideo --deck-nc 5 --deck-ynr 1 --deck-cnr 1 --deck-sharpness 0.2 --doc luma` |
| `deck-edit` | `--deck-sharpness 0 --deck-nc 0 --deck-cnr 0`: edit mode as on pro decks (JVC BR-S422), no sharpening, noise canceller and colour noise reducer off; the line noise reducer and source enhancement are left as set |

- **Line TBC** (`--line-tbc`): re-times every line from its own sync, removing most jitter and wobble, the top-of-field skew and the displacement of the lines after the head switch (the switch line itself and the switching noise stay).
- **Frame TBC** (`--frame-tbc`): holds the picture vertically, removing `--v-jitter` (vertical bounce, a new tape effect: whole fields jump by whole lines).
- **Deck noise reduction** (`--deck-dnr`): recursive, motion-adaptive temporal filter on the output frames; cleans luma and colour noise in still areas, while moving edges and thin lines pass (faint, low-contrast moving detail can leave a short soft trail). It needs consecutive frames, so it runs as a sequential stage after the parallel encode/decode and does nothing on single images (the GUI renders the previous 8 frames of a video so it shows in the preview).
- **Noise canceller** (`--deck-nc`, IRE): the luma coring nearly every VCR has: detail under the threshold (typical 4–10) above 1 MHz is removed with the noise; flat areas look clean and slightly waxy, fine texture goes.
- **Line and chroma noise reducers** (`--deck-ynr`, `--deck-cnr`): HQ decks' recursive 1H reducers; small line-to-line differences are averaged, strong vertical changes pass; non-recursive near the head switch.
- **Deck sharpness** (`--deck-sharpness -1..1`): the deck's picture control, after the noise canceller: a short symmetric peaker around 2.4 MHz (up to +6 dB, 1.5 IRE coring); negative softens, 0 is edit mode. `--edge-ringing` is enhancement already on the tape (source chain).
- **Dropout compensator** (`--doc composite|luma|off`): `luma` copies the luma of the line above and mutes the colour, as real decks do; `composite` (the original) copies the whole line, whose colour decodes in the complementary hue.
- **Decoder comb** (`--comb 2sample|1h|3d|trap`): the TV or capture card's luma/chroma separation for composite signals. `trap` is a chroma trap (± 0.6 MHz), what capture chips use for VCR input: luma keeps its detail to about 2.9 MHz (the 2-sample comb is −3 dB at 1.8 MHz). `3d` is a motion-adaptive frame comb that uses the previous frame in still areas (clean luma and chroma there) and falls back to the 2-sample comb where the picture moves (with the FM tape models, motion is checked against both earlier frames). `--comb-1h` still selects the line comb.
- **Capture peaking** (`--capture-peaking`, `--capture-peaking-mhz`): the capture chip's luma peaking (TVP5150 off by default, SAA7113 a small boost at 4.1 MHz).

### DVD-Video path

`--format dvd` replaces the analog chain with a DVD-Video (NTSC) chain: mastering, MPEG-2 encoding within the DVD-Video limits, player decoding, and the player's output.

```bash
# A video through a typical DVD (digital output, like a rip)
ntsc-composite-simulator roundtrip input.mp4 -o dvd.mkv --format dvd --dvd-bitrate 6

# Film title (23.976p with soft pulldown), halos and DNR from the master, played over composite,
# keeping the MPEG-2 program stream
ntsc-composite-simulator roundtrip film.mp4 -o dvd.mkv --format dvd --dvd-rate film \
  --dvd-ee 0.6 --dvd-dnr 0.4 --dvd-output composite --dvd-keep-mpeg2 movie.vob

# Check a stream against the DVD-Video limits
ntsc-composite-simulator dvd-check movie.vob
```

The chain:

1. **Mastering**: the source is fitted into the 704-sample active area (8 samples of black each side in a 720-wide frame) with a Lanczos-3 filter and the correct non-square sample aspect (10:11 for 4:3, 40:33 for 16:9 anamorphic), letterboxed or pillarboxed if its aspect differs. Optional studio stages: a pre-compression lowpass (`--dvd-prefilter`), edge-preserving noise reduction (`--dvd-dnr`) and edge enhancement halos (`--dvd-ee`).
2. **4:2:0**: conversion to BT.601 studio-range Y'CbCr; chroma is subsampled per field for interlaced pictures (as interlaced 4:2:0 must be) and per frame for progressive ones, with MPEG-2 chroma siting.
3. **MPEG-2 encode** with ffmpeg's `mpeg2video` (run as a separate program): Main Profile @ Main Level, 4:2:0, 720×480 / 704×480 / 352×480 / 352×240, 29.97 fps interlaced top field first or 23.976 fps progressive with soft 3:2 pulldown, 4:3 or 16:9 flags, closed GOPs with M = 3 and at most 18 pictures, VBR with `--dvd-bitrate` average and `--dvd-maxrate` peak (≤ 9.8 Mbit/s), VBV buffer 1 835 008 bits (224 KiB), 9-bit intra DC, optional low-rate quantiser matrices (`--dvd-matrix lowrate`) and two-pass (`--dvd-2pass`). ffmpeg cannot write pulldown flags, so the soft pulldown (repeat_first_field / top_field_first in the 2:3 pattern, 29.97 fps sequence rate) is patched into the stream here.
4. **Player decode**: chroma upsampled per field for interlaced pictures; `--dvd-cue` blends in the progressive upsampling many players used (the chroma upsampling error: streaky, jagged colour edges on interlaced material).
5. **Player output** (`--dvd-output`): `digital` (the decoded frames, like a rip), or `svideo` / `composite` through the NTSC encoder and decoder (film titles get 3:2 pulldown on the analog output). `--dvd-output-size square` scales the active area to 640×480 (854×480 for 16:9).

Still images are encoded as a short sequence (one GOP plus a few pictures) so they get real I, P and B pictures; `--dvd-frame-type` picks which picture is returned (default `b`, since two thirds of the pictures on a DVD are B pictures). `--dvd-keep-mpeg2` saves the stream (`.m2v` elementary stream, `.vob`/`.mpg` DVD program stream). Tape and reception effects are not applied on the DVD path unless set explicitly, and then only on the analog outputs.

`dvd-check` reports profile/level, chroma format, frame size, frame rate (and soft pulldown), aspect flag, maximum GOP length, closed GOPs, VBV buffer size, header and peak 1-second bit rate, and field order, plus ffprobe's view of the stream.

**How accurate is it?** The streams meet the DVD-Video video limits listed above, but real discs were made with professional encoders (with their own rate control, motion search, quantiser matrices and adaptive quantisation), so their artefacts differ in detail from ffmpeg's `mpeg2video`, which is spec-compliant but not identical to any of them. ffmpeg's closed GOPs end on a P picture, so they are 3k+1 pictures long (15 gives 13). Only the video stream is produced (no audio, navigation or authoring); `dvd-check` checks the video limits only. Mastering and player stages are models of typical behaviour, not of a specific telecine or player.

## Options Reference

### General

| Flag | Commands | Description |
|---|---|---|
| `-o, --output` | all | Output file path |
| `--width` | roundtrip, image, colorbars | Output width (default: 640) |
| `--height` | roundtrip, image, colorbars | Output height (default: 480) |
| `--comb-1h` | roundtrip, image, colorbars | Use 1H line-delay comb filter instead of 2-sample delay |
| `--save-source` | colorbars | Save the source SMPTE pattern as PNG |
| `--seed` | roundtrip, image, colorbars | Seed for all random choices (sampled ranges and per-frame effects), for exactly reproducible output; in batch mode each file gets its own seed derived from it |
| `--legacy-burst-phase` | roundtrip, image, colorbars | Decode with the inverted burst phase estimate of older builds, which doubles burst phase errors (e.g. from `--luma-noise`) into per-line hue noise. Together with `--seed`, reproduces output from builds before fb47e5c exactly |

### Video encoding

| Flag | Commands | Description |
|---|---|---|
| `--telecine` | roundtrip | Enable 3:2 pulldown telecine |
| `--lossy` | roundtrip | Lossy x264 4:2:0 output at `--crf` instead of the lossless default |
| `--crf` | roundtrip | x264 CRF quality with `--lossy`, 0=lossless, 51=worst (default: 17) |
| `--x264-preset` | roundtrip | x264 encoding preset for `--lossy` and lossless `.mp4`, e.g. `ultrafast`, `fast`, `slow` (default: fast); an x264 preset name given to `--preset` still works |
| `--input-matrix` | roundtrip, extract-frame | Source colour matrix: `auto` (tag, else BT.709 for HD / BT.601 for SD), `601`, `709` |
| `--input-range` | roundtrip, extract-frame | Source range: `auto` (tag, else limited), `limited`, `full` |
| `--video-out` | roundtrip, image, colorbars | Connection to the TV / capture card: `composite` (default; comb filter, cross-colour) or `svideo` (separate Y/C wires, no comb, no cross-colour) |
| `--threads` | roundtrip | Number of parallel worker threads (default: all logical cores) |

### Signal degradation (reception)

All numeric flags below accept either a single value (e.g. `0.05`) or a `min..max` range (e.g. `0.02..0.10`). With a range, each file in a batch gets a randomly sampled value.

| Flag | Commands | Description |
|---|---|---|
| `--noise` | roundtrip, image, colorbars | Snow amplitude (0.05 = subtle, 0.2 = heavy) |
| `--attenuation` | roundtrip, image, colorbars | Signal attenuation 0–1 (0 = none, 1 = flat at blanking) |
| `--reception-before-tape` | roundtrip, image, colorbars | Apply snow, ghosts and attenuation before the tape path (an off-air recording) instead of after it |
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
| `--tape-model` | roundtrip, image, colorbars | `classic` (default, the original tape path), `fm` (the VHS record and playback circuits) or `rf` (fm with the FM carrier simulated) |
| `--vhs-luma-bw` | roundtrip, image, colorbars | Luma bandwidth in MHz. Classic model: a lowpass cutoff. FM model: the −10 dB point of record + playback (SP 3.0, LP/EP 2.8–2.9) |
| `--color-under-bw` | roundtrip, image, colorbars | Color-under chroma bandwidth in kHz (typical 300–500) |
| `--white-clip`, `--dark-clip` | roundtrip, image, colorbars | FM model: emphasis clips in % of the sync-to-white span (defaults 160 and 40) |
| `--nl-emphasis` | roundtrip, image, colorbars | FM model: nonlinear emphasis strength (0–1) |
| `--tape-snr` | roundtrip, image, colorbars | FM model: tape noise S/N in dB before the deck's noise reduction (37–38 SP, 34–36 LP, 30–33 worn EP) |
| `--rf-loss` | roundtrip, image, colorbars | FM model: head-contact RF loss in drifting bands (0–1) |
| `--tracking-bars` | roundtrip, image, colorbars | FM model: tracking noise bar (0–1) |
| `--luma-crosstalk` | roundtrip, image, colorbars | FM model: adjacent-track luma crosstalk beats (0–1; 1 = −20 dB) |
| `--ep-interleave` | roundtrip, image, colorbars | FM model: EP carrier-interleave residue, field-alternating DC in IRE (0–1.1) |
| `--bw-inversion` | roundtrip, image, colorbars | FM model: black/white inversion after bright edges (0–1) |
| `--head-level-step` | roundtrip, image, colorbars | FM model: level difference between the heads in % (0–5) |
| `--generations` | roundtrip, image, colorbars | FM model: tape generations (1 = original, 2+ = dubs) |
| `--edge-ringing` | roundtrip, image, colorbars | Source enhancement already on the tape (0.5–3.0); with the FM model it is recorded through the tape path |
| `--luma-noise` | roundtrip, image, colorbars | Classic model: luminance-dependent tape noise amplitude (0.02–0.10) |
| `--tape-dropout-rate` | roundtrip, image, colorbars | Classic dropouts, average per frame (real tapes: well under 0.1) |
| `--dropouts` | roundtrip, image, colorbars | FM model: dropouts per minute as counted by the industry (≥ 15 µs, ≥ 16 dB; typical 5–30) |
| `--dropout-span` | roundtrip, image, colorbars | FM model: most consecutive fields one defect hits (1 SP … 4 EP) |
| `--doc` | roundtrip, image, colorbars | Dropout compensator: `composite` (default), `luma`, `off` |
| `--tape-dropout-len` | roundtrip, image, colorbars | Average dropout length in µs (default: 15.0) |
| `--head-smear` | roundtrip, image, colorbars | Worn-head symmetric luma blur (0.3–1.0); varies in bands across the frame |
| `--tape-trail` | roundtrip, image, colorbars | Causal rightward luma smear (0.3–1.0); the classic worn-tape trailing effect |
| `--head-switch` | roundtrip, image, colorbars | Head-switching noise at the bottom of each field (0–1): a displaced switch line with a switching transient, then near-black lines with bright noise streaks; stable over a tape with per-field flicker and occasional sudden jumps |
| `--head-switch-lines` | roundtrip, image, colorbars | Active lines per field torn by head switching at full strength (default: 3, i.e. 6 rows at 480; fewer at low strength) |
| `--head-switch-point` | roundtrip, image, colorbars | Switch point in lines before vertical sync (6.5 ± 1.5): sets the torn lines (about 5–7 rows at 6.5) regardless of strength |
| `--head-switch-after` | roundtrip, image, colorbars | Lines after the switch: `black` (signal lost, default) or `picture` (displaced picture) |
| `--flagging` | roundtrip, image, colorbars | Top-of-field skew from the head switch and a slow PLL (0–1) |
| `--top-band` | roundtrip, image, colorbars | Faint blue/purple chroma bands on the first lines of each field (0–1) |
| `--vbi-data` | roundtrip, image, colorbars | Closed-caption data on lines 21/284, visible as short light dashes near the top (0–1) |
| `--chroma-noise` | roundtrip, image, colorbars | Colour-under chroma noise, std dev of I/Q noise (subtle 0.01–0.03, heavy 0.06); horizontal colour blotches |
| `--chroma-phase-noise` | roundtrip, image, colorbars | Per-line chroma phase (hue) jitter, std dev in degrees (subtle 1–3) |
| `--chroma-noise-bw` | roundtrip, image, colorbars | Colour-under noise bandwidth in kHz (about 450); blotches about 1 µs long |
| `--chroma-crosstalk` | roundtrip, image, colorbars | Adjacent-track colour crosstalk (0–1; 1 = −20 dB), cancelled by the 1H comb except at vertical colour changes |
| `--chroma-vblur` | roundtrip, image, colorbars | Playback 1H chroma comb (0–1): chroma mixed with the previous line of the field, softening colour vertically (real decks: 1) |
| `--chroma-delay` | roundtrip, image, colorbars | Horizontal chroma delay relative to luma in output pixels at 640 wide (typical 1–4, positive = right) |
| `--chroma-delay-v` | roundtrip, image, colorbars | Vertical chroma delay in field lines (typical 0–1; 1 field line = 2 output rows) |
| `--wobble` | roundtrip, image, colorbars | Correlated time-base wobble: peak horizontal displacement in output pixels (typical 0.5–3); smooth across lines, burst moves with the line so hue is kept |
| `--wobble-rate` | roundtrip, image, colorbars | Wobble frequency scale (default 1.0 = periods of ~20–300 lines) |
| `--temporal-coherence` | roundtrip | Tape-state coherence over time (0–1). 0 (default) re-rolls head-smear bands, wobble, head-switch shape and top-band tint every frame; 1 makes them drift slowly from frame to frame like a real tape. Chroma noise stays random |
| `--preset` | roundtrip, image, colorbars | `lq`, `mq`, `hq`, `legacy-lq`, `legacy-mq`, `legacy-hq`, `dvd-hq`, `dvd-mq`, `dvd-mq-ee`, `dvd-lq`, `dvd-interlaced`: fills in all flags not given explicitly (alias `--vhs-preset`) |
| `--preset-file` | roundtrip, image, colorbars | JSON preset file; fills in flags not given explicitly, before `--preset` |

### DVD path

| Flag | Commands | Description |
|---|---|---|
| `--format` | roundtrip, image, colorbars | `analog` (default) or `dvd` |
| `--dvd-size` | roundtrip, image, colorbars | `720x480` (default), `704x480`, `352x480`, `352x240` |
| `--dvd-aspect` | roundtrip, image, colorbars | `4:3` (default) or `16:9` anamorphic |
| `--dvd-rate` | roundtrip, image, colorbars | `interlaced` (29.97i TFF, default) or `film` (23.976p, soft pulldown) |
| `--dvd-bitrate` | roundtrip, image, colorbars | Average video bit rate, Mbit/s (default 6) |
| `--dvd-maxrate` | roundtrip, image, colorbars | Peak video bit rate, Mbit/s (default 9.5, at most 9.8) |
| `--dvd-gop` | roundtrip, image, colorbars | Maximum GOP length (default 15, at most 18; closed GOPs come out 3k+1 long) |
| `--dvd-matrix` | roundtrip, image, colorbars | `standard` (MPEG defaults) or `lowrate` quantiser matrices |
| `--dvd-2pass` | roundtrip, image, colorbars | Two-pass VBR encode |
| `--dvd-ee` | roundtrip, image, colorbars | Mastering edge enhancement halos (0–1) |
| `--dvd-dnr` | roundtrip, image, colorbars | Mastering noise reduction (0–1) |
| `--dvd-prefilter` | roundtrip, image, colorbars | Pre-compression luma lowpass (0–1) |
| `--dvd-cue` | roundtrip, image, colorbars | Player chroma upsampling error (0–1) |
| `--dvd-output` | roundtrip, image, colorbars | `digital` (default), `svideo`, `composite` |
| `--dvd-output-size` | roundtrip, image, colorbars | `native` (default, e.g. 720×480) or `square` (640×480 / 854×480) |
| `--dvd-frame-type` | image, colorbars | Picture type returned for stills: `b` (default), `p`, `i` |
| `--dvd-keep-mpeg2` | roundtrip, image, colorbars | Save the MPEG-2 stream (`.m2v` elementary, `.vob`/`.mpg` program stream) |

### Playback deck / capture

| Flag | Commands | Description |
|---|---|---|
| `--deck` | roundtrip, image, colorbars | Deck profile: `deck-basic`, `deck-svhs-tbc`, `deck-edit` |
| `--line-tbc` | roundtrip, image, colorbars | Line time-base corrector strength (0–1) |
| `--frame-tbc` | roundtrip, image, colorbars | Frame TBC / frame synchroniser strength (0–1) |
| `--v-jitter` | roundtrip, image, colorbars | Vertical bounce, std dev in output rows |
| `--deck-dnr` | roundtrip | Recursive temporal noise reduction (0–1); no effect on single images |
| `--deck-nc` | roundtrip, image, colorbars | Noise canceller threshold in IRE (4–10 typical) |
| `--deck-ynr` | roundtrip, image, colorbars | Recursive line noise reducer (0–1) |
| `--deck-cnr` | roundtrip, image, colorbars | Recursive chroma noise reducer (0–1) |
| `--deck-sharpness` | roundtrip, image, colorbars | Deck picture control (−1 soft … 0 edit … 1 sharp), after the noise canceller |
| `--comb` | roundtrip, image, colorbars | Decoder comb: `2sample` (default), `1h`, `3d`, `trap` |
| `--capture-peaking` | roundtrip, image, colorbars | Capture-chip luma peaking gain (0–2) |
| `--capture-peaking-mhz` | roundtrip, image, colorbars | Its centre frequency in MHz (default 2.6) |

Run `ntsc-composite-simulator <command> -h` for full details.

## Third-party licences

Direct dependencies:

| Crate | Licence | Used for |
|---|---|---|
| `realfft` | MIT | FFT filtering |
| `rayon` | MIT OR Apache-2.0 | parallel frame processing |
| `clap` | MIT OR Apache-2.0 | command-line parsing |
| `image` | MIT OR Apache-2.0 | image I/O |
| `indicatif` | MIT | CLI progress bars |
| `anyhow` | MIT OR Apache-2.0 | error handling |
| `rand`, `rand_distr` | MIT OR Apache-2.0 | random numbers |
| `serde`, `serde_json` | MIT OR Apache-2.0 | preset files |
| `eframe` (egui) | MIT OR Apache-2.0 | GUI (`gui` feature) |
| `rfd` | MIT | GUI file dialogs (`gui` feature) |

The GUI embeds egui's default fonts (`epaint_default_fonts`): Ubuntu-Light (Ubuntu Font Licence 1.0), Noto Emoji (SIL Open Font License 1.1), Hack (MIT) and emoji-icon-font (MIT). One transitive crate, `self_cell`, is dual-licensed Apache-2.0 OR GPL-2.0-only and is used under Apache-2.0.

[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) lists every crate that can end up in a release build, with its licence and the licence texts. Ship it with any binaries. Regenerate it after changing dependencies:

```bash
python tools/third_party_licenses.py > THIRD_PARTY_LICENSES.md
```

Crates whose package contains no licence file get their text from `licenses/overrides/` (egui's LICENSE-MIT and LICENSE-APACHE from its repository), or are listed with their licence expression and repository.

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
3. **Playback chroma path** — chroma is demodulated to I/Q per line and rotated into the frame of that line's burst, so lines can be mixed without the line-to-line subcarrier inversion cancelling them. In that frame, colour-under noise (lowpassed at 150 kHz and half-correlated with the previous line of the field, so it forms blotches) is added, each line can get a random hue rotation (phase jitter), and a 1H comb can mix each line's chroma with the previous line of the field (vertical colour blur); finally the chroma can be delayed relative to luma, by a fractional number of samples to the right and by whole field lines down. Only the change is remodulated, so luma is untouched
4. **Edge ringing** — unsharp-mask peaking with intentional Gibbs-phenomenon overshoot simulates the sharpness-enhancement circuit in playback decks
5. **Luminance-dependent noise** — band-limited noise scaled by signal darkness (darker = noisier), matching real FM tape noise characteristics
6. **Tape dropouts** — random horizontal streaks from oxide damage; short ones flash white, longer ones trigger the dropout compensator (previous-line hold)
7. **Head-wear smear** — variable-width symmetric box blur on luma only, modulated in slow bands across the frame to simulate uneven head-to-tape contact
8. **Closed-caption data** — an EIA-608 line-21 waveform (7 cycles of 503 kHz clock run-in, start bits, two random 7-bit characters with odd parity as 50 IRE NRZ) is written on lines 21 and 284 before the tape path, so it is band-limited like the rest of the recording
9. **Time-base wobble** — each visible line, burst included, is shifted by a smooth, fractional displacement: three sinusoids across the lines (periods ~20–45, 60–140 and 150–300 lines, random phases) plus a mean-reverting random walk, modelling tape stretch, capstan flutter and drum speed variation
10. **Top chroma bands** — on the first 3–6 active lines of each field the chroma is demodulated, rotated by a random angle and biased toward −I/+Q (blue/purple) in the line's burst frame, with a slight luma lift, all decaying down the field; this models the deck's chroma phase loop settling after the vertical interval
11. **Head switching** — the last few active lines of each field get a time-base jump: they are shifted right by a fractional, growing number of samples, exposing black on the left, and the burst moves with them apart from a small residual error that shows as colour fringes; the switching line itself drops toward black over part of its width and carries wideband noise and bright speckles
12. **Tape trailing** — causal one-pole IIR on luma only, applied as the final processing step so everything on the tape (noise, dropouts, detail) gets the characteristic rightward decay tail of worn playback electronics

Items 1, 4–7 and 12 are the classic model, kept unchanged for the `legacy-*` presets. The FM tape model (`--tape-model fm`, see [The FM tape model](#the-fm-tape-model)) replaces them: luma through emphasis, clips and de-emphasis with FM noise and research-based dropouts, the deck's noise canceller and picture control after the noise, and a band-filter luma/chroma split.

### Testing

`cargo test --profile fasttest` runs the suite with optimisation but without the slow LTO link. `tests/output_lock.rs` renders fixed pictures and short videos with fixed seeds: `tests/locks/legacy.txt` locks the legacy presets, effects off and the DVD path (these must never change), `tests/locks/presets.txt` locks `lq`/`mq`/`hq` and the deck profiles and is rewritten deliberately with `NTSC_UPDATE_LOCKS=presets`. `cargo run --release --example vhs_measure -- <preset> [flag=value ...]` prints the luma frequency response and S/N of a setting.

### Signal Specifications

| Parameter | Value |
|---|---|
| Sample rate | 14,318,180 Hz (4 × F_SC) |
| Color subcarrier | 3,579,545 Hz |
| Frame rate | 29.97 fps |
| Lines per frame | 525 (480 visible) |
| Samples per line | 910 |

## Changelog

Newest first.

- **GUI file dialogs no longer block the window**: Open…, Save PNG…, preset Load…/Save… and the Process dialog's File…/Folder…/Browse… run the native dialog on its own thread. Shown inside the UI callback, a dialog stopped winit's event loop and winit buffered every incoming event without a limit while it was open; one such session ran out of memory and closed the GUI (0xc0000409).
- **GUI crash log**: panics and failed allocations are written with a backtrace to `%LOCALAPPDATA%\ntsc-simulator\crash.log`; panics in background jobs end that job with an error instead of the program. Release builds keep line tables in a separate `.pdb` (`debug = "line-tables-only"`, no longer stripped).
- **GUI video playback**: **Play** / **Pause** (Space) plays the processed video in the preview at the source frame rate, with loop, scrubbing while playing, live setting changes, before/after and zoom. Real time by default (late frames are not shown and, unless the deck noise reduction is on, skipped), or **Every frame**; the status bar shows the achieved fps and dropped frames. Shown frames are byte-identical to a `roundtrip` with the same settings and seed; the roundtrip's frame processing now lives in `pipeline::FrameProcessor`, which both use (output unchanged).
- **No echo outlines from `deck-svhs-tbc` on motion**: thin lines and outlines in moving footage no longer leave crisp copies where they were in earlier frames. The deck noise reduction detected motion from a 3×3 mean of the frame difference, which diluted the difference of thin detail to a third so it counted as still and was blended back frame after frame; it now lowpasses the signed difference, spreads the motion signal over ±2 pixels and ±1 row, and blends luma and colour by separate motion signals, which also cleans still areas more (static luma noise 1.70 -> 0.74 instead of 0.83, colour noise halved). The 3D comb detected motion only against the frame two back, so whatever was only in the previous frame was copied in; with the FM tape models it now also checks the previous frame's luma (classic-model output unchanged). `lq --deck deck-svhs-tbc` video lock updated; a composite variant is locked too.
- **`deck-edit` matches pro-deck EDIT mode**: besides no sharpening it turns the noise canceller and the colour noise reducer off, as the JVC BR-S422 does for dubbing; source enhancement (`--edge-ringing`) and the line noise reducer are left alone.
- **`lq` softer than `mq`/`hq`**: a worn EP tape now uses a 2.5 MHz luma bandwidth (below the 2.8 MHz of a new EP recording) and less source enhancement (0.15–0.25), so its resolution lands below `mq` and `hq` on composite and S-Video for every seed.
- **Research-based VHS model** (opt-in; legacy presets byte-identical, locked by `tests/output_lock.rs`): the FM tape model (`--tape-model fm|rf`) with emphasis, white/dark clips (`--white-clip`, `--dark-clip`), nonlinear emphasis (`--nl-emphasis`), FM noise (`--tape-snr`), dropouts per minute (`--dropouts`, `--dropout-span`), head-contact loss and tracking bars (`--rf-loss`, `--tracking-bars`), luma crosstalk, EP interleave, black/white inversion, head level step and dubs (`--luma-crosstalk`, `--ep-interleave`, `--bw-inversion`, `--head-level-step`, `--generations`); deck noise canceller, line and chroma noise reducers (`--deck-nc`, `--deck-ynr`, `--deck-cnr`), the picture control as a short peaker after them, a luma-only dropout compensator (`--doc`); head switching from the switch point with picture or black after it and top-of-field skew (`--head-switch-point`, `--head-switch-after`, `--flagging`); colour-under noise bandwidth and adjacent-track chroma crosstalk (`--chroma-noise-bw`, `--chroma-crosstalk`); the trap decoder comb and capture peaking (`--comb trap`, `--capture-peaking`); `--reception-before-tape`. `lq`/`mq`/`hq` and the deck profiles use them (EP / LP-like / SP); measured `hq` reaches −3 dB at 2.2 MHz and 45 dB S/N, close to SP decks. Fixes: caption data now on lines 21/284, `--jitter` no longer wraps picture into the sync, `--edge-ringing` relabelled as source enhancement, `--tape-dropout-rate` help corrected. lq/mq/hq run about as fast as before.
- **Head switching rework**: matches a real capture. Only the last 2–3 field lines tear (`--head-switch-lines` default and lq/mq/hq values now 3): a switch line that jumps sideways part-way along with a short switching transient and wide, washed-out light smears, then near-black lines with sparse white, green and purple noise streaks and the odd displaced piece of picture. No more mid-band dark stripe or black wedge on the left. Over a video the switch is a fixed property of the tape with per-field flicker and occasional sudden jumps that hold for a few frames or seconds, instead of sliding and shrinking. Legacy presets are unchanged.
- **3D comb independent of `--threads`**: frame 1 now uses frame 0 as received for both earlier frames, and telecine runs carry the previous batch's frames across batch boundaries, so `--comb 3d` output is the same at any thread count (frame 1 of `--comb 3d` roundtrips changes; nothing else does).
- **Faster processing, same output**: S-Video runs each tape effect and the encoder's band-limiting once instead of per wire, the 3D comb makes each frame's signal once instead of three times, and the colour-under noise, line shifts, deck noise reduction, decoder resize and DVD mastering are vectorised. On one thread at 640x480: `--preset lq` 16.8 -> 28.5 fps, `legacy-lq` 22 -> 40 fps, `lq --deck deck-svhs-tbc` 11.2 -> 25 fps, `lq --comb 3d` 6 -> 25 fps. Output is bit-identical to before.
- **Portable builds**: release builds no longer use `target-cpu=native`, so a binary built on a new CPU runs on older ones (no illegal-instruction crashes); hot loops pick AVX2 or SSE2 at run time. Tape-path, ringing and other FIR filters are vectorised; output is unchanged bit for bit.
- **GUI copy frame and pan limits**: **Copy frame** / Ctrl+C copies the processed preview to the clipboard; zoomed-out frames stay centred and zoomed-in frames cannot be dragged past their edges.

- **GUI wheel zoom**: scroll over the preview to zoom (0.25× to 16×) about the cursor.
- **AV1 input fallback**: AV1 sources decode with libdav1d when ffmpeg has it, otherwise with hardware decoding, so files that libaom rejects (for example an unknown level index) still load; frame-grab errors now include ffmpeg's own message.
- **Playback deck and capture stages**: line TBC, frame TBC with a new vertical-bounce effect, recursive deck noise reduction (sequential stage), deck sharpness control, motion-adaptive 3D comb (`--comb 2sample|1h|3d`), deck profiles (`--deck`, and a "Playback deck" dropdown in the GUI).
- **DVD presets** (`dvd-hq`, `dvd-mq`, `dvd-mq-ee`, `dvd-lq`, `dvd-interlaced`); `--vhs-preset` renamed to `--preset` (old name kept as an alias); roundtrip's x264 `--preset` is now `--x264-preset`.
- **DVD-Video path** (`--format dvd`): mastering (aspect-correct Lanczos fit, pre-filter, DNR, edge enhancement), field-based 4:2:0, MPEG-2 within the DVD-Video limits via ffmpeg with soft pulldown for film, player chroma upsampling error, digital / S-Video / composite player output, kept MPEG-2 streams and `dvd-check`.
- **S-Video connection** (`--video-out svideo`): the encoder can produce separate Y and C wires; the decoder takes luma from the Y wire without a comb filter and chroma from the C wire, so there is no cross-colour or dot crawl. Tape effects run per wire (luma effects on Y, colour effects on C, timing effects identically on both).
- **Lossless, colour-correct video I/O**: lossless RGB output by default (FFV1 `.mkv` / x264 RGB `.mp4`, verified bit-exact), `--lossy` for the old x264 4:2:0 output; input converted with the source's colour matrix and range (`--input-matrix`, `--input-range`); `extract-frame`.
- **GUI** (`ntsc-gui`) with live preview, presets and batch processing; library + CLI split with one shared parameter model; JSON presets (`--preset-file`); `legacy-*` presets.
- **VHS effects**: head switching, top chroma bands, caption data line, colour-under chroma noise, chroma phase jitter, 1H chroma comb, chroma delay, time-base wobble, temporal coherence; `--seed`; `--legacy-burst-phase`; decoder burst-phase sign fix.

## Comb Filter Modes

- **2-sample delay** (default): Cancels chroma from luma using a horizontal delay. Can produce rainbow cross-color on sharp edges, and softens luma (−3 dB at 1.8 MHz).
- **Chroma trap** (`--comb trap`): a notch of ± 0.6 MHz around the subcarrier, as capture chips use for VCR input; luma keeps its detail to about 2.9 MHz.
- **1H line-delay** (`--comb-1h`): Uses the adjacent line from the same field. Reduces rainbow artifacts but introduces hanging-dot patterns on horizontal color transitions as well as some line doubling.
