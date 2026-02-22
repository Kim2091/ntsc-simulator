"""SMPTE color bar test pattern generator."""

import numpy as np


def generate_colorbars(width=640, height=480):
    """Generate a standard SMPTE color bar test pattern.

    The pattern consists of 7 vertical bars (left to right):
    White, Yellow, Cyan, Green, Magenta, Red, Blue

    Args:
        width: Output image width.
        height: Output image height.

    Returns:
        RGB frame as numpy array (height x width x 3, uint8).
    """
    # SMPTE color bar colors at 75% amplitude (standard)
    # Order: White, Yellow, Cyan, Green, Magenta, Red, Blue
    colors_75 = np.array([
        [191, 191, 191],   # White (75%)
        [191, 191,   0],   # Yellow
        [  0, 191, 191],   # Cyan
        [  0, 191,   0],   # Green
        [191,   0, 191],   # Magenta
        [191,   0,   0],   # Red
        [  0,   0, 191],   # Blue
    ], dtype=np.uint8)

    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Main color bars (top 2/3)
    bar_height = height * 2 // 3
    bar_width = width // 7

    for i, color in enumerate(colors_75):
        x_start = i * bar_width
        x_end = (i + 1) * bar_width if i < 6 else width
        frame[0:bar_height, x_start:x_end] = color

    # Middle section: reverse-order castellations (1/12 of height)
    strip_height = height // 12
    strip_top = bar_height

    reverse_colors = np.array([
        [  0,   0, 191],   # Blue
        [  0,   0,   0],   # Black
        [191,   0, 191],   # Magenta
        [  0,   0,   0],   # Black
        [  0, 191, 191],   # Cyan
        [  0,   0,   0],   # Black
        [191, 191, 191],   # White
    ], dtype=np.uint8)

    for i, color in enumerate(reverse_colors):
        x_start = i * bar_width
        x_end = (i + 1) * bar_width if i < 6 else width
        frame[strip_top:strip_top + strip_height, x_start:x_end] = color

    # Bottom section: SMPTE EG 1-1990 PLUGE layout
    # | -I  | White | +Q  | Black | sub-black | black | above-black | black |
    # Columns 0-2 aligned with bars 0-2, column 3-6 aligned with bars 3-6
    pluge_top = strip_top + strip_height

    # -I signal (approximation in RGB)
    x0 = 0
    x1 = bar_width
    frame[pluge_top:height, x0:x1] = [0, 68, 130]

    # 100% White
    x0 = bar_width
    x1 = 2 * bar_width
    frame[pluge_top:height, x0:x1] = [255, 255, 255]

    # +Q signal (approximation in RGB)
    x0 = 2 * bar_width
    x1 = 3 * bar_width
    frame[pluge_top:height, x0:x1] = [67, 0, 130]

    # Black fill from bar 3 through bar 6 (remainder of width)
    x0 = 3 * bar_width
    frame[pluge_top:height, x0:] = [16, 16, 16]

    # PLUGE sub-bars within bar 5's width: superblack, black, above-black
    pluge_left = 4 * bar_width
    pluge_right = 5 * bar_width
    sub_width = (pluge_right - pluge_left) // 3
    # Superblack (-4 IRE below setup)
    frame[pluge_top:height, pluge_left:pluge_left + sub_width] = [1, 1, 1]
    # Black reference (7.5 IRE setup)
    frame[pluge_top:height, pluge_left + sub_width:pluge_left + 2 * sub_width] = [16, 16, 16]
    # Slightly above black (+4 IRE above setup, ~11.5 IRE)
    frame[pluge_top:height, pluge_left + 2 * sub_width:pluge_right] = [33, 33, 33]

    return frame
