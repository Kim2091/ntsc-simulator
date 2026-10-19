//! SMPTE color bar test pattern generator.

/// Generate a standard SMPTE color bar test pattern.
///
/// Returns an RGB24 buffer (height * width * 3 bytes).
///
/// Layout:
/// - Top 2/3: 7 vertical bars at 75% amplitude
/// - Middle 1/12: Reverse castellations
/// - Bottom: PLUGE layout
pub fn generate_colorbars(width: usize, height: usize) -> Vec<u8> {
    let mut frame = vec![0u8; width * height * 3];

    // 75% amplitude color bars: White, Yellow, Cyan, Green, Magenta, Red, Blue
    let colors_75: [[u8; 3]; 7] = [
        [191, 191, 191], // White
        [191, 191, 0],   // Yellow
        [0, 191, 191],   // Cyan
        [0, 191, 0],     // Green
        [191, 0, 191],   // Magenta
        [191, 0, 0],     // Red
        [0, 0, 191],     // Blue
    ];

    let bar_height = height * 2 / 3;
    let bar_width = width / 7;

    // Top 2/3: main color bars
    for i in 0..7 {
        let x_start = i * bar_width;
        let x_end = if i < 6 { (i + 1) * bar_width } else { width };
        let color = colors_75[i];
        for y in 0..bar_height {
            for x in x_start..x_end {
                let idx = (y * width + x) * 3;
                frame[idx] = color[0];
                frame[idx + 1] = color[1];
                frame[idx + 2] = color[2];
            }
        }
    }

    // Middle 1/12: reverse-order castellations
    let strip_height = height / 12;
    let strip_top = bar_height;

    let reverse_colors: [[u8; 3]; 7] = [
        [0, 0, 191],     // Blue
        [0, 0, 0],       // Black
        [191, 0, 191],   // Magenta
        [0, 0, 0],       // Black
        [0, 191, 191],   // Cyan
        [0, 0, 0],       // Black
        [191, 191, 191], // White
    ];

    for i in 0..7 {
        let x_start = i * bar_width;
        let x_end = if i < 6 { (i + 1) * bar_width } else { width };
        let color = reverse_colors[i];
        for y in strip_top..strip_top + strip_height {
            for x in x_start..x_end {
                let idx = (y * width + x) * 3;
                frame[idx] = color[0];
                frame[idx + 1] = color[1];
                frame[idx + 2] = color[2];
            }
        }
    }

    // Bottom section: PLUGE layout
    let pluge_top = strip_top + strip_height;

    let fill_rect = |frame: &mut [u8], y0: usize, y1: usize, x0: usize, x1: usize, color: [u8; 3]| {
        for y in y0..y1 {
            for x in x0..x1 {
                let idx = (y * width + x) * 3;
                frame[idx] = color[0];
                frame[idx + 1] = color[1];
                frame[idx + 2] = color[2];
            }
        }
    };

    // -I signal (approximation in RGB)
    fill_rect(&mut frame, pluge_top, height, 0, bar_width, [0, 68, 130]);

    // 100% White
    fill_rect(&mut frame, pluge_top, height, bar_width, 2 * bar_width, [255, 255, 255]);

    // +Q signal (approximation in RGB)
    fill_rect(&mut frame, pluge_top, height, 2 * bar_width, 3 * bar_width, [67, 0, 130]);

    // Black fill from bar 3 through end
    fill_rect(&mut frame, pluge_top, height, 3 * bar_width, width, [16, 16, 16]);

    // PLUGE sub-bars within bar 5's width: superblack, black, above-black
    let pluge_left = 4 * bar_width;
    let pluge_right = 5 * bar_width;
    let sub_width = (pluge_right - pluge_left) / 3;

    // Superblack (-4 IRE below setup)
    fill_rect(&mut frame, pluge_top, height, pluge_left, pluge_left + sub_width, [1, 1, 1]);
    // Black reference (7.5 IRE setup)
    fill_rect(&mut frame, pluge_top, height, pluge_left + sub_width, pluge_left + 2 * sub_width, [16, 16, 16]);
    // Slightly above black (+4 IRE above setup)
    fill_rect(&mut frame, pluge_top, height, pluge_left + 2 * sub_width, pluge_right, [33, 33, 33]);

    frame
}
