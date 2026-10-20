//! Crash reports.  The GUI has no console, so a panic message would be
//! lost: every panic and every failed memory allocation is appended to
//! `crash.log` (in `%LOCALAPPDATA%\ntsc-simulator` on Windows) with the
//! thread, the location and a backtrace.  A panic on the UI thread also
//! shows a message box naming the file; a failed allocation is only
//! logged.  Worker threads catch their own
//! panics (see `catch`) and report them as errors.

use std::alloc::{GlobalAlloc, Layout, System};
use std::any::Any;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// The crash log file (`NTSC_GUI_CRASH_LOG` overrides it).
pub fn log_path() -> PathBuf {
    if let Some(p) = std::env::var_os("NTSC_GUI_CRASH_LOG") {
        return PathBuf::from(p);
    }
    let dir = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))
        .unwrap_or_else(std::env::temp_dir);
    dir.join("ntsc-simulator").join("crash.log")
}

/// UTC time as "YYYY-MM-DD HH:MM:SS".
fn timestamp() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Days since 1970-01-01 to a civil date (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC", y, m, d, rem / 3600, rem / 60 % 60, rem % 60)
}

/// Append one report to the crash log; the path on success.
fn append(what: &str) -> Option<PathBuf> {
    let path = log_path();
    std::fs::create_dir_all(path.parent()?).ok()?;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok()?;
    let thread = std::thread::current();
    let backtrace = std::backtrace::Backtrace::force_capture();
    writeln!(
        f,
        "=== {} ntsc-gui {} (thread {})\n{}\nBacktrace:\n{}\n",
        timestamp(),
        env!("CARGO_PKG_VERSION"),
        thread.name().unwrap_or("unnamed"),
        what,
        backtrace
    )
    .ok()?;
    Some(path)
}

fn message_box(text: &str) {
    rfd::MessageDialog::new()
        .set_level(rfd::MessageLevel::Error)
        .set_title("NTSC Composite Simulator crashed")
        .set_description(text)
        .set_buttons(rfd::MessageButtons::Ok)
        .show();
}

/// The text of a panic payload.
pub fn panic_text(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Install the panic hook.  Panics on the UI thread also show a message
/// box, as the process ends right after.
pub fn install() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info.location().map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column())).unwrap_or_default();
        let text = format!("panic at {}: {}", location, panic_text(info.payload()));
        let path = append(&text);
        default(info);
        if std::thread::current().name() == Some("main") {
            let saved = path.map(|p| format!("Details were saved to\n{}", p.display())).unwrap_or_default();
            message_box(&format!("{}\n\n{}", text, saved));
        }
    }));
}

/// Run `f`, turning a panic into an error message (the panic hook has
/// already logged it).
pub fn catch<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .map_err(|p| format!("internal error: {} (details in {})", panic_text(&*p), log_path().display()))
}

/// The system allocator, logging failed allocations before the process
/// aborts (Rust does not run the panic hook for them).
pub struct LoggingAlloc;

static IN_REPORT: AtomicBool = AtomicBool::new(false);

fn report_alloc_failure(layout: Layout) {
    if IN_REPORT.swap(true, Ordering::SeqCst) {
        return;
    }
    // Only logged: a caller using `try_reserve` may handle the failure.
    let text = format!(
        "memory allocation of {} bytes (align {}) failed; the process aborts unless the caller handles it",
        layout.size(),
        layout.align()
    );
    append(&text);
    IN_REPORT.store(false, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_panics_are_logged_and_caught() {
        let log = std::env::temp_dir().join(format!("ntsc-gui-crash-test-{}.log", std::process::id()));
        let _ = std::fs::remove_file(&log);
        std::env::set_var("NTSC_GUI_CRASH_LOG", &log);
        install();
        let r = std::thread::Builder::new()
            .name("worker".into())
            .spawn(|| catch(|| -> u32 { panic!("boom {}", 7) }))
            .unwrap()
            .join()
            .unwrap();
        let _ = std::panic::take_hook();
        let err = r.unwrap_err();
        assert!(err.contains("boom 7"), "{err}");
        let text = std::fs::read_to_string(&log).unwrap();
        assert!(text.contains("(thread worker)") && text.contains("boom 7") && text.contains("crash.rs"), "{text}");
        assert!(text.contains("Backtrace:"));
        // Times are written as UTC calendar dates.
        assert!(timestamp().ends_with(" UTC") && timestamp().starts_with("20"));
        let _ = std::fs::remove_file(&log);

        // A failed allocation is logged before the null pointer is returned.
        let p = unsafe { LoggingAlloc.alloc(Layout::from_size_align(isize::MAX as usize / 2, 1).unwrap()) };
        assert!(p.is_null());
        let text = std::fs::read_to_string(&log).unwrap();
        assert!(text.contains("memory allocation of 4611686018427387903 bytes"), "{text}");
        let _ = std::fs::remove_file(&log);
    }
}

unsafe impl GlobalAlloc for LoggingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if p.is_null() {
            report_alloc_failure(layout);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if p.is_null() {
            report_alloc_failure(layout);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if p.is_null() {
            report_alloc_failure(Layout::from_size_align(new_size, layout.align()).unwrap_or(layout));
        }
        p
    }
}
