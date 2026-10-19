//! Runtime CPU dispatch for hot loops.
//!
//! The default build targets baseline x86-64 (SSE2) so it runs on any
//! x86-64 CPU.  [`multiversion!`] compiles a function body twice, once as is
//! and once with AVX2 enabled, and picks the AVX2 copy at run time when the
//! CPU has it.  Both copies perform the same operations in the same order
//! (FMA is not enabled, and the compiler does not reassociate float
//! arithmetic), so they give bit-identical results.
//!
//! Setting the environment variable `NTSC_SIMD=sse2` forces the SSE2
//! copies, to check them on a machine with AVX2.

use std::sync::OnceLock;

/// `true` if the AVX2 copies should run.
pub fn use_avx2() -> bool {
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx2")
                && std::env::var("NTSC_SIMD").map_or(true, |v| !v.eq_ignore_ascii_case("sse2"))
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// Define a function whose body is compiled for SSE2 and for AVX2, with the
/// AVX2 copy chosen at run time.  Write loops in the body so that the
/// compiler can vectorise them (independent lanes, no float reductions).
macro_rules! multiversion {
    ($(#[$meta:meta])* $vis:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block) => {
        $(#[$meta])*
        $vis fn $name($($arg: $ty),*) $(-> $ret)? {
            #[inline(always)]
            fn body($($arg: $ty),*) $(-> $ret)? $body

            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2")]
                unsafe fn avx2($($arg: $ty),*) $(-> $ret)? {
                    body($($arg),*)
                }
                if $crate::simd::use_avx2() {
                    // SAFETY: the CPU supports AVX2.
                    return unsafe { avx2($($arg),*) };
                }
            }
            body($($arg),*)
        }
    };
}

pub(crate) use multiversion;
