//! StereoForge's Rust SDK adapter. Borrowed C buffers are copied before return.
//! ABI calls are serialized by one C++ owner; no upstream Rerun modifications.
use std::{ffi::{CStr, c_char}, panic::{AssertUnwindSafe, catch_unwind}, slice};
mod cameras;
mod layout;
use rerun::{Clear, Color, Points3D, RecordingStream, TextLog, TimeCell, ViewCoordinates};

struct Session {
    recording: RecordingStream,
    step: i64,
    camera_root: Option<String>,
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn guarded(error: *mut c_char, capacity: usize, operation: impl FnOnce() -> Result<()>) -> i32 {
    let message = match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => return 0,
        Ok(Err(error)) => error.to_string(),
        Err(_) => "Rust visualization panicked".to_owned(),
    };
    if !error.is_null() && capacity > 0 {
        let bytes = message.as_bytes();
        let count = bytes.len().min(capacity - 1);
        // SAFETY: the caller supplies a writable buffer of `capacity` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), error.cast::<u8>(), count);
            *error.add(count) = 0;
        }
    }
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn sf_rerun_open(path: *const c_char, output: *mut *mut std::ffi::c_void,
                                error: *mut c_char, capacity: usize) -> i32 {
    guarded(error, capacity, || {
        if path.is_null() || output.is_null() { return Err("Null recording path/handle".into()); }
        // SAFETY: C++ supplies a terminated UTF-8 path and writable handle.
        let path = unsafe { CStr::from_ptr(path) }.to_str()?;
        // Launch the official viewer (or reuse its existing server). Use the
        // actual port returned by the SDK, not an assumed localhost port.
        let viewer = rerun::spawn(&rerun::SpawnOptions::default())?;
        let uri = format!("rerun+http://127.0.0.1:{}/proxy", viewer.port).parse()?;
        let recording = rerun::RecordingStreamBuilder::new("StereoForge").with_blueprint(layout::blueprint()).set_sinks((
            rerun::sink::FileSink::new(path)?,
            rerun::sink::GrpcSink::new(uri),
        ))?;
        // Display convention only: no gravity alignment or coordinate change.
        recording.log_static("world", &ViewCoordinates::RDF)?;
        recording.log_static("local_window", &ViewCoordinates::RDF)?;
        recording.log("pipeline/stage", &TextLog::new(
            "Preparing reconstruction. Images/features appear first; each completed VGGT window has independent coordinates."
        ))?;
        recording.flush_async()?;
        let session = Box::new(Session { recording, step: 0, camera_root: None });
        unsafe { *output = Box::into_raw(session).cast(); }
        Ok(())
    })
}

// Arrays: XYZ/RGB triples; camera records are 18 floats: row-major camera-to-world
// rotation (9), center (3), fx/fy/cx/cy (4), width/height (2). IDs are keyframe IDs.
#[unsafe(no_mangle)]
pub extern "C" fn sf_rerun_snapshot(handle: *mut std::ffi::c_void, stage: u32,
    xyz: *const f32, rgb: *const u8, points: usize,
    cameras: *const f32, ids: *const i64, camera_count: usize,
    error: *mut c_char, capacity: usize) -> i32 {
    guarded(error, capacity, || {
        if handle.is_null() || stage > 3 { return Err("Invalid recording handle/stage".into()); }
        if (points > 0 && (xyz.is_null() || rgb.is_null())) ||
            (camera_count > 0 && (cameras.is_null() || ids.is_null())) {
            return Err("Null snapshot buffer".into());
        }
        let point_len = points.checked_mul(3).ok_or("Point buffer overflow")?;
        let camera_len = camera_count.checked_mul(18).ok_or("Camera buffer overflow")?;
        // SAFETY: C++ guarantees these buffer lengths and exclusive session access.
        let session = unsafe { &mut *handle.cast::<Session>() };
        let xyz = if points == 0 { &[] } else { unsafe { slice::from_raw_parts(xyz, point_len) } };
        let rgb = if points == 0 { &[] } else { unsafe { slice::from_raw_parts(rgb, point_len) } };
        let cameras = if camera_count == 0 { &[] } else { unsafe { slice::from_raw_parts(cameras, camera_len) } };
        let ids = if camera_count == 0 { &[] } else { unsafe { slice::from_raw_parts(ids, camera_count) } };
        session.step += 1;
        let rec = &session.recording;
        rec.set_time("reconstruction_step", TimeCell::from_sequence(session.step));
        let (root, label) = match stage {
            0 => ("local_window", "VGGT initialization (independent coordinates)"),
            1 => ("world/incoming", "Locally refined window after Sim(3); provisional"),
            2 => ("world/map", "Accepted common map"),
            _ => ("world/map", "Accepted shared-calibration global BA"),
        };
        rec.log("local_window", &Clear::recursive())?;
        rec.log("world/incoming", &Clear::recursive())?;
        rec.log(root, &Clear::recursive())?;
        rec.log("pipeline/stage", &TextLog::new(format!("{label}: {camera_count} cameras, {points} displayed points")))?;
        let positions: Vec<[f32; 3]> = xyz.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        let colors: Vec<Color> = rgb.chunks_exact(3).map(|c| Color::from_rgb(c[0], c[1], c[2])).collect();
        rec.log(format!("{root}/points"), &Points3D::new(positions).with_colors(colors))?;

        cameras::log(rec, root, cameras, ids)?;
        session.camera_root = Some(root.to_owned());
        rec.flush_async()?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn sf_rerun_close(handle: *mut std::ffi::c_void) {
    if !handle.is_null() {
        // SAFETY: only the owning C++ object calls close, exactly once.
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let session = unsafe { Box::from_raw(handle.cast::<Session>()) };
            // Bound shutdown waiting if the live viewer has been closed.
            if let Err(error) = session.recording.flush_with_timeout(std::time::Duration::from_secs(5)) {
                eprintln!("Rerun shutdown flush: {error}");
            }
            drop(session);
        }));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sf_rerun_image(handle: *mut std::ffi::c_void, rgb: *const u8,
    width: u32, height: u32, frame: i64, error: *mut c_char, capacity: usize) -> i32 {
    guarded(error, capacity, || {
        if handle.is_null() || rgb.is_null() || width == 0 || height == 0 {
            return Err("Invalid image buffer".into());
        }
        let count = (width as usize).checked_mul(height as usize)
            .and_then(|n| n.checked_mul(3)).ok_or("Image size overflow")?;
        // SAFETY: C++ validates packed RGB length before passing this buffer.
        let session = unsafe { &mut *handle.cast::<Session>() };
        let bytes = unsafe { slice::from_raw_parts(rgb, count) }.to_vec();
        session.recording.set_time("reconstruction_step", TimeCell::from_sequence(session.step));
        session.recording.log("frontend", &Clear::recursive())?;
        session.recording.log("frontend/image", &rerun::Image::from_rgb24(bytes.clone(), [width, height]))?;
        if let Some(root) = &session.camera_root {
            session.recording.log(format!("{root}/cameras/{frame}/image"),
                &rerun::Image::from_rgb24(bytes, [width, height]))?;
        }
        session.recording.log("pipeline/image_info", &TextLog::new(format!("Window representative keyframe {frame}; processed RGB")))?;
        session.recording.flush_async()?;
        Ok(())
    })
}

// Typed frontend event; no JSON or Python Rerun SDK. Optional packed RGB,
// pixel coordinates (Nx2), and verified match segments (Nx4) share one canvas.
#[unsafe(no_mangle)]
pub extern "C" fn sf_rerun_event(handle: *mut std::ffi::c_void, message: *const c_char,
    rgb: *const u8, width: u32, height: u32, xy: *const f32, point_count: usize,
    segments: *const f32, segment_count: usize, error: *mut c_char, capacity: usize) -> i32 {
    guarded(error, capacity, || {
        if handle.is_null() || message.is_null() { return Err("Invalid frontend event".into()); }
        let count = (width as usize).checked_mul(height as usize).and_then(|n| n.checked_mul(3)).ok_or("Image overflow")?;
        let point_len = point_count.checked_mul(2).ok_or("Point overflow")?;
        let line_len = segment_count.checked_mul(4).ok_or("Match overflow")?;
        if (count > 0 && rgb.is_null()) || (point_len > 0 && xy.is_null()) || (line_len > 0 && segments.is_null()) {
            return Err("Null frontend buffer".into());
        }
        // SAFETY: StereoForge's binding validates shapes and retains all buffers
        // through this call. SDK archetypes own copies before the call returns.
        let session = unsafe { &mut *handle.cast::<Session>() };
        let message = unsafe { CStr::from_ptr(message) }.to_str()?;
        session.step += 1;
        session.camera_root = None;
        let rec = &session.recording;
        rec.set_time("reconstruction_step", TimeCell::from_sequence(session.step));
        rec.log("pipeline/stage", &TextLog::new(message))?;
        if count > 0 {
            rec.log("frontend", &Clear::recursive())?;
            let bytes = unsafe { slice::from_raw_parts(rgb, count) }.to_vec();
            rec.log("frontend/image", &rerun::Image::from_rgb24(bytes, [width, height]))?;
        }
        if point_len > 0 {
            let points: Vec<[f32; 2]> = unsafe { slice::from_raw_parts(xy, point_len) }
                .chunks_exact(2).map(|p| [p[0], p[1]]).collect();
            rec.log("frontend/features", &rerun::Points2D::new(points)
                .with_colors([Color::from_rgb(80, 255, 80)]).with_radii([2.0]))?;
        }
        if line_len > 0 {
            let lines: Vec<[[f32; 2]; 2]> = unsafe { slice::from_raw_parts(segments, line_len) }
                .chunks_exact(4).map(|p| [[p[0], p[1]], [p[2], p[3]]]).collect();
            rec.log("frontend/matches", &rerun::LineStrips2D::new(lines)
                .with_colors([Color::from_rgb(80, 255, 80)]))?;
        }
        rec.flush_async()?;
        Ok(())
    })
}
