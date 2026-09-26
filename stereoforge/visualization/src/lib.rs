//! StereoForge's Rust SDK adapter. Borrowed C buffers are copied before return.
//! ABI calls are serialized by one C++ owner; no upstream Rerun modifications.
use std::{ffi::{CStr, c_char}, panic::{AssertUnwindSafe, catch_unwind}, slice};
use rerun::{Clear, Color, LineStrips3D, Points3D, RecordingStream, TextLog, TimeCell};

struct Session {
    recording: RecordingStream,
    step: i64,
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
        let recording = rerun::RecordingStreamBuilder::new("StereoForge").set_sinks((
            rerun::sink::FileSink::new(path)?,
            rerun::sink::GrpcSink::new(uri),
        ))?;
        recording.log("pipeline/stage", &TextLog::new(
            "Preparing reconstruction. Map updates begin after VGGT window inference."
        ))?;
        recording.flush_async()?;
        let session = Box::new(Session { recording, step: 0 });
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

        // OpenCV camera coordinates: +X right, +Y down, +Z forward. Construct
        // frustums from K and camera-to-world directly; never invert the pose.
        let centers: Vec<[f32; 3]> = cameras.chunks_exact(18).map(|c| [c[9], c[10], c[11]]).collect();
        let mut steps: Vec<f32> = centers.windows(2).map(|p| {
            ((p[0][0]-p[1][0]).powi(2)+(p[0][1]-p[1][1]).powi(2)+(p[0][2]-p[1][2]).powi(2)).sqrt()
        }).filter(|v| v.is_finite() && *v > 0.0).collect();
        steps.sort_by(f32::total_cmp);
        let size = if steps.is_empty() { 0.1 } else { steps[steps.len()/2]*0.5 };
        let mut frustums: Vec<Vec<[f32; 3]>> = Vec::new();
        let mut trajectory: Vec<Vec<[f32; 3]>> = Vec::new();
        for (index, c) in cameras.chunks_exact(18).enumerate() {
            if !c.iter().all(|v| v.is_finite()) || c[12] <= 0.0 || c[13] <= 0.0 {
                return Err("Invalid camera supplied to viewer".into());
            }
            let center = centers[index];
            let mut corners = Vec::new();
            for [u, v] in [[0.0, 0.0], [c[16], 0.0], [c[16], c[17]], [0.0, c[17]]] {
                let ray = [(u-c[14])/c[12]*size, (v-c[15])/c[13]*size, size];
                let p: [f32; 3] = std::array::from_fn(|r| center[r]+c[3*r]*ray[0]+c[3*r+1]*ray[1]+c[3*r+2]*ray[2]);
                corners.push(p);
                frustums.push(vec![center, p]);
            }
            corners.push(corners[0]);
            frustums.push(corners);
            if index > 0 && ids[index] == ids[index-1]+1 {
                trajectory.push(vec![centers[index-1], center]);
            }
        }
        rec.log(format!("{root}/cameras"), &Points3D::new(centers)
            .with_labels(ids.iter().map(ToString::to_string)).with_colors([Color::from_rgb(255, 180, 40)]))?;
        rec.log(format!("{root}/frustums"), &LineStrips3D::new(frustums).with_colors([Color::from_rgb(255, 180, 40)]))?;
        rec.log(format!("{root}/trajectory"), &LineStrips3D::new(trajectory).with_colors([Color::from_rgb(50, 200, 255)]))?;
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
        session.recording.log("active_keyframe/image", &rerun::Image::from_rgb24(bytes, [width, height]))?;
        session.recording.log("active_keyframe/info", &TextLog::new(format!("Window representative keyframe {frame}; processed RGB")))?;
        session.recording.flush_async()?;
        Ok(())
    })
}
