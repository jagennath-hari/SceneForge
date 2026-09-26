//! StereoForge's Rust SDK adapter. Borrowed C buffers are copied before return.
//! ABI calls are serialized by one C++ owner; no upstream Rerun modifications.
use std::{collections::BTreeSet, ffi::{CStr, c_char}, panic::{AssertUnwindSafe, catch_unwind}, slice};
mod cameras;
mod layout;
mod staging;
mod activity;
use rerun::{Color, Points3D, RecordingStream, TextLog, TimeCell, ViewCoordinates};

struct Session {
    recording: RecordingStream,
    step: i64,
    map_fit: Option<([f32;3], f32)>,
    centers: std::collections::BTreeMap<i64,[f32;3]>,
    camera_root: Option<String>,
    staging: staging::Staging,
    logged_images: BTreeSet<String>,
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
        recording.log_static("world", &ViewCoordinates::RDF())?;
        // Static presentation reference survives every stage and timeline seek.
        // It is the normalized display origin, not a surveyed world coordinate.
        recording.log_static("world/origin/axes", &rerun::LineStrips3D::new([
            vec![[0.0_f32,0.0,0.0],[1.0,0.0,0.0]],
            vec![[0.0_f32,0.0,0.0],[0.0,1.0,0.0]],
            vec![[0.0_f32,0.0,0.0],[0.0,0.0,1.0]],
        ]).with_colors([Color::from_rgb(235,75,75), Color::from_rgb(80,210,100), Color::from_rgb(80,140,255)])
          .with_radii([rerun::components::Radius::new_ui_points(1.5)]))?;
        recording.log_static("world/origin/labels", &Points3D::new([
            [0.0_f32,0.0,0.0],[1.1,0.0,0.0],[0.0,1.1,0.0],[0.0,0.0,1.1],
        ]).with_labels(["Display origin", "+X", "+Y", "+Z"])
          .with_colors([Color::from_rgb(220,220,220), Color::from_rgb(235,75,75),
                        Color::from_rgb(80,210,100), Color::from_rgb(80,140,255)])
          .with_radii([rerun::components::Radius::new_ui_points(2.0)]))?;
        recording.log_static("pipeline/layout_legend", &TextLog::new(
            "GRAY cameras = schematic sphere, arbitrary FOV. ORANGE groups = independently normalized VGGT windows. BLUE map = accepted geometry in normalized display coordinates."))?;
        recording.log("pipeline/stage", &TextLog::new(
            "Preparing reconstruction. Selected images appear on a schematic sphere; their camera entities move through VGGT groups into the accepted map."
        ))?;
        recording.flush_async()?;
        let session = Box::new(Session { recording, step: 0, map_fit: None, centers: Default::default(), camera_root: None, staging: staging::Staging::default(), logged_images: BTreeSet::new() });
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
        if handle.is_null() || stage > 4 { return Err("Invalid recording handle/stage".into()); }
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
        // Provisional aligned windows are not yet accepted. Keep their current
        // display positions until commit, rather than jumping to a different fit.
        if stage == 1 { return Ok(()); }
        session.step += 1;
        let rec = &session.recording;
        rec.set_time("reconstruction_step", TimeCell::from_sequence(session.step));
        activity::clear(rec)?;
        if stage == 4 {
            let (offset, scale) = session.map_fit.ok_or("Dense preview requires an accepted map display transform")?;
            let mut positions = Vec::new();
            let mut colors = Vec::new();
            for (point,color) in xyz.chunks_exact(3).zip(rgb.chunks_exact(3)) {
                let p: [f32;3] = std::array::from_fn(|a| point[a]*scale+offset[a]);
                if p.iter().all(|v| v.is_finite()) && p.iter().map(|v| v*v).sum::<f32>() <= 100.0 {
                    positions.push(p); colors.push(Color::from_rgb(color[0],color[1],color[2]));
                }
            }
            rec.log("world/map/dense", &Points3D::new(positions).with_colors(colors)
                .with_radii([rerun::components::Radius::new_ui_points(1.0)]))?;
            rec.flush_async()?;
            return Ok(());
        }
        let (root, label) = match stage {
            0 => (session.staging.window(ids), "VGGT group — schematic placement, independent gauge"),
            2 => ("world/map".to_owned(), "Accepted common map"),
            _ => ("world/map".to_owned(), "Accepted shared-calibration global BA"),
        };
        if stage >= 2 { session.staging.retire(rec, ids)?; }
        let (display_xyz, display_cameras, fit) = staging::fit(xyz, cameras, ids, stage == 0);
        if stage >= 2 { session.map_fit = Some(fit); }
        for (id,camera) in ids.iter().zip(display_cameras.chunks_exact(18)) {
            if stage >= 2 || !session.staging.accepted.contains(id) {
                session.centers.insert(*id,[camera[9],camera[10],camera[11]]);
            }
        }
        let xyz = display_xyz.as_slice();
        let cameras = display_cameras.as_slice();
        rec.log("pipeline/stage", &TextLog::new(format!("{label}: {camera_count} cameras, {points} displayed points")))?;
        // Exclude extreme display outliers without altering saved geometry.
        // Filter positions and colors together to preserve their association.
        let visible: Vec<(&[f32], &[u8])> = xyz.chunks_exact(3).zip(rgb.chunks_exact(3))
            .filter(|(p, _)| p.iter().all(|v| v.is_finite()) && p.iter().map(|v| v*v).sum::<f32>() <= 100.0)
            .collect();
        let positions: Vec<[f32; 3]> = visible.iter().map(|(p,_)| [p[0],p[1],p[2]]).collect();
        let colors: Vec<Color> = visible.iter().map(|(_,c)| Color::from_rgb(c[0],c[1],c[2])).collect();
        rec.log(format!("{root}/points"), &Points3D::new(positions).with_colors(colors)
            .with_radii([rerun::components::Radius::new_ui_points(1.5)]))?;

        let protected = if stage < 2 { session.staging.accepted.clone() } else { BTreeSet::new() };
        cameras::log(rec, &root, cameras, ids, &session.staging.images, &mut session.logged_images, &protected)?;
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
        if session.camera_root.is_some() {
            // Calibrated camera images were logged with their correctly scaled K
            // by cameras::log. Avoid overwriting thumbnail-sized image planes.
            if !session.staging.images.contains_key(&frame) {
                session.recording.log(format!("world/cameras/{frame}/image"),
                    &rerun::Image::from_rgb24(bytes, [width, height]))?;
            }
        } else {
            session.step += 1;
            session.recording.set_time("reconstruction_step", TimeCell::from_sequence(session.step));
            session.staging.keyframe(&session.recording, frame, staging::Thumbnail { rgb: bytes, width, height })?;
        }
        session.recording.flush_async()?;
        Ok(())
    })
}

// Stage text only: frontend matches are not rendered in this presentation.
#[unsafe(no_mangle)]
pub extern "C" fn sf_rerun_status(handle: *mut std::ffi::c_void, message: *const c_char,
    error: *mut c_char, capacity: usize) -> i32 {
    guarded(error, capacity, || {
        if handle.is_null() || message.is_null() { return Err("Invalid frontend event".into()); }
        // SAFETY: C++ retains the terminated UTF-8 message and exclusive session.
        let session = unsafe { &mut *handle.cast::<Session>() };
        let message = unsafe { CStr::from_ptr(message) }.to_str()?;
        session.step += 1;
        session.camera_root = None;
        session.recording.set_time("reconstruction_step", TimeCell::from_sequence(session.step));
        activity::log(&session.recording, message, &session.centers, &session.staging.images)?;
        session.recording.log("pipeline/stage", &TextLog::new(message))?;
        session.recording.flush_async()?;
        Ok(())
    })
}
