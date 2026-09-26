//! Bounded frontend activity in the schematic scene, not estimated 3D tracks.
use rerun::{Clear, Color, LineStrips3D, Points3D, RecordingStream};
use rerun::components::Radius;
use crate::{Result, staging::sphere};

pub fn clear(rec: &RecordingStream) -> Result<()> {
    rec.log("world/activity", &Clear::recursive())?;
    Ok(())
}

pub fn log(rec: &RecordingStream, message: &str, centers: &std::collections::BTreeMap<i64,[f32;3]>) -> Result<()> {
    clear(rec)?;
    let position = |id: i64| centers.get(&id).copied().unwrap_or_else(|| sphere(id,9.0));
    let payload = serde_json::from_str::<serde_json::Value>(message).ok();
    let label = payload.as_ref().and_then(|v| v["label"].as_str()).unwrap_or(message);
    // One in-scene label, replacing the previous activity rather than accumulating.
    rec.log("world/activity/status", &Points3D::new([[0.0_f32,-10.0,0.0]])
        .with_labels([label]).with_colors([Color::from_rgb(220,220,220)])
        .with_radii([Radius::new_ui_points(1.0)]))?;
    if let Some(payload) = payload {
        let ready = payload["ready"].as_bool().unwrap_or(false);
        let color = if ready { Color::from_rgb(70,220,140) } else { Color::from_rgb(255,190,55) };
        if let Some(frames) = payload["frames"].as_array() {
            let centers: Vec<[f32;3]> = frames.iter().filter_map(|f| f.as_i64())
                .take(64).map(position).collect();
            rec.log("world/activity/cameras", &Points3D::new(centers).with_colors([color])
                .with_radii([Radius::new_ui_points(5.0)]))?;
        }
        if let Some(pairs) = payload["pairs"].as_array() {
            let lines: Vec<Vec<[f32;3]>> = pairs.iter().take(16).filter_map(|pair| {
                Some(vec![position(pair.get(0)?.as_i64()?),position(pair.get(1)?.as_i64()?)])
            }).collect();
            rec.log("world/activity/connections", &LineStrips3D::new(lines).with_colors([color])
                .with_radii([Radius::new_ui_points(1.0)]))?;
        }
    }
    Ok(())
}
