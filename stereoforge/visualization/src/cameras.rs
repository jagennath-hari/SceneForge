//! Camera-to-world extrinsics and pinhole intrinsics are distinct entities.
use rerun::{Color, LineStrips3D, Pinhole, Points3D, RecordingStream, Transform3D, TransformAxes3D, ViewCoordinates};
use crate::Result;

pub fn log(rec: &RecordingStream, root: &str, cameras: &[f32], ids: &[i64]) -> Result<()> {
    let centers: Vec<[f32; 3]> = cameras.chunks_exact(18).map(|c| [c[9], c[10], c[11]]).collect();
    let mut steps: Vec<f32> = centers.windows(2).map(|p| {
        ((p[0][0]-p[1][0]).powi(2)+(p[0][1]-p[1][1]).powi(2)+(p[0][2]-p[1][2]).powi(2)).sqrt()
    }).filter(|v| v.is_finite() && *v > 0.0).collect();
    steps.sort_by(f32::total_cmp);
    let size = if steps.is_empty() { 0.1 } else { steps[steps.len()/2]*0.5 };
    let color = if root == "world/map" { Color::from_rgb(50, 200, 255) } else { Color::from_rgb(255, 180, 40) };
    let mut trajectory: Vec<Vec<[f32; 3]>> = Vec::new();
    for (index, c) in cameras.chunks_exact(18).enumerate() {
        if !c.iter().all(|v| v.is_finite()) || c[12] <= 0.0 || c[13] <= 0.0 || c[16] <= 0.0 || c[17] <= 0.0 {
            return Err("Invalid camera supplied to viewer".into());
        }
        let path = format!("{root}/cameras/{}", ids[index]);
        // Rerun matrices are column-major; our C ABI sends row-major R_c2w.
        let columns = [[c[0], c[3], c[6]], [c[1], c[4], c[7]], [c[2], c[5], c[8]]];
        rec.log(&path, &Transform3D::from_translation_mat3x3(centers[index], columns))?;
        rec.log(&path, &TransformAxes3D::new(size))?;
        // Optical coordinates are right/down/forward. The calibrated principal
        // point is explicit; never substitute the image center after global BA.
        rec.log(format!("{path}/image"), &Pinhole::new([
            [c[12], 0.0, 0.0], [0.0, c[13], 0.0], [c[14], c[15], 1.0],
        ]).with_resolution([c[16], c[17]])
          .with_camera_xyz(ViewCoordinates::RDF)
          .with_image_plane_distance(size).with_color(color))?;
        if index > 0 && ids[index] == ids[index-1]+1 {
            trajectory.push(vec![centers[index-1], centers[index]]);
        }
    }
    rec.log(format!("{root}/camera_centers"), &Points3D::new(centers)
        .with_labels(ids.iter().map(ToString::to_string)).with_colors([color]))?;
    rec.log(format!("{root}/trajectory"), &LineStrips3D::new(trajectory).with_colors([color]))?;
    Ok(())
}
