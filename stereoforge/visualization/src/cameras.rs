//! Camera-to-world extrinsics and pinhole intrinsics are distinct entities.
use rerun::{Color, LineStrips3D, Pinhole, Points3D, RecordingStream, Transform3D};
use rerun::components::{Radius, ViewCoordinates};
use crate::{Result, staging::Thumbnail};
use std::collections::{BTreeMap, BTreeSet};

pub fn log(rec: &RecordingStream, root: &str, cameras: &[f32], ids: &[i64], images: &BTreeMap<i64, Thumbnail>, logged_images: &mut BTreeSet<String>, protected: &BTreeSet<i64>) -> Result<()> {
    let centers: Vec<[f32; 3]> = cameras.chunks_exact(18).map(|c| [c[9], c[10], c[11]]).collect();
    let size = 0.07; // Fixed presentation size in normalized display coordinates.
    let color = if root == "world/map" { Color::from_rgb(50, 200, 255) } else { Color::from_rgb(255, 180, 40) };
    let mut trajectory: Vec<Vec<[f32; 3]>> = Vec::new();
    for (index, c) in cameras.chunks_exact(18).enumerate() {
        if !c.iter().all(|v| v.is_finite()) || c[12] <= 0.0 || c[13] <= 0.0 || c[16] <= 0.0 || c[17] <= 0.0 {
            return Err("Invalid camera supplied to viewer".into());
        }
        if protected.contains(&ids[index]) { continue; }
        let path = format!("world/cameras/{}", ids[index]);
        // Rerun matrices are column-major; our C ABI sends row-major R_c2w.
        let columns = [[c[0], c[3], c[6]], [c[1], c[4], c[7]], [c[2], c[5], c[8]]];
        rec.log(path.as_str(), &Transform3D::from_translation_mat3x3(centers[index], columns))?;
        // Optical coordinates are right/down/forward. The calibrated principal
        // point is explicit; never substitute the image center after global BA.
        let image = images.get(&ids[index]);
        let resolution = image.map(|im| [im.width as f32, im.height as f32]).unwrap_or([c[16],c[17]]);
        let sx = resolution[0]/c[16];
        let sy = resolution[1]/c[17];
        // Thumbnail resizing changes the pixel grid, so scale K with it.
        rec.log(format!("{path}/image"), &Pinhole::new([
            [c[12]*sx, 0.0, 0.0], [0.0, c[13]*sy, 0.0], [c[14]*sx, c[15]*sy, 1.0],
        ]).with_resolution(resolution)
          .with_camera_xyz(ViewCoordinates::RDF)
          .with_image_plane_distance(size).with_color(color))?;
        if let Some(image) = image {
            let image_path = format!("{path}/image");
            if logged_images.insert(image_path.clone()) {
                rec.log(image_path, &rerun::Image::from_rgb24(image.rgb.clone(), [image.width,image.height]))?;
            }
        }
        if index > 0 && ids[index] == ids[index-1]+1 {
            trajectory.push(vec![centers[index-1], centers[index]]);
        }
    }
    rec.log(format!("{root}/camera_centers"), &Points3D::new(centers)
        .with_radii([Radius::new_ui_points(2.0)]).with_colors([color]))?;
    rec.log(format!("{root}/trajectory"), &LineStrips3D::new(trajectory).with_colors([color])
        .with_radii([Radius::new_ui_points(0.75)]))?;
    Ok(())
}
