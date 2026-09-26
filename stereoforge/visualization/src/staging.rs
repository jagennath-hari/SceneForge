//! Display-only coordinates. Reconstruction buffers and calibration are never changed.
use std::collections::{BTreeMap, BTreeSet};
use rerun::{Clear, Color, Pinhole, RecordingStream, Transform3D};
use rerun::components::ViewCoordinates;
use crate::Result;

pub struct Thumbnail {
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

// Stable pseudo-random sphere placement, independent of arrival order/count.
pub fn sphere(id: i64, radius: f32) -> [f32; 3] {
    let seed = (id as u64).wrapping_add(1).wrapping_mul(0x9e3779b97f4a7c15);
    let z = 2.0 * ((seed >> 32) as u32 as f64 / u32::MAX as f64) - 1.0;
    let angle = (seed as u32 as f64 / u32::MAX as f64) * std::f64::consts::TAU;
    let r = (1.0-z*z).max(0.0).sqrt();
    [radius*(r*angle.cos()) as f32, radius*z as f32, radius*(r*angle.sin()) as f32]
}

// Shared with activity rendering so feature markers land on the actual image plane.
pub fn schematic_pose(frame: i64) -> ([f32;3], [[f32;3];3]) {
        let center = sphere(frame, 9.0);
        let forward = center.map(|v| -v/9.0);
        let up = if forward[1].abs() > 0.99 { [1.0,0.0,0.0] } else { [0.0,-1.0,0.0] };
        let mut right = [forward[1]*up[2]-forward[2]*up[1], forward[2]*up[0]-forward[0]*up[2], forward[0]*up[1]-forward[1]*up[0]];
        let norm = right.iter().map(|v| v*v).sum::<f32>().sqrt();
        right = right.map(|v| v/norm);
        let down = [forward[1]*right[2]-forward[2]*right[1], forward[2]*right[0]-forward[0]*right[2], forward[0]*right[1]-forward[1]*right[0]];
    (center, [right,down,forward])
}

pub fn image_pixel(frame: i64, uv: [f32;2], image: &Thumbnail) -> [f32;3] {
    let (center, basis) = schematic_pose(frame);
    let local = [(uv[0]-0.5)*0.07,
                 (uv[1]-0.5)*0.07*image.height as f32/image.width as f32, 0.07];
    std::array::from_fn(|axis| center[axis]+(0..3).map(|i| basis[i][axis]*local[i]).sum::<f32>())
}

#[derive(Default)]
pub struct Staging {
    pub images: BTreeMap<i64, Thumbnail>,
    windows: BTreeMap<String, Vec<i64>>,
    pub accepted: BTreeSet<i64>,
}

impl Staging {
    pub fn keyframe(&mut self, rec: &RecordingStream, frame: i64, image: Thumbnail) -> Result<()> {
        let path = format!("world/cameras/{frame}");
        let (center, [right, down, forward]) = schematic_pose(frame);
        // Updating a processed thumbnail must not re-emit a camera pose.
        if !self.images.contains_key(&frame) {
            rec.log(path.as_str(), &Transform3D::from_translation_mat3x3(center, [right,down,forward]))?;
        }
        // Arbitrary presentation FOV until VGGT supplies measured intrinsics.
        rec.log(format!("{path}/image"), &Pinhole::from_focal_length_and_resolution(
            [image.width as f32, image.width as f32], [image.width as f32,image.height as f32])
            .with_camera_xyz(ViewCoordinates::RDF).with_image_plane_distance(0.07)
            .with_color(Color::from_rgb(160,160,160)))?;
        rec.log(format!("{path}/image"), &rerun::Image::from_rgb24(image.rgb.clone(), [image.width,image.height]))?;
        self.images.insert(frame,image);
        Ok(())
    }

    pub fn window(&mut self, ids: &[i64]) -> String {
        let key = format!("{:08}_{:08}", ids.first().copied().unwrap_or(0), ids.last().copied().unwrap_or(0));
        self.windows.insert(key.clone(),ids.to_vec());
        format!("world/windows/{key}")
    }

    pub fn retire(&mut self, rec: &RecordingStream, ids: &[i64]) -> Result<()> {
        self.accepted.extend(ids.iter().copied());
        let complete: Vec<String> = self.windows.iter()
            .filter(|(_, frames)| frames.iter().all(|id| self.accepted.contains(id)))
            .map(|(key,_)| key.clone()).collect();
        for key in complete {
            rec.log(format!("world/windows/{key}"), &Clear::recursive())?;
            self.windows.remove(&key);
        }
        Ok(())
    }
}

/// Robust display fit: ignore extreme point outliers when choosing scene extent.
/// Camera rotations and intrinsics stay intact; the caller filters display outliers.
pub fn fit(xyz: &[f32], cameras: &[f32], ids: &[i64], window: bool) -> (Vec<f32>, Vec<f32>, ([f32;3], f32)) {
    let mut center = [0.0;3];
    for axis in 0..3 {
        let mut values: Vec<f32> = cameras.chunks_exact(18).map(|c| c[9+axis]).filter(|v| v.is_finite()).collect();
        values.sort_by(f32::total_cmp);
        if !values.is_empty() { center[axis] = values[values.len()/2]; }
    }
    let distance = |p: &[f32]| (0..3).map(|a| (p[a]-center[a]).powi(2)).sum::<f32>().sqrt();
    let mut radii: Vec<f32> = xyz.chunks_exact(3).chain(cameras.chunks_exact(18).map(|c| &c[9..12]))
        .map(distance).filter(|r| r.is_finite()).collect();
    radii.sort_by(f32::total_cmp);
    let extent = if radii.is_empty() { 1.0 } else { radii[(radii.len()-1)*95/100].max(0.001) };
    // Include every camera in the fit so camera entities never expand the bounds.
    let extent = cameras.chunks_exact(18).map(|c| distance(&c[9..12]))
        .filter(|r| r.is_finite()).fold(extent, f32::max);
    let radius = if window { 2.0 } else { 7.0 };
    let offset = if window { sphere(ids.first().copied().unwrap_or(0), 5.0) } else { [0.0;3] };
    let transform = |p: &[f32]| -> [f32;3] {
        let scale = radius/extent;
        std::array::from_fn(|a| (p[a]-center[a])*scale+offset[a])
    };
    let positions = xyz.chunks_exact(3).flat_map(transform).collect();
    let mut display_cameras = cameras.to_vec();
    for c in display_cameras.chunks_exact_mut(18) {
        let position = transform(&c[9..12]);
        c[9..12].copy_from_slice(&position);
    }
    (positions,display_cameras, (std::array::from_fn(|a| offset[a]-center[a]*radius/extent), radius/extent))
}
