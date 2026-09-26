//! One viewport following a separate presentation camera; never an SfM camera.
use rerun::{RecordingStream, RecordingStreamBuilder};
use rerun::external::re_log_types::BlueprintActivationCommand;
use rerun::external::re_sdk_types::blueprint::archetypes::{
    ContainerBlueprint, EyeControls3D, PanelBlueprint, TimePanelBlueprint,
    ViewBlueprint, ViewContents, ViewportBlueprint,
};
use rerun::blueprint::components::{ContainerKind, Eye3DKind, IncludedContent, PanelState, PlayState, RootContainer};
use crate::Result;

pub const EYE: &str = "world/presentation_eye";
pub const EYE_PINHOLE: &str = "world/presentation_eye/image";
const OVERVIEW_DISTANCE: f32 = 22.0;
pub const OVERVIEW_POSITION: [f32;3] = [
    0.57735026*OVERVIEW_DISTANCE,
    -0.57735026*OVERVIEW_DISTANCE,
    -0.57735026*OVERVIEW_DISTANCE,
];

pub fn install(recording: &RecordingStream, position: [f32;3], target: [f32;3], up: [f32;3]) -> Result<()> {
    // Build the official blueprint archetypes directly: the pinned Rust SDK's
    // high-level Spatial3DView builder does not expose EyeControls3D setters.
    let (blueprint, storage) = RecordingStreamBuilder::new("StereoForge").blueprint().memory()?;
    blueprint.set_time_sequence("blueprint", 0);
    let view_id = uuid::Uuid::new_v4();
    let container_id = uuid::Uuid::new_v4();
    let view = format!("view/{view_id}");
    blueprint.log(view.as_str(), &ViewBlueprint::new("3D")
        .with_display_name("StereoForge — schematic staging / reconstructed map")
        .with_space_origin("world"))?;
    blueprint.log(format!("{view}/ViewContents"), &ViewContents::new([
        "+ world/**",
    ]))?;
    blueprint.log(format!("{view}/EyeControls3D"), &EyeControls3D::new()
        .with_kind(Eye3DKind::FirstPerson).with_tracking_entity(EYE_PINHOLE)
        .with_position(position).with_look_target(target).with_eye_up(up))?;
    blueprint.log(format!("container/{container_id}"), &ContainerBlueprint::new(ContainerKind::Tabs)
        .with_contents([IncludedContent(view.into())]))?;
    blueprint.log("viewport", &ViewportBlueprint::new()
        .with_root_container(RootContainer(container_id.into()))
        .with_auto_layout(false).with_auto_views(false))?;
    for panel in ["blueprint_panel", "selection_panel"] {
        blueprint.log(panel, &PanelBlueprint::new().with_state(PanelState::Collapsed))?;
    }
    blueprint.log("time_panel", &TimePanelBlueprint::new().with_state(PanelState::Collapsed)
        .with_timeline("reconstruction_step").with_play_state(PlayState::Following))?;
    let id = blueprint.store_info().ok_or("Missing blueprint store")?.store_id.clone();
    recording.send_blueprint(storage.take(), BlueprintActivationCommand {
        blueprint_id: id, make_active: true, make_default: true,
    });
    Ok(())
}

/// A fixed oblique direction avoids spinning/jumping between unrelated windows.
/// Framing changes only with geometry stages, never with individual text events.
pub fn frame(recording: &RecordingStream, radius: f32) -> Result<()> {
    let direction = [0.57735026_f32,-0.57735026,-0.57735026];
    let center = direction.map(|v| v*radius*(OVERVIEW_DISTANCE/10.0));
    let forward = direction.map(|v| -v);
    let right = [0.70710677_f32,0.0,0.70710677];
    let down = [forward[1]*right[2]-forward[2]*right[1],
                forward[2]*right[0]-forward[0]*right[2],
                forward[0]*right[1]-forward[1]*right[0]];
    recording.log(EYE, &rerun::Transform3D::from_translation_mat3x3(center,[right,down,forward]))?;
    recording.log(EYE_PINHOLE, &rerun::Pinhole::from_focal_length_and_resolution(
        [900.0_f32,900.0],[1600.0_f32,1000.0])
        .with_camera_xyz(rerun::components::ViewCoordinates::RDF)
        .with_image_plane_distance(0.0001))?;
    Ok(())
}

/// Rigid camera-relative view: exactly behind the current processing camera.
/// Use its complete orientation, avoiding fixed-up singularities and lag when
/// refinement restarts at the first frame after calibration finishes.
pub fn follow(rec: &RecordingStream, center: [f32;3], rotation: [f32;9]) -> Result<()> {
    if !center.iter().chain(rotation.iter()).all(|v| v.is_finite()) {
        return Err("Nonfinite presentation camera pose".into());
    }
    // Incoming R_c2w is row-major; Rerun takes its basis vectors as columns.
    let right = [rotation[0],rotation[3],rotation[6]];
    let down = [rotation[1],rotation[4],rotation[7]];
    let forward = [rotation[2],rotation[5],rotation[8]];
    let eye: [f32;3] = std::array::from_fn(|a| center[a]-1.2*forward[a]);
    rec.log(EYE, &rerun::Transform3D::from_translation_mat3x3(eye,[right,down,forward]))?;
    // Lens and distance stay fixed throughout calibration and fusion.
    Ok(())
}

/// One progress-driven revolution at the overview's fixed distance and elevation.
pub fn orbit(rec: &RecordingStream, fraction: f32) -> Result<()> {
    if !fraction.is_finite() { return Ok(()); }
    let angle = -std::f32::consts::FRAC_PI_4 + fraction.clamp(0.0,1.0)*std::f32::consts::TAU;
    let horizontal = (2.0_f32/3.0).sqrt();
    let direction = [horizontal*angle.cos(), -1.0/3.0_f32.sqrt(), horizontal*angle.sin()];
    let center = direction.map(|v| v*OVERVIEW_DISTANCE);
    let forward = direction.map(|v| -v);
    let right = [-angle.sin(),0.0,angle.cos()];
    let down = [forward[1]*right[2]-forward[2]*right[1],
                forward[2]*right[0]-forward[0]*right[2],
                forward[0]*right[1]-forward[1]*right[0]];
    rec.log(EYE, &rerun::Transform3D::from_translation_mat3x3(center,[right,down,forward]))?;
    Ok(())
}

/// Geometry-stage close-up. Rotation is deliberately modest; distance is based
/// on the scene being inspected, not the unposed staging sphere.
pub fn focus(rec: &RecordingStream, target: [f32;3], radius: f32, progress: f32) -> Result<()> {
    let angle = -std::f32::consts::FRAC_PI_4 + progress.clamp(0.0,1.0)*0.6;
    let horizontal = (2.0_f32/3.0).sqrt();
    let direction = [horizontal*angle.cos(),-1.0/3.0_f32.sqrt(),horizontal*angle.sin()];
    let eye: [f32;3] = std::array::from_fn(|a| target[a]+direction[a]*radius.max(0.7)*2.8);
    let forward = direction.map(|v| -v);
    let right = [-angle.sin(),0.0,angle.cos()];
    let down = [forward[1]*right[2]-forward[2]*right[1],
                forward[2]*right[0]-forward[0]*right[2],
                forward[0]*right[1]-forward[1]*right[0]];
    rec.log(EYE, &rerun::Transform3D::from_translation_mat3x3(eye,[right,down,forward]))?;
    Ok(())
}
