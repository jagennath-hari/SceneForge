//! One viewport following a separate presentation camera; never an SfM camera.
use rerun::{RecordingStream, RecordingStreamBuilder};
use rerun::external::re_log_types::BlueprintActivationCommand;
use rerun::external::re_sdk_types::blueprint::archetypes::{
    ContainerBlueprint, EyeControls3D, PanelBlueprint, TimePanelBlueprint,
    ViewBlueprint, ViewContents, ViewportBlueprint,
};
use rerun::blueprint::components::{ContainerKind, IncludedContent, PanelState, PlayState, RootContainer};
use crate::Result;

pub const EYE: &str = "world/presentation_eye";

pub fn install(recording: &RecordingStream) -> Result<()> {
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
        "+ world/**", "- world/presentation_eye/**",
    ]))?;
    blueprint.log(format!("{view}/EyeControls3D"), &EyeControls3D::new()
        .with_tracking_entity(EYE).with_look_target([0.0_f32,0.0,0.0])
        .with_eye_up([0.0_f32,-1.0,0.0]))?;
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
    let center = direction.map(|v| v*radius*3.2);
    let forward = direction.map(|v| -v);
    let right = [0.70710677_f32,0.0,0.70710677];
    let down = [forward[1]*right[2]-forward[2]*right[1],
                forward[2]*right[0]-forward[0]*right[2],
                forward[0]*right[1]-forward[1]*right[0]];
    recording.log(EYE, &rerun::Transform3D::from_translation_mat3x3(center,[right,down,forward]))?;
    recording.log(EYE, &rerun::Pinhole::from_focal_length_and_resolution(
        [900.0_f32,900.0],[1600.0_f32,1000.0])
        .with_camera_xyz(rerun::components::ViewCoordinates::RDF))?;
    Ok(())
}

/// Smoothed chase view with fixed distance/FOV and a stable up direction.
/// No per-frame bounding-box fitting, which can cause zoom pumping.
#[derive(Default)]
pub struct ChaseEye {
    state: Option<([f32;3], [f32;3], [f32;3], [f32;3])>,
}

fn unit(v: [f32;3]) -> Option<[f32;3]> {
    let norm = v.iter().map(|x| x*x).sum::<f32>().sqrt();
    if !norm.is_finite() || norm < 1e-6 { None } else { Some(v.map(|x| x/norm)) }
}
fn cross(a: [f32;3], b: [f32;3]) -> [f32;3] {
    [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
}

impl ChaseEye {
    pub fn follow(&mut self, rec: &RecordingStream, center: [f32;3], rotation: [f32;9]) -> Result<()> {
        let Some(desired_forward) = unit([rotation[2],rotation[5],rotation[8]]) else { return Ok(()); };
        let desired_up = [-rotation[1],-rotation[4],-rotation[7]];
        let (old_eye, old_forward, up, old_right) = self.state.unwrap_or((
            std::array::from_fn(|a| center[a]-1.2*desired_forward[a]+0.35*desired_up[a]),
            desired_forward, desired_up, [rotation[0],rotation[3],rotation[6]],
        ));
        let forward = unit(std::array::from_fn(|a| old_forward[a]*0.85+desired_forward[a]*0.15))
            .unwrap_or(old_forward);
        // Near the up-axis, project the previous right direction onto the new
        // view plane rather than constructing a non-orthogonal camera basis.
        let dot = (0..3).map(|a| old_right[a]*forward[a]).sum::<f32>();
        let fallback = unit(std::array::from_fn(|a| old_right[a]-dot*forward[a]));
        let Some(right) = unit(cross(forward,up)).or(fallback) else { return Ok(()); };
        let down = unit(cross(forward,right)).unwrap_or(up.map(|v| -v));
        let desired_eye: [f32;3] = std::array::from_fn(|a| center[a]-1.2*forward[a]+0.35*up[a]);
        let delta: [f32;3] = std::array::from_fn(|a| desired_eye[a]-old_eye[a]);
        let distance = delta.iter().map(|v| v*v).sum::<f32>().sqrt();
        let gain = if distance < 0.01 { 0.0 } else { 0.15_f32.min(0.25/distance) };
        let eye: [f32;3] = std::array::from_fn(|a| old_eye[a]+gain*delta[a]);
        rec.log(EYE, &rerun::Transform3D::from_translation_mat3x3(eye,[right,down,forward]))?;
        // Keep the same lens as the overview. Only the view pose follows the camera.
        self.state = Some((eye,forward,up,right));
        Ok(())
    }
}
