//! Reconstruction fills the viewport; inspection panels remain available on demand.
use rerun::blueprint::{Blueprint, BlueprintPanel, SelectionPanel, Spatial3DView, TimePanel};
use rerun::blueprint::components::{PanelState, PlayState};

pub fn blueprint() -> Blueprint {
    Blueprint::new(Spatial3DView::new("StereoForge — gray staging is schematic")
        .with_origin("world").with_contents(["world/**"]))
        .with_auto_views(false)
        .with_auto_layout(false)
        .with_blueprint_panel(BlueprintPanel::new().with_state(PanelState::Collapsed))
        .with_selection_panel(SelectionPanel::new().with_state(PanelState::Collapsed))
        .with_time_panel(TimePanel::new()
            .with_state(PanelState::Collapsed)
            .with_timeline("reconstruction_step")
            .with_play_state(PlayState::Following))
}
