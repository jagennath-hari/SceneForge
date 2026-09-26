//! Keep unrelated coordinate systems and unposed frontend images separate.
use rerun::blueprint::{Blueprint, ContainerLike, Grid, Spatial2DView, Spatial3DView, TextLogView, TimePanel};
use rerun::external::re_sdk_types::blueprint::components::PlayState;

pub fn blueprint() -> Blueprint {
    Blueprint::new(Grid::new(vec![
        ContainerLike::from(Spatial3DView::new("Common map").with_origin("world").with_contents(["world/**"])),
        ContainerLike::from(Spatial2DView::new("Images and measured features").with_origin("frontend").with_contents(["frontend/**"])),
        ContainerLike::from(Spatial3DView::new("VGGT window (local coordinates)").with_origin("local_window").with_contents(["local_window/**"])),
        ContainerLike::from(TextLogView::new("Pipeline").with_origin("/").with_contents(["pipeline/**"])),
    ])).with_auto_views(false)
      .with_time_panel(TimePanel::new().with_timeline("reconstruction_step").with_play_state(PlayState::Following))
}
