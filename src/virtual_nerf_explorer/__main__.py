from __future__ import annotations

import argparse
from pathlib import Path

from virtual_nerf_explorer.app import run_app
from virtual_nerf_explorer.config import AppConfig, RenderConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a minimal Nerfstudio scene explorer.")
    parser.add_argument("--load-config", required=True, type=Path, help="Path to a Nerfstudio config.yml file.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for the viser server.")
    parser.add_argument("--port", type=int, default=8080, help="Port for the viser server.")
    parser.add_argument(
        "--show-training-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show sampled training camera frustums on startup.",
    )
    parser.add_argument(
        "--show-world-axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the origin frame and world grid on startup.",
    )
    parser.add_argument(
        "--max-display-cameras",
        type=int,
        default=24,
        help="Maximum number of training camera frustums to draw.",
    )
    parser.add_argument(
        "--min-orbit-distance",
        type=float,
        default=1.0,
        help="Minimum distance maintained between the viewer camera position and look-at target.",
    )
    parser.add_argument(
        "--static-max-res",
        type=int,
        default=960,
        help="Maximum render resolution used after camera motion settles.",
    )
    parser.add_argument(
        "--moving-max-res",
        type=int,
        default=320,
        help="Maximum render resolution used while the camera is moving.",
    )
    args = parser.parse_args()
    run_app(
        AppConfig(
            load_config=args.load_config,
            host=args.host,
            port=args.port,
            show_training_cameras=args.show_training_cameras,
            show_world_axes=args.show_world_axes,
            max_display_cameras=args.max_display_cameras,
            min_orbit_distance=args.min_orbit_distance,
            render=RenderConfig(
                static_max_res=args.static_max_res,
                moving_max_res=args.moving_max_res,
            ),
        )
    )


if __name__ == "__main__":
    main()
