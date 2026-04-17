# VirtualNeRFExplorer

`VirtualNeRFExplorer` is a minimal `viser`-based scene explorer for Nerfstudio-compatible models.

This repository is intentionally narrow in scope. The current goal is only to:
- load a Nerfstudio `config.yml`
- restore the trained pipeline checkpoint
- open a live `viser` scene explorer
- move around the rendered scene interactively
- optionally show a reduced set of training camera frustums

It is not yet an ROI tool, selection tool, or export workspace.

## v1 Scope

Implemented:
- load any model that Nerfstudio can restore from `config.yml`
- correct path repair for moved configs by rebasing relative paths during load
- live RGB rerender while moving
- lower resolution during motion, higher resolution after motion settles
- origin frame and world grid
- optional sampled training camera frustums
- click a displayed training camera to snap to that pose
- minimal GUI for visibility toggles and render resolution

Not implemented:
- ROI selection
- depth or accumulation visualization
- export panels
- training controls
- project registry integration
- multi-model comparison

## Repository Layout

```text
VirtualNeRFExplorer/
├── pyproject.toml
├── README.md
└── src/
    └── virtual_nerf_explorer/
        ├── __init__.py
        ├── __main__.py
        ├── app.py
        ├── camera.py
        ├── config.py
        ├── loader.py
        ├── render.py
        ├── session.py
        ├── state.py
        └── viewer/
            ├── __init__.py
            ├── explorer.py
            ├── gui.py
            └── scene.py
```

## Architecture

The implementation follows Nerfstudio viewer patterns, but strips them down to only what is required for interactive scene exploration.

- `loader.py`
  - loads the Nerfstudio config and pipeline through `eval_setup(...)`
  - repairs relative `data` and `output_dir` paths for moved configs
- `camera.py`
  - converts between `viser` client camera state and Nerfstudio render camera state
- `render.py`
  - owns a per-client render worker thread
  - renders low resolution during motion and high resolution after motion stops
- `viewer/scene.py`
  - builds the world grid, origin frame, and training camera frustums
- `viewer/gui.py`
  - builds the minimal control panel and status block
- `viewer/explorer.py`
  - wires client lifecycle, render workers, camera snapping, and GUI callbacks

## Install

```bash
cd /workspace/Desktop/Repos/VirtualNeRFExplorer
pip install -e .
```

## Run

```bash
cd /workspace/Desktop/Repos/VirtualNeRFExplorer
virtual-nerf-explorer \
  --load-config /path/to/config.yml \
  --host 0.0.0.0 \
  --port 8080
```

Example with your current dataset:

```bash
cd /workspace/Desktop/Repos/VirtualNeRFExplorer
virtual-nerf-explorer \
  --load-config /workspace/Desktop/DATASETS/CHERRY_DATASET_2_UOH_O/CHERRY_DATASET_2_UOH/nerfacto/2026-04-16_162613/config.yml \
  --host 0.0.0.0 \
  --port 8080
```

## Main Flags

- `--load-config`
  - required Nerfstudio config file
- `--show-training-cameras` / `--no-show-training-cameras`
  - show or hide sampled training frustums on startup
- `--show-world-axes` / `--no-show-world-axes`
  - show or hide origin frame and grid on startup
- `--max-display-cameras`
  - cap the number of frustums drawn in the viewer
- `--static-max-res`
  - maximum resolution after the camera stops moving
- `--moving-max-res`
  - maximum resolution while the camera is moving

## Notes

- The explorer uses RGB only in v1.
- Rendering is serialized through a lock to avoid concurrent model access from multiple clients.
- Relative paths in moved Nerfstudio configs are repaired automatically using the config location and inferred run root.
- This project is intentionally simpler than the full Nerfstudio viewer. The current product is a scene explorer, not a training UI.
