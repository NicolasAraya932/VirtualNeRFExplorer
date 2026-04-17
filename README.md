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
- depth and accumulation render modes
- lower resolution during motion, higher resolution after motion settles
- origin frame and world grid
- optional sampled training camera frustums
- click a displayed training camera to snap to that pose
- reset/focus/next-camera navigation controls
- saved camera viewpoints
- image capture of the current rendered frame
- minimal GUI for visibility toggles and render resolution

Not implemented:
- ROI selection
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
- `--min-orbit-distance`
  - enforce a minimum camera-to-look-at distance to avoid the orbit controls becoming sluggish near the scene center
- `--static-max-res`
  - maximum resolution after the camera stops moving
- `--moving-max-res`
  - maximum resolution while the camera is moving
- `--depth-quantile`
  - cumulative weight threshold used by the depth view; `0.5` matches Nerfstudio median depth

## Notes

- Scene capture is available from the GUI via `Capture`.
- Capture downloads the latest rendered NeRF frame already shown in the explorer instead of triggering a separate browser-side scene render.
- The `Name` field lets you choose the download base name safely; invalid filename characters are sanitized automatically.
- `Tensors -> Export` downloads the current render package as `.npz` or `.pt`, including image-space ray tensors plus model outputs such as RGB, depth, accumulation, density, and related arrays when available.
- `Render` switches between `rgb`, `depth`, and `accumulation` outputs using the current model outputs.
- `Depth cumsum` adjusts the cumulative-weight split used by the depth view. `0.5` is the standard median depth; lower values bias toward nearer samples and higher values bias deeper into the volume.
- `Reset`, `Focus`, `Next cam`, and `Snap` provide basic scene-navigation controls without patching `viser`.
- Saved views are stored in memory for the current explorer session and can be loaded or deleted from the `Views` folder.
- The current installed `viser` API does not expose a backend keyboard shortcut hook, so capture is implemented as a button rather than a true key binding.
- Rendering is serialized through a lock to avoid concurrent model access from multiple clients.
- Relative paths in moved Nerfstudio configs are repaired automatically using the config location and inferred run root.
- The package sets `TORCHINDUCTOR_COMPILE_THREADS=1` by default before Nerfstudio imports so the viewer does not spawn a large compile-worker process pool and `Ctrl+C` remains usable.
- This project is intentionally simpler than the full Nerfstudio viewer. The current product is a scene explorer, not a training UI.
