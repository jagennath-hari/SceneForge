# StereoForge

VGGT-Ω geometry → adaptive stereo baseline → StereoSpace right-eye synthesis →
side-by-side video. pyCuSFM provides optional geometric refinement.

Run `bash scripts/build_and_start.sh` to build and enter the environment.

All Python dependencies share `/opt/stereoforge-venv`, created by uv using
Ubuntu's Python and already on `PATH`. uv 0.12.13 installs the Python packages;
the legacy TensorRT components retain their working pip installation in the same
environment. Both installers use `docker/constraints.txt`. BuildKit caches uv
and pip downloads, and installed files are independent of those caches.

Add dependency changes to the Dockerfiles and rebuild. This uses `uv pip install`,
not `uv sync`; a full dependency lockfile has not been generated.
