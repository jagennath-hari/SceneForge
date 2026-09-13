#!/bin/bash
set -euo pipefail

# Configuration
ORG="stereoforge"
TAG="latest"
CUDA_IMAGE="nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04"

BASE_IMAGE="${ORG}/base:${TAG}"
GEOMETRY_IMAGE="${ORG}/geometry:${TAG}"
STEREO_IMAGE="${ORG}/stereo:${TAG}"
PYCUSFM_IMAGE="${ORG}/pycusfm:${TAG}"
RUN_CONTAINER="stereoforge"
RUN_IMAGE="${PYCUSFM_IMAGE}"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Attach to the existing environment without rebuilding.
if docker ps --format '{{.Names}}' | grep -Fxq "${RUN_CONTAINER}"; then
    echo "Container '${RUN_CONTAINER}' already running. Attaching..."
    exec docker exec -it "${RUN_CONTAINER}" bash
fi

# Build order: NVIDIA CUDA → base → geometry → stereo → pyCuSFM.
declare -A DOCKERFILES=(
    ["${BASE_IMAGE}"]="docker/Dockerfile.base"
    ["${GEOMETRY_IMAGE}"]="docker/Dockerfile.geometry"
    ["${STEREO_IMAGE}"]="docker/Dockerfile.stereo"
    ["${PYCUSFM_IMAGE}"]="docker/Dockerfile.pycusfm"
)
declare -A PARENTS=(
    ["${BASE_IMAGE}"]="${CUDA_IMAGE}"
    ["${GEOMETRY_IMAGE}"]="${BASE_IMAGE}"
    ["${STEREO_IMAGE}"]="${GEOMETRY_IMAGE}"
    ["${PYCUSFM_IMAGE}"]="${STEREO_IMAGE}"
)
BUILD_SEQUENCE=("${BASE_IMAGE}" "${GEOMETRY_IMAGE}" "${STEREO_IMAGE}" "${PYCUSFM_IMAGE}")

for image in "${BUILD_SEQUENCE[@]}"; do
    echo "Building '${image}' using parent '${PARENTS[$image]}'..."
    DOCKER_BUILDKIT=1 docker build \
        --build-arg BASE_FROM="${PARENTS[$image]}" \
        --build-arg USER_UID="${HOST_UID}" \
        --build-arg USER_GID="${HOST_GID}" \
        --tag "${image}" \
        --file "${DOCKERFILES[$image]}" .
done

# Start an interactive environment with only the essential host directories.
mkdir -p .cache data weights
exec docker run -it --rm \
    --name "${RUN_CONTAINER}" \
    --init \
    --runtime=nvidia \
    --gpus all \
    --privileged \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --net=host \
    --pid=host \
    --ipc=host \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY="${DISPLAY:-}" \
    -e XAUTHORITY=/home/stereoforge/.Xauthority \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e XDG_RUNTIME_DIR=/tmp/runtime-root \
    -v /tmp/runtime-root:/tmp/runtime-root \
    -v "${HOME}/.Xauthority:/root/.Xauthority" \
    -v "${HOME}/.Xauthority:/home/stereoforge/.Xauthority:ro" \
    --user "${HOST_UID}:${HOST_GID}" \
    --workdir /workspace/StereoForge \
    --mount "type=bind,source=${REPO_ROOT}/stereoforge,target=/workspace/StereoForge/stereoforge" \
    --mount "type=bind,source=${REPO_ROOT}/configs,target=/workspace/StereoForge/configs,readonly" \
    --mount "type=bind,source=${REPO_ROOT}/data,target=/workspace/StereoForge/data" \
    --mount "type=bind,source=${REPO_ROOT}/weights,target=/workspace/StereoForge/weights,readonly" \
    --mount "type=bind,source=${REPO_ROOT}/.cache,target=/home/stereoforge/.cache" \
    "${RUN_IMAGE}" /bin/bash
