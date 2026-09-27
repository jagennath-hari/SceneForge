#!/bin/bash
set -euo pipefail

# Configuration
ORG="stereoforge"
TAG="latest"
CUDA_IMAGE="nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04"

BASE_IMAGE="${ORG}/base:${TAG}"
GEOMETRY_IMAGE="${ORG}/geometry:${TAG}"
CUNLS_IMAGE="${ORG}/cunls:${TAG}"
RERUN_IMAGE="${ORG}/rerun:${TAG}"
RUN_CONTAINER="stereoforge"
RUN_IMAGE="${RERUN_IMAGE}"

USERNAME="$(id -un)"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# HF login stores credentials here through the cache bind mount.
# Restrict directory access before either attaching or starting a container.
mkdir -p .cache/huggingface
chmod 700 .cache/huggingface

# Attach to the existing environment without rebuilding.
if docker ps --format '{{.Names}}' | grep -Fxq "${RUN_CONTAINER}"; then
    echo "Container '${RUN_CONTAINER}' already running. Attaching..."
    exec docker exec -it "${RUN_CONTAINER}" bash
fi

# Build order: NVIDIA CUDA → base → geometry → cuNLS → Rust/Rerun.
declare -A DOCKERFILES=(
    ["${BASE_IMAGE}"]="docker/Dockerfile.base"
    ["${GEOMETRY_IMAGE}"]="docker/Dockerfile.geometry"
    ["${CUNLS_IMAGE}"]="docker/Dockerfile.cunls"
    ["${RERUN_IMAGE}"]="docker/Dockerfile.rerun"
)
declare -A PARENTS=(
    ["${BASE_IMAGE}"]="${CUDA_IMAGE}"
    ["${GEOMETRY_IMAGE}"]="${BASE_IMAGE}"
    ["${CUNLS_IMAGE}"]="${GEOMETRY_IMAGE}"
    ["${RERUN_IMAGE}"]="${CUNLS_IMAGE}"
)
BUILD_SEQUENCE=("${BASE_IMAGE}" "${GEOMETRY_IMAGE}" "${CUNLS_IMAGE}" "${RERUN_IMAGE}")

# Optional runtime credential file; never pass its contents to Docker or builds.
HF_SECRET_ARGS=()
if [[ -f "${REPO_ROOT}/.secrets/hf_token" ]]; then
    if [[ ! -s "${REPO_ROOT}/.secrets/hf_token" ]]; then
        echo "The HF token file is empty: .secrets/hf_token" >&2
        exit 1
    fi
    chmod 700 "${REPO_ROOT}/.secrets"
    chmod 600 "${REPO_ROOT}/.secrets/hf_token"
    HF_SECRET_ARGS=(
        --mount "type=bind,source=${REPO_ROOT}/.secrets/hf_token,target=/run/secrets/hf_token,readonly"
        --env HF_TOKEN_PATH=/run/secrets/hf_token
    )
fi

for image in "${BUILD_SEQUENCE[@]}"; do
    echo "Building '${image}' using parent '${PARENTS[$image]}'..."
    DOCKER_BUILDKIT=1 docker build \
        --build-arg BASE_FROM="${PARENTS[$image]}" \
        --build-arg USERNAME="${USERNAME}" \
        --build-arg USER_UID="${HOST_UID}" \
        --build-arg USER_GID="${HOST_GID}" \
        --tag "${image}" \
        --file "${DOCKERFILES[$image]}" .
done

# Start an interactive environment with only the essential host directories.
mkdir -p .cache data
exec docker run -it --rm \
    --name "${RUN_CONTAINER}" \
    --init \
    "${HF_SECRET_ARGS[@]}" \
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
    -e XAUTHORITY="/home/${USERNAME}/.Xauthority" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e XDG_RUNTIME_DIR=/tmp/runtime-root \
    -v /tmp/runtime-root:/tmp/runtime-root \
    -v "${HOME}/.Xauthority:/root/.Xauthority" \
    -v "${HOME}/.Xauthority:/home/${USERNAME}/.Xauthority:ro" \
    --user "${HOST_UID}:${HOST_GID}" \
    --workdir /workspace/StereoForge \
    --mount "type=bind,source=${REPO_ROOT}/stereoforge,target=/workspace/StereoForge/stereoforge" \
    --mount "type=bind,source=${REPO_ROOT}/configs,target=/workspace/StereoForge/configs,readonly" \
    --mount "type=bind,source=${REPO_ROOT}/data,target=/workspace/StereoForge/data" \
    --mount "type=bind,source=${REPO_ROOT}/.cache,target=/home/${USERNAME}/.cache" \
    "${RUN_IMAGE}" /bin/bash
