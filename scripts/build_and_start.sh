#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jagennath Hari
#
# This file is part of SceneForge.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Resolve the input before changing directories: relative paths belong to the caller.
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash scripts/build_and_start.sh VIDEO_FILE"
    exit 0
fi
if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/build_and_start.sh VIDEO_FILE" >&2
    exit 2
fi
if [[ ! -f "$1" || ! -r "$1" ]]; then
    echo "Video must be a readable regular file: $1" >&2
    exit 1
fi
VIDEO_PATH="$(realpath -- "$1")"
if ! command -v ffprobe >/dev/null 2>&1; then
    echo "ffprobe is required on the host to validate videos. Install FFmpeg and retry." >&2
    exit 1
fi
# Inspect actual media, excluding attached cover art. Decode a short prefix to
# reject still images and files without at least two readable video frames.
if ! VIDEO_FRAMES="$(ffprobe -v error -select_streams V:0 -read_intervals '%+#64' \
    -count_frames -show_entries stream=nb_read_frames \
    -of default=noprint_wrappers=1:nokey=1 "$VIDEO_PATH")" ||
    [[ ! "$VIDEO_FRAMES" =~ ^[0-9]+$ ]] || (( VIDEO_FRAMES < 2 )); then
    echo "Not a readable video with at least two frames: $VIDEO_PATH" >&2
    exit 1
fi

# Configuration
ORG="sceneforge"
TAG="latest"
CUDA_IMAGE="nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04"

BASE_IMAGE="${ORG}/base:${TAG}"
GEOMETRY_IMAGE="${ORG}/geometry:${TAG}"
CUNLS_IMAGE="${ORG}/cunls:${TAG}"
RERUN_IMAGE="${ORG}/rerun:${TAG}"
RUN_CONTAINER="sceneforge"
RUN_IMAGE="${RERUN_IMAGE}"

USERNAME="$(id -un)"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Required runtime credential file; never print the token or pass it into builds.
HF_TOKEN_FILE="${REPO_ROOT}/.secrets/hf_token"
if [[ ! -f "${HF_TOKEN_FILE}" || ! -r "${HF_TOKEN_FILE}" || ! -s "${HF_TOKEN_FILE}" ]] ||
    ! LC_ALL=C grep -q '[^[:space:]]' "${HF_TOKEN_FILE}"; then
    cat >&2 <<EOF
Hugging Face token file is missing, unreadable, or empty:
  ${HF_TOKEN_FILE}

Request access to https://huggingface.co/facebook/VGGT-Omega and wait for approval.
Use a read token from that approved account with permission to download this repository.
Create/edit the file from the project directory:

  mkdir -p .secrets
  chmod 700 .secrets
  (umask 077; touch .secrets/hf_token)
  chmod 600 .secrets/hf_token
  nano .secrets/hf_token

Paste only the token into the file, save it, then rerun this command.
The file is ignored by Git and mounted read-only; it is not included in the image.
EOF
    exit 1
fi
chmod 700 "${REPO_ROOT}/.secrets"
chmod 600 "${HF_TOKEN_FILE}"
HF_SECRET_ARGS=(
    --mount "type=bind,source=${HF_TOKEN_FILE},target=/run/secrets/hf_token,readonly"
    --env HF_TOKEN_PATH=/run/secrets/hf_token
)

# Each invocation runs a reconstruction, rather than attaching to an old shell.
# Existing containers may be processing another video; do not interrupt them.
if docker container inspect "${RUN_CONTAINER}" >/dev/null 2>&1; then
    echo "Container '${RUN_CONTAINER}' already exists. Stop/remove it before starting a new run." >&2
    exit 1
fi

# The source is read-only; keyframe caches live on a separate writable mount.
CONTAINER_VIDEO=/input/video
mkdir -p .cache/huggingface .cache/sceneforge/video-keyframes data
chmod 700 .cache/huggingface

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

# Run directly, forwarding the reconstruction's exit status and terminal signals.
TERMINAL_ARGS=()
if [[ -t 0 && -t 1 ]]; then
    TERMINAL_ARGS=(-it)
fi
# Docker --mount uses CSV: quote/escape the source so spaces and commas are valid.
VIDEO_MOUNT_SOURCE="${VIDEO_PATH//\"/\"\"}"
echo "Reconstructing '${VIDEO_PATH}' (outputs: ${REPO_ROOT}/data/output)..."
exec docker run "${TERMINAL_ARGS[@]}" --rm \
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
    --workdir /workspace/SceneForge \
    --mount "type=bind,source=${REPO_ROOT}/sceneforge,target=/workspace/SceneForge/sceneforge" \
    --mount "type=bind,source=${REPO_ROOT}/configs,target=/workspace/SceneForge/configs,readonly" \
    --mount "type=bind,source=${REPO_ROOT}/data,target=/workspace/SceneForge/data" \
    --mount "type=bind,source=${REPO_ROOT}/.cache,target=/home/${USERNAME}/.cache" \
    --mount "type=bind,\"source=${VIDEO_MOUNT_SOURCE}\",target=${CONTAINER_VIDEO},readonly" \
    --mount "type=bind,source=${REPO_ROOT}/.cache/sceneforge/video-keyframes,target=/input/.keyframes" \
    "${RUN_IMAGE}" python -m sceneforge.reconstruction --video "${CONTAINER_VIDEO}"
