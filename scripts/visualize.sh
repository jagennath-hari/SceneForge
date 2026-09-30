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

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash scripts/visualize.sh PATH_TO_PIPELINE.rrd"
    exit 0
fi
if [[ $# -ne 1 || ! -f "$1" || ! -r "$1" || "$1" != *.rrd ]]; then
    echo "Provide a readable .rrd recording: bash scripts/visualize.sh PATH_TO_PIPELINE.rrd" >&2
    exit 2
fi
if [[ -z "${DISPLAY:-}" ]]; then
    echo "A graphical desktop with DISPLAY set is required to open the viewer." >&2
    exit 1
fi
RUN_IMAGE="sceneforge/rerun:latest"
if ! docker image inspect "${RUN_IMAGE}" >/dev/null 2>&1; then
    echo "SceneForge Docker image is missing. Build it with scripts/build_and_start.sh first." >&2
    exit 1
fi
RECORDING_PATH="$(realpath -- "$1")"
# Docker --mount parses CSV; preserve spaces and commas in recording paths.
RECORDING_SOURCE="${RECORDING_PATH//\"/\"\"}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
AUTH_ARGS=()
AUTH_FILE="${XAUTHORITY:-${HOME}/.Xauthority}"
if [[ -f "${AUTH_FILE}" ]]; then
    AUTH_SOURCE="${AUTH_FILE//\"/\"\"}"
    AUTH_ARGS=(--mount "type=bind,\"source=${AUTH_SOURCE}\",target=/tmp/viewer.Xauthority,readonly"
               --env XAUTHORITY=/tmp/viewer.Xauthority)
fi

# Run only the installed viewer: no model setup, credentials or reconstruction.
exec docker run --rm --init \
    --runtime=nvidia --gpus all \
    --net=host --ipc=host \
    --user "${HOST_UID}:${HOST_GID}" \
    --entrypoint /usr/local/bin/rerun \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY="${DISPLAY}" \
    -e HOME=/tmp \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    "${AUTH_ARGS[@]}" \
    --mount "type=bind,\"source=${RECORDING_SOURCE}\",target=/recording/pipeline.rrd,readonly" \
    "${RUN_IMAGE}" --port auto /recording/pipeline.rrd
