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

if [[ $# -eq 0 ]]; then
    echo "Usage: sceneforge-entrypoint COMMAND [ARGUMENTS...]" >&2
    exit 2
fi

# Runtime preparation sees the actual GPUs and the host-mounted persistent cache.
# The same config/cache is used by streaming keyframes and later pair matching.
python -m sceneforge.video.learned_models \
    --config /workspace/SceneForge/configs/keyframes_raco.json

exec "$@"
