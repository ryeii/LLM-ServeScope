#!/bin/bash
# Native macOS CPU installation. Upstream sources remain pristine.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SIM_SOURCE="$ROOT/.vendor/LLMServingSim"
PYTHON_BIN="${MEMANALYSIS_PYTHON:-/opt/homebrew/bin/python3.10}"
if [[ ! -d "$SIM_SOURCE/.git" ]]; then
  mkdir -p .vendor
  git clone https://github.com/casys-kaist/LLMServingSim.git "$SIM_SOURCE"
fi
git -C "$SIM_SOURCE" checkout b09c7aaab4b072d73136641841cc2688616c49eb
git -C "$SIM_SOURCE" submodule update --init --recursive --depth 1
"$PYTHON_BIN" -m venv .venv-sim
.venv-sim/bin/python -m pip install -r integration/requirements-native.txt
.venv-sim/bin/python -m pip install --no-deps --no-build-isolation "$SIM_SOURCE/astra-sim/extern/graph_frontend/chakra"
mkdir -p .native/generated .native/logs
if [[ ! -d .native/protobuf-src/.git ]]; then
  git clone --depth 1 --branch v3.21.12 https://github.com/protocolbuffers/protobuf.git .native/protobuf-src
fi
git -C .native/protobuf-src checkout f0dc78d7e6e331b8c6bb2d5283e06aa26883ca7c
cmake -S .native/protobuf-src/cmake -B .native/protobuf-build \
  -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT/.native/protobuf"
cmake --build .native/protobuf-build -j 2
cmake --install .native/protobuf-build
.native/protobuf/bin/protoc \
  --proto_path="$SIM_SOURCE/astra-sim/extern/graph_frontend/chakra/schema/protobuf" \
  --cpp_out=.native/generated et_def.proto
cmake -S integration/native-cmake -B .native/astra-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$ROOT/.native/protobuf"
cmake --build .native/astra-build -j 2
.venv-sim/bin/python integration/native_runner.py --check
