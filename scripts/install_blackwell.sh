#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

fail() {
    echo "[blackwell] ERROR: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

[[ $(uname -s) == "Linux" ]] || fail "This installer only supports Linux."
[[ $(uname -m) == "x86_64" ]] || fail "This installer only supports Linux x86_64."

if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    [[ ${ID:-} == "ubuntu" && ${VERSION_ID:-} == "22.04" ]] || \
        fail "Ubuntu 22.04 is required (found ${PRETTY_NAME:-unknown})."
else
    fail "Cannot identify the operating system: /etc/os-release is missing."
fi

for command_name in uv git nvidia-smi nvcc; do
    require_command "$command_name"
done

[[ -x "$PYTHON_BIN" ]] || fail "Run 'uv sync --frozen' before this installer."

driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')
driver_major=${driver_version%%.*}
[[ $driver_major =~ ^[0-9]+$ ]] || fail "Could not parse NVIDIA driver version: $driver_version"
(( driver_major >= 570 )) || fail "NVIDIA driver 570+ is required (found $driver_version)."

mapfile -t compute_caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '[:space:]' | sort -u)
[[ ${#compute_caps[@]} -gt 0 ]] || fail "No NVIDIA GPU was detected."
for compute_cap in "${compute_caps[@]}"; do
    [[ $compute_cap == "12.0" ]] || fail "Blackwell compute capability 12.0 is required (found $compute_cap)."
done

nvcc_release=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)
[[ $nvcc_release == "12.8" ]] || fail "CUDA Toolkit 12.8 with nvcc is required (found ${nvcc_release:-unknown})."

nvcc_path=$(readlink -f "$(command -v nvcc)")
CUDA_HOME=$(cd "$(dirname "$nvcc_path")/.." && pwd -P)
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
export FLASH_ATTN_CUDA_ARCHS="120"
export FLASH_ATTENTION_FORCE_BUILD="TRUE"
export MAX_JOBS=${MAX_JOBS:-4}
export NVCC_THREADS=${NVCC_THREADS:-4}

"$PYTHON_BIN" - <<'PY'
import sys
import torch

errors = []
if sys.version_info[:2] != (3, 10):
    errors.append(f"Python 3.10 is required, found {sys.version.split()[0]}")
if torch.__version__.split("+")[0] != "2.7.1":
    errors.append(f"PyTorch 2.7.1 is required, found {torch.__version__}")
if torch.version.cuda != "12.8":
    errors.append(f"PyTorch must use CUDA 12.8, found {torch.version.cuda}")
if not torch.cuda.is_available():
    errors.append("torch.cuda.is_available() is false")
if errors:
    raise SystemExit("\n".join(f"[blackwell] ERROR: {error}" for error in errors))
PY

BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/reconviagen-blackwell.XXXXXX")
cleanup() {
    if [[ -n ${BUILD_ROOT:-} && -d $BUILD_ROOT && $BUILD_ROOT == *reconviagen-blackwell.* ]]; then
        rm -rf "$BUILD_ROOT"
    fi
}
trap cleanup EXIT

clone_revision() {
    local name=$1
    local url=$2
    local revision=$3
    local recursive=${4:-false}
    local destination="$BUILD_ROOT/$name"

    echo "[blackwell] Fetching $name at $revision" >&2
    git clone --quiet --no-checkout "$url" "$destination"
    git -C "$destination" checkout --quiet --detach "$revision"
    if [[ $recursive == "true" ]]; then
        git -C "$destination" submodule update --init --recursive --quiet
    fi
    printf '%s\n' "$destination"
}

install_source() {
    local source_path=$1
    echo "[blackwell] Building $(basename "$source_path") for sm_120"
    uv pip install \
        --python "$PYTHON_BIN" \
        --no-build-isolation \
        --no-deps \
        --reinstall \
        "$source_path"
}

NVDIFFRAST_DIR=$(clone_revision \
    nvdiffrast https://github.com/NVlabs/nvdiffrast.git \
    253ac4fcea7de5f396371124af597e6cc957bfae)
NVDIFFREC_DIR=$(clone_revision \
    nvdiffrec https://github.com/JeffreyXiang/nvdiffrec.git \
    b296927cc7fd01c2ac1087c8065c4d7248f72da4)
CUMESH_DIR=$(clone_revision \
    CuMesh https://github.com/JeffreyXiang/CuMesh.git \
    12289e1062f0603f2f0d0771b02e1395d247f26f true)
FLEXGEMM_DIR=$(clone_revision \
    FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git \
    6dd94a859c26ee8246888502eada3dd8ad85532e true)
FLASH_ATTN_DIR=$(clone_revision \
    flash-attention https://github.com/Dao-AILab/flash-attention.git \
    060c9188beec3a8b62b33a3bfa6d5d2d44975fab true)

install_source "$NVDIFFRAST_DIR"
install_source "$NVDIFFREC_DIR"
install_source "$CUMESH_DIR"
install_source "$FLEXGEMM_DIR"
install_source "$FLASH_ATTN_DIR"
install_source "$ROOT_DIR/wheels/TRELLIS.2/o-voxel"

echo "[blackwell] CUDA extensions installed. Running smoke checks..."
"$PYTHON_BIN" "$ROOT_DIR/scripts/check_blackwell.py"
