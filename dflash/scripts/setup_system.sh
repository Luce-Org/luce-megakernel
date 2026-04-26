#!/usr/bin/env bash
# Install system build dependencies for dflash.
# Ubuntu 22.04 (jammy) and 24.04 (noble). Run with sudo.
set -euo pipefail

# ── helpers ──────────────────────────────────────────────────────────────────

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
die()   { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

if [[ $EUID -ne 0 ]]; then
    die "This script must be run as root (try: sudo $0)"
fi

# ── distro check ─────────────────────────────────────────────────────────────

if [[ ! -f /etc/os-release ]]; then
    die "Cannot detect OS. This script supports Ubuntu 22.04 and 24.04 only."
fi
source /etc/os-release

if [[ "${ID:-}" != "ubuntu" ]]; then
    die "Unsupported distro '${ID:-unknown}'. This script supports Ubuntu only."
fi

case "${VERSION_CODENAME:-}" in
    jammy)  UBUNTU_VER="2204" ;;
    noble)  UBUNTU_VER="2404" ;;
    *)      die "Unsupported Ubuntu release '${VERSION_CODENAME:-unknown}'. Supported: 22.04 (jammy), 24.04 (noble)." ;;
esac

info "Detected Ubuntu ${VERSION_ID} (${VERSION_CODENAME})"

# ── apt build deps ────────────────────────────────────────────────────────────

info "Installing build-essential, cmake, git, git-lfs..."
apt-get update -qq
apt-get install -y build-essential cmake git git-lfs

git lfs install --system 2>/dev/null || git lfs install
ok "Build tools installed."

# ── huggingface-cli (pipx, installed for $SUDO_USER not root) ────────────────

REAL_USER="${SUDO_USER:-$USER}"

apt-get install -y pipx

if sudo -u "${REAL_USER}" pipx list 2>/dev/null | grep -q huggingface_hub; then
    ok "huggingface-cli already installed for ${REAL_USER}."
else
    info "Installing huggingface-cli for ${REAL_USER}..."
    sudo -u "${REAL_USER}" pipx install "huggingface_hub[cli]"
    ok "huggingface-cli installed."
fi

sudo -u "${REAL_USER}" pipx ensurepath --quiet 2>/dev/null || true

# ── CUDA Toolkit ─────────────────────────────────────────────────────────────

CUDA_NEWLY_INSTALLED=0

if command -v nvcc &>/dev/null; then
    ok "nvcc already on PATH ($(nvcc --version | grep -oP 'release \K[\d.]+')). Skipping CUDA install."
else
    KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
    KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VER}/x86_64/${KEYRING_DEB}"

    info "Downloading NVIDIA CUDA keyring from ${KEYRING_URL}..."
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    wget -q --show-progress -O "${TMP}/${KEYRING_DEB}" "${KEYRING_URL}"

    info "Installing CUDA keyring..."
    dpkg -i "${TMP}/${KEYRING_DEB}"
    apt-get update -qq

    info "Installing cuda-toolkit..."
    apt-get install -y cuda-toolkit
    ok "CUDA toolkit installed."
    CUDA_NEWLY_INSTALLED=1

    # PATH persistence
    BASH_PROFILE=/etc/profile.d/cuda.sh
    if [[ ! -f "${BASH_PROFILE}" ]]; then
        printf 'export PATH=/usr/local/cuda/bin:$PATH\n' > "${BASH_PROFILE}"
        ok "Bash PATH configured via ${BASH_PROFILE}."
    fi

    # Ubuntu's default zsh does not source /etc/profile.d, so write zshenv too.
    ZSH_ENV=/etc/zsh/zshenv
    if [[ -d /etc/zsh ]] && ! grep -q 'cuda' "${ZSH_ENV}" 2>/dev/null; then
        printf 'export PATH=/usr/local/cuda/bin:$PATH\n' >> "${ZSH_ENV}"
        ok "Zsh PATH configured via ${ZSH_ENV}."
    fi
fi

# ── summary ──────────────────────────────────────────────────────────────────

printf '\n'

if [[ ${CUDA_NEWLY_INSTALLED} -eq 1 ]]; then
    warn "nvcc is not yet on your current PATH. Activate it now with:"
    printf '    export PATH=/usr/local/cuda/bin:$PATH\n'
    printf '\n'
    warn "If you ran cmake before this script, wipe the build directory first to avoid stale compiler cache:"
    printf '    rm -rf build\n'
    printf '\n'
fi

info "All system dependencies installed. Next steps:"
printf '    git submodule update --init --recursive\n'
printf '    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release\n'
printf '    cmake --build build --target test_dflash -j\n'
