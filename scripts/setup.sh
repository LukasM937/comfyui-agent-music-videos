#!/usr/bin/env bash
# One-stop setup for the music-video pipeline on Arch Linux (1x RTX 5090, 128 GB RAM).
#
#   scripts/setup.sh check               read-only preflight, changes nothing
#   scripts/setup.sh                     = all: system -> python -> comfyui -> llama -> models core -> services -> test
#   scripts/setup.sh system|python|comfyui|llama|models [group...]|services|test
#
# Options:  --dry-run  print what would run      --yes  no confirmation prompts
#           --comfy-dir DIR (~/ComfyUI)
#           --comfy-ref REF|latest               --install-driver  (also installs the NVIDIA driver)
#           --torch-index URL                    --skip-models  --skip-test
#
# Idempotent: every step first checks what is already there and only installs what is missing.
# Only pacman calls use sudo. The kernel driver is never touched unless --install-driver is given.

# shellcheck disable=SC2015  # "cond && ok || warn" is intentional: ok never fails
set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/src/llama.cpp}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
# ComfyUI commit all bundled workflows were validated against (reports version 0.39.0)
VALIDATED_COMFY_REF="e8665ca8e4089ab917567cdb491308e1397663d4"
COMFY_REF="${COMFY_REF:-$VALIDATED_COMFY_REF}"
LLAMA_REF="${LLAMA_REF:-master}"
TORCH_INDEX="${TORCH_INDEX:-}"
PY_VERSION="${PY_VERSION:-3.12}"
DRY=0 YES=0 INSTALL_DRIVER=0 SKIP_MODELS=0 SKIP_TEST=0

# ----------------------------------------------------------------------------- helpers
if [[ -t 1 ]]; then C_OK=$'\e[32m'; C_WARN=$'\e[33m'; C_ERR=$'\e[31m'; C_DIM=$'\e[2m'; C_OFF=$'\e[0m'
else C_OK=""; C_WARN=""; C_ERR=""; C_DIM=""; C_OFF=""; fi
FAILS=0 WARNS=0
say()  { printf '%s\n' "$*"; }
step() { printf '\n%s== %s ==%s\n' "$C_DIM" "$*" "$C_OFF"; }
ok()   { printf '  %sok%s    %s\n' "$C_OK" "$C_OFF" "$*"; }
warn() { printf '  %swarn%s  %s\n' "$C_WARN" "$C_OFF" "$*"; WARNS=$((WARNS + 1)); }
bad()  { printf '  %sFAIL%s  %s\n' "$C_ERR" "$C_OFF" "$*"; FAILS=$((FAILS + 1)); }
die()  { printf '%serror:%s %s\n' "$C_ERR" "$C_OFF" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

run() { # run CMD... : echo, and execute unless --dry-run
  printf '  %s$ %s%s\n' "$C_DIM" "$*" "$C_OFF"
  [[ $DRY -eq 1 ]] || "$@"
}
confirm() {
  [[ $YES -eq 1 || $DRY -eq 1 ]] && return 0
  read -r -p "  $1 [y/N] " a; [[ "$a" =~ ^[Yy]$ ]]
}
version_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]; } # $1 >= $2
is_arch() { [[ -r /etc/os-release ]] && grep -q '^ID=arch' /etc/os-release; }

# ----------------------------------------------------------------------------- checks
gpu_info() { # prints: name|mem_mib|driver|cc   (first GPU)
  have nvidia-smi || return 1
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits \
    2>/dev/null | head -n1 | awk -F', *' '{print $1"|"$2"|"$3"|"$4}'
}

check_all() {
  step "preflight (read-only)"
  if is_arch; then ok "Arch Linux, kernel $(uname -r)"
  else warn "not Arch ($(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")): package installation steps will be skipped"; fi

  local info name mem drv cc
  if info="$(gpu_info)" && [[ -n "$info" ]]; then
    IFS='|' read -r name mem drv cc <<<"$info"
    ok "GPU: $name, $((mem / 1024)) GB, driver $drv, sm_${cc/./}"
    (( mem >= 30000 )) || warn "less than 32 GB VRAM: the planner LLM and video models will offload heavily"
    if [[ "$cc" == 12* ]] && ! version_ge "$drv" 570; then bad "driver $drv too old for Blackwell (need >= 570)"; fi
    version_ge "$drv" 580 && ok "driver supports CUDA 13 wheels" || warn "driver < 580: will use cu128 PyTorch wheels"
  else
    bad "no NVIDIA GPU/driver found (install: sudo pacman -S nvidia-open-dkms nvidia-utils linux-headers, or run with --install-driver)"
  fi

  local ram_gb; ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
  if (( ram_gb >= 96 )); then ok "RAM: ${ram_gb} GB"
  elif (( ram_gb >= 60 )); then warn "RAM: ${ram_gb} GB (works, one big model at a time)"
  else bad "RAM: ${ram_gb} GB (need >= 64 GB; weights are streamed from RAM)"; fi

  local probe="$HOME"; [[ -d "$COMFY_DIR" ]] && probe="$COMFY_DIR"
  local free_gb; free_gb=$(df --output=avail -BG "$probe" | tail -n1 | tr -dc '0-9')
  if (( free_gb >= 150 )); then ok "disk: ${free_gb} GB free at $probe"
  elif (( free_gb >= 120 )); then warn "disk: ${free_gb} GB free (core+lipsync models need ~120 GB)"
  else bad "disk: ${free_gb} GB free at $probe (need >= 120 GB)"; fi

  for t in git curl ffmpeg ffprobe cmake ninja uv; do
    have "$t" && ok "$t: $(command -v "$t")" || warn "$t missing (installed by 'system')"
  done
  if have nvcc; then
    local v; v=$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
    version_ge "$v" 12.8 && ok "nvcc $v (can target sm_120)" || bad "nvcc $v too old for sm_120 (need >= 12.8)"
  else warn "nvcc missing (package 'cuda'; needed to build llama.cpp)"; fi
  have llama-server && ok "llama-server: $(command -v llama-server)" || warn "llama-server missing (installed by 'llama')"
  [[ -f "$COMFY_DIR/main.py" ]] && ok "ComfyUI: $COMFY_DIR ($(git -C "$COMFY_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'))" \
    || warn "ComfyUI not installed at $COMFY_DIR (installed by 'comfyui')"
  [[ -x "$REPO_DIR/.venv/bin/mvs" ]] && ok "mvs: $REPO_DIR/.venv/bin/mvs" || warn "mvstudio venv missing (installed by 'python')"
  [[ -n "${HF_TOKEN:-}" ]] && ok "HF_TOKEN set" || warn "HF_TOKEN not set: LTX-2.x is gated (accept the licence at huggingface.co/Lightricks/LTX-2.5, then export HF_TOKEN)"
  say; say "preflight: $FAILS failure(s), $WARNS warning(s)"
  return $(( FAILS > 0 ))
}

# ----------------------------------------------------------------------------- steps
step_system() {
  step "system packages"
  if ! is_arch; then warn "not Arch: install manually: base-devel git cmake ninja ffmpeg uv curl cuda (>=12.8)"; return 0; fi
  local want=(base-devel git cmake ninja ffmpeg uv curl cuda)
  [[ $INSTALL_DRIVER -eq 1 ]] && want+=(linux-headers nvidia-open-dkms nvidia-utils)
  local missing=()
  for p in "${want[@]}"; do pacman -Qi "$p" >/dev/null 2>&1 || missing+=("$p"); done
  if (( ${#missing[@]} == 0 )); then ok "all packages present: ${want[*]}"; else
    say "  missing: ${missing[*]}"
    confirm "install with pacman?" || die "aborted"
    run sudo pacman -S --needed --noconfirm "${missing[@]}"
  fi
  if ! have nvidia-smi; then
    warn "NVIDIA driver not installed. For an RTX 5090: sudo pacman -S linux-headers nvidia-open-dkms nvidia-utils && reboot (or rerun with --install-driver)"
  fi
  # CUDA lives in /opt/cuda on Arch
  if [[ -d /opt/cuda/bin ]] && ! have nvcc; then export PATH="/opt/cuda/bin:$PATH"; warn "add /opt/cuda/bin to PATH (e.g. in ~/.zshenv or ~/.bash_profile)"; fi
}

torch_index() {
  [[ -n "$TORCH_INDEX" ]] && { echo "$TORCH_INDEX"; return; }
  local drv; drv="$(gpu_info | cut -d'|' -f3 || true)"
  if [[ -n "$drv" ]] && ! version_ge "$drv" 580; then echo "https://download.pytorch.org/whl/cu128"
  else echo "https://download.pytorch.org/whl/cu130"; fi
}

step_python() {
  step "mvstudio python environment"
  have uv || die "uv missing (run the 'system' step)"
  [[ -d "$REPO_DIR/.venv" ]] || run uv venv --python "$PY_VERSION" "$REPO_DIR/.venv"
  run uv pip install --python "$REPO_DIR/.venv/bin/python" -e "${REPO_DIR}[asr,dev]"
  # faster-whisper's ctranslate2 wheels are linked against CUDA 12 libraries
  run uv pip install --python "$REPO_DIR/.venv/bin/python" nvidia-cublas-cu12 nvidia-cudnn-cu12
  run mkdir -p "$BIN_DIR"
  run ln -sf "$REPO_DIR/.venv/bin/mvs" "$BIN_DIR/mvs"
  ok "mvs installed ($BIN_DIR/mvs); make sure $BIN_DIR is in PATH"
}

step_comfyui() {
  step "ComfyUI"
  have git || die "git missing"
  if [[ ! -d "$COMFY_DIR/.git" ]]; then run git clone https://github.com/Comfy-Org/ComfyUI.git "$COMFY_DIR"
  else ok "ComfyUI present at $COMFY_DIR"; fi
  if [[ "$COMFY_REF" == latest ]]; then
    run git -C "$COMFY_DIR" fetch origin && run git -C "$COMFY_DIR" checkout master && run git -C "$COMFY_DIR" pull --ff-only
  else
    run git -C "$COMFY_DIR" fetch origin
    run git -C "$COMFY_DIR" checkout --quiet "$COMFY_REF"
  fi
  [[ -d "$COMFY_DIR/.venv" ]] || run uv venv --python "$PY_VERSION" "$COMFY_DIR/.venv"
  local idx; idx="$(torch_index)"; say "  PyTorch wheels: $idx"
  run uv pip install --python "$COMFY_DIR/.venv/bin/python" torch torchvision torchaudio --index-url "$idx"
  run uv pip install --python "$COMFY_DIR/.venv/bin/python" -r "$COMFY_DIR/requirements.txt"
  run mkdir -p "$COMFY_DIR/models/"{checkpoints,diffusion_models,text_encoders,vae,loras,latent_upscale_models}
  if [[ $DRY -eq 0 ]]; then
    "$COMFY_DIR/.venv/bin/python" - <<'PY' || bad "PyTorch cannot use the GPU (driver/CUDA mismatch?)"
import torch
assert torch.cuda.is_available(), "CUDA not available"
cap = torch.cuda.get_device_capability(0)
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); y = (x @ x).float().mean().item()
print(f"  ok    torch {torch.__version__} / CUDA {torch.version.cuda} on {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}, bf16 matmul works")
PY
  fi
}

step_llama() {
  step "llama.cpp (planner LLM server)"
  if have llama-server && [[ "${REBUILD:-0}" != 1 ]]; then ok "llama-server present: $(command -v llama-server)"; return 0; fi
  if [[ $DRY -eq 1 ]]; then have nvcc || warn "nvcc missing (the 'system' step installs the cuda package)"
  else have nvcc || die "nvcc missing: install the 'cuda' package (>= 12.8) and put /opt/cuda/bin in PATH"; fi
  have cmake || [[ $DRY -eq 1 ]] || die "cmake missing"
  [[ -d "$LLAMA_DIR/.git" ]] || run git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
  run git -C "$LLAMA_DIR" fetch origin
  run git -C "$LLAMA_DIR" checkout --quiet "$LLAMA_REF"
  [[ "$LLAMA_REF" != master ]] || run git -C "$LLAMA_DIR" pull --ff-only
  run cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=ON
  run cmake --build "$LLAMA_DIR/build" --target llama-server -j"$(nproc)"
  run mkdir -p "$BIN_DIR"
  run install -m 0755 "$LLAMA_DIR/build/bin/llama-server" "$BIN_DIR/llama-server"
  ok "llama-server installed to $BIN_DIR"
}

step_models() {
  step "models (${*:-core})"
  [[ $SKIP_MODELS -eq 1 ]] && { warn "skipped (--skip-models)"; return 0; }
  local mvs="$REPO_DIR/.venv/bin/mvs"; [[ -x "$mvs" ]] || die "run the 'python' step first"
  local groups=("$@"); (( ${#groups[@]} )) || groups=(core asr)
  local args=(); for g in "${groups[@]}"; do args+=(--groups "$g"); done
  MVS_COMFY_DIR="$COMFY_DIR" "$mvs" models status "${args[@]}" || true
  if [[ -z "${HF_TOKEN:-}" ]]; then warn "HF_TOKEN not set: gated LTX downloads will fail until you set it"; fi
  confirm "download missing files? (core is ~75 GB, lipsync adds ~42 GB)" || { warn "skipped"; return 0; }
  [[ $DRY -eq 1 ]] && { say "  (dry run: not downloading)"; return 0; }
  MVS_COMFY_DIR="$COMFY_DIR" "$mvs" models download "${args[@]}"
}

step_services() {
  step "systemd user service for ComfyUI"
  local unit_dir="$HOME/.config/systemd/user"
  run mkdir -p "$unit_dir"
  local unit="$unit_dir/comfyui.service"
  if [[ $DRY -eq 0 ]]; then
    sed -e "s|@COMFY_DIR@|$COMFY_DIR|g" "$REPO_DIR/systemd/comfyui.service" >"$unit"
  else say "  (would write $unit)"; fi
  run systemctl --user daemon-reload
  say "  start with:   systemctl --user enable --now comfyui"
  say "  headless box: loginctl enable-linger ${USER:-$(id -un)}   (keeps it running after logout)"
  say "  logs:         journalctl --user -u comfyui -f"
}

step_test() {
  step "verification"
  [[ $SKIP_TEST -eq 1 ]] && { warn "skipped (--skip-test)"; return 0; }
  local mvs="$REPO_DIR/.venv/bin/mvs"; [[ -x "$mvs" ]] || die "run the 'python' step first"
  export MVS_COMFY_DIR="$COMFY_DIR"
  local started=0 pid=""
  if ! curl -fsS http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
    if [[ $DRY -eq 1 ]]; then say "  (would start ComfyUI temporarily)"; else
      say "  starting ComfyUI temporarily ..."
      (cd "$COMFY_DIR" && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        nohup .venv/bin/python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch \
        >"$COMFY_DIR/mvs-setup.log" 2>&1 & echo $! >"$COMFY_DIR/.mvs-setup.pid")
      pid=$(cat "$COMFY_DIR/.mvs-setup.pid"); started=1
      for _ in $(seq 1 90); do curl -fsS http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && break; sleep 2; done
    fi
  fi
  say "  -> mvs doctor"
  [[ $DRY -eq 1 ]] || "$mvs" doctor || warn "doctor reported problems (see above)"
  say "  -> mvs selftest   (tiny real renders; steps with missing models are skipped)"
  [[ $DRY -eq 1 ]] || "$mvs" selftest || warn "selftest reported problems (see above)"
  if [[ $started -eq 1 && -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; rm -f "$COMFY_DIR/.mvs-setup.pid"; fi
  say; say "Run the unit tests of the pipeline itself: $REPO_DIR/.venv/bin/python -m pytest"
}

# ----------------------------------------------------------------------------- main
CMD="all"; ARGS=()
while (( $# )); do
  case "$1" in
    --dry-run) DRY=1 ;;
    --yes|-y) YES=1 ;;
    --install-driver) INSTALL_DRIVER=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --skip-test) SKIP_TEST=1 ;;
    --comfy-dir) COMFY_DIR="$2"; shift ;;
    --comfy-ref) COMFY_REF="$2"; shift ;;
    --torch-index) TORCH_INDEX="$2"; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    check|all|system|python|comfyui|llama|models|services|test) CMD="$1" ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
if [[ $DRY -eq 0 && "$CMD" != check ]]; then mkdir -p "$HOME/models/llm"; fi
export PATH="$BIN_DIR:/opt/cuda/bin:$PATH"

case "$CMD" in
  check)    check_all ;;
  system)   step_system ;;
  python)   step_python ;;
  comfyui)  step_comfyui ;;
  llama)    step_llama ;;
  models)   step_models "${ARGS[@]}" ;;
  services) step_services ;;
  test)     step_test ;;
  all)
    check_all || { (( INSTALL_DRIVER )) || true; }
    confirm "continue with installation?" || exit 1
    step_system; step_python; step_comfyui; step_llama
    step_models; step_services; step_test
    say; say "${C_OK}setup finished${C_OFF}. Next: mvs init ~/videos/my-song"
    ;;
esac
