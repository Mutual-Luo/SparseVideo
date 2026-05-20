#!/usr/bin/env bash
# Download all-backbone model weights from Hugging Face.
# Proxy: localhost:10001
# Usage:
#   bash scripts/download_models.sh [--tier 1|2|3] [--all] [--dry-run] [KEY ...]
#
# Examples:
#   bash scripts/download_models.sh --dry-run
#   bash scripts/download_models.sh --tier 1
#   bash scripts/download_models.sh cogvideox-5b-t2v mochi-1
#   bash scripts/download_models.sh --all    # all tiers

set -euo pipefail
trap 'echo; echo "[ABORT] interrupted"; kill -- -$$ 2>/dev/null; exit 130' INT TERM

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python
HF_CLI=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/hf
MODELS_DIR=/home/dataset-assist-0/luojy/models
PROXY=http://127.0.0.1:10001
MAX_ATTEMPTS=5
BASE_DELAY=10        # seconds; doubles each retry (10 20 40 80 160)
DRY_RUN=0
ALL=0
TIER_FILTER=99       # 99 = no filter

export HTTP_PROXY=$PROXY
export HTTPS_PROXY=$PROXY
export http_proxy=$PROXY
export https_proxy=$PROXY

# ── Model catalogue ───────────────────────────────────────────────────────────
# Format: "KEY|REPO_ID|TIER|LOCAL_DIR"
CATALOGUE=(
    # Tier 1 — Wan family / same SkyReelsV2Transformer3DModel block (reuse wan processor)
    # Using 720P Diffusers-format repos (WanPipeline + SkyReelsV2Transformer3DModel)
    "skyreels-v2-t2v-14b|Skywork/SkyReels-V2-T2V-14B-720P-Diffusers|1|skyreels-v2-t2v-14b"
    "skyreels-v2-i2v-14b|Skywork/SkyReels-V2-I2V-14B-720P-Diffusers|1|skyreels-v2-i2v-14b"
    "wan22-animate-14b|Wan-AI/Wan2.2-Animate-14B-Diffusers|1|Wan2.2-Animate-14B-Diffusers"
    "wan21-vace-1.3b|Wan-AI/Wan2.1-VACE-1.3B-diffusers|1|Wan2.1-VACE-1.3B-diffusers"
    "wan21-vace-14b|Wan-AI/Wan2.1-VACE-14B-diffusers|1|Wan2.1-VACE-14B-diffusers"

    # Tier 2 — separate self-attn, new processor needed
    "ltx-video|Lightricks/LTX-Video|2|ltx-video"
    "cogvideox-5b-t2v|THUDM/CogVideoX-5b|2|CogVideoX-5b"
    "cogvideox-5b-i2v|THUDM/CogVideoX-5b-I2V|2|CogVideoX-5b-I2V"
    "allegro|rhymes-ai/Allegro|2|allegro"

    # Tier 3 — joint / dual-stream attention
    "mochi-1|genmo/mochi-1-preview|3|mochi-1"
    # EasyAnimate V5.1 diffusers-format (EasyAnimatePipeline + EasyAnimateTransformer3DModel)
    "easyanimate-v5-t2v-12b|alibaba-pai/EasyAnimateV5.1-12b-zh-diffusers|3|easyanimate-v5-t2v-12b"
    # MotifVideo / LTX Video 2: no confirmed public HF repo yet — add when known
)

MARKER_NAME=.sparsevideo_download.json

usage() {
    sed -n '2,11p' "$0" >&2
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

print_available_keys() {
    local entry key repo tier local_dir
    echo "Available keys:" >&2
    for entry in "${CATALOGUE[@]}"; do
        IFS='|' read -r key repo tier local_dir <<< "$entry"
        echo "  ${key}" >&2
    done
}

is_known_key() {
    local requested=$1 entry key repo tier local_dir
    for entry in "${CATALOGUE[@]}"; do
        IFS='|' read -r key repo tier local_dir <<< "$entry"
        [[ "$requested" == "$key" ]] && return 0
    done
    return 1
}

# ── Parse args ────────────────────────────────────────────────────────────────
EXPLICIT_KEYS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)  usage; exit 0 ;;
        --tier)
            [[ $# -ge 2 && "$2" != -* ]] || die "--tier requires one of: 1, 2, 3"
            case "$2" in
                1|2|3) TIER_FILTER="$2" ;;
                *)     die "--tier must be one of: 1, 2, 3" ;;
            esac
            shift 2
            ;;
        --dry-run)  DRY_RUN=1; shift ;;
        --all)      ALL=1; shift ;;
        --adopt-existing) shift ;;  # Backward-compatible no-op; existing dirs are auto-checked.
        -*)         echo "Unknown option: $1"; exit 1 ;;
        *)          EXPLICIT_KEYS+=("$1"); shift ;;
    esac
done

if [[ $ALL -eq 1 && ( ${#EXPLICIT_KEYS[@]} -gt 0 || $TIER_FILTER -ne 99 ) ]]; then
    die "--all cannot be combined with --tier or explicit keys"
fi

if [[ $DRY_RUN -eq 0 && $ALL -eq 0 && ${#EXPLICIT_KEYS[@]} -eq 0 && $TIER_FILTER -eq 99 ]]; then
    die "refusing to download every tier by default; pass --all, --tier N, or explicit keys"
fi

if [[ ${#EXPLICIT_KEYS[@]} -gt 0 ]]; then
    UNKNOWN_KEYS=()
    for key in "${EXPLICIT_KEYS[@]}"; do
        is_known_key "$key" || UNKNOWN_KEYS+=("$key")
    done
    if [[ ${#UNKNOWN_KEYS[@]} -gt 0 ]]; then
        echo "[ERROR] Unknown key(s): ${UNKNOWN_KEYS[*]}" >&2
        print_available_keys
        exit 1
    fi
fi

write_marker() {
    local key=$1 repo=$2 dest=$3 status=$4
    $PYTHON - "$dest/$MARKER_NAME" "$key" "$repo" "$status" <<'PYEOF'
import json
import sys
import time
from pathlib import Path

marker = Path(sys.argv[1])
key, repo_id, status = sys.argv[2:5]
payload = {}

if marker.exists():
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[ERROR] invalid download marker: {marker}: {exc}", flush=True)
        sys.exit(1)
    existing_repo = payload.get("repo_id")
    if existing_repo and existing_repo != repo_id:
        print(
            f"[ERROR] destination marker belongs to {existing_repo}, refusing to write {repo_id}",
            flush=True,
        )
        sys.exit(1)

now = time.time()
payload.update(
    {
        "key": key,
        "repo_id": repo_id,
        "status": status,
        "updated_at": now,
    }
)
payload.setdefault("created_at", now)
marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PYEOF
}

check_model_state() {
    local key=$1 repo=$2 dest=$3
    $PYTHON - "$key" "$repo" "$dest" "$MARKER_NAME" <<'PYEOF'
import json
import os
import sys
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

key, repo_id, dest_arg, marker_name = sys.argv[1:5]
dest = Path(dest_arg)
marker = dest / marker_name
ignore = ["flax_model*", "tf_model*", "rust_model*", "onnx/*", "*.msgpack", "*.h5"]
checkpoint_suffixes = (".safetensors", ".bin", ".ckpt", ".pt", ".pth")

def gib(num: int) -> str:
    return f"{num / 1024**3:.1f}GiB"

if marker.exists():
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"  [ERROR] {key}: invalid marker {marker}: {exc}", file=sys.stderr)
        print("error")
        sys.exit(0)
    existing_repo = payload.get("repo_id")
    if existing_repo and existing_repo != repo_id:
        print(
            f"  [ERROR] {key}: marker belongs to {existing_repo}, refusing to write {repo_id}",
            file=sys.stderr,
        )
        print("error")
        sys.exit(0)

api = HfApi()
remote = {}
for item in api.list_repo_tree(repo_id, repo_type="model", recursive=True, expand=True):
    if not isinstance(item, RepoFile):
        continue
    path = item.path
    if any(fnmatch(path, pattern) for pattern in ignore):
        continue
    remote[path] = int(item.size or 0)

if not remote:
    print(f"  [ERROR] {key}: remote repo has no matching files: {repo_id}", file=sys.stderr)
    print("error")
    sys.exit(0)

local = {}
if dest.exists():
    for path in dest.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(dest).as_posix()
        except ValueError:
            continue
        if rel == marker_name or rel.startswith(".cache/"):
            continue
        local[rel] = path.stat().st_size

missing = [(path, size) for path, size in remote.items() if path not in local]
size_mismatch = [
    (path, remote_size, local[path])
    for path, remote_size in remote.items()
    if path in local and remote_size and local[path] != remote_size
]
extra_checkpoint = [
    path for path in local
    if path not in remote and path.endswith(checkpoint_suffixes)
]
incomplete_count = 0
incomplete_size = 0
cache_dir = dest / ".cache" / "huggingface" / "download"
if cache_dir.exists():
    incomplete_files = list(cache_dir.rglob("*.incomplete"))
    incomplete_count = len(incomplete_files)
    incomplete_size = sum(path.stat().st_size for path in incomplete_files)

remote_size = sum(remote.values())
local_expected_size = sum(size for path, size in local.items() if path in remote)
missing_size = sum(size for _, size in missing)
remaining_size = max(missing_size - incomplete_size, 0)

if extra_checkpoint:
    print(f"  [ERROR] {key}: found checkpoint files that are not in {repo_id}", file=sys.stderr)
    for path in extra_checkpoint[:5]:
        print(f"          extra checkpoint: {path}", file=sys.stderr)
    print("error")
elif not missing and not size_mismatch:
    print(
        f"  [CHECK] {key}: complete ({len(remote)} files, {gib(remote_size)})",
        file=sys.stderr,
    )
    print("complete")
else:
    print(
        f"  [CHECK] {key}: incomplete "
        f"(missing={len(missing)}, damaged={len(size_mismatch)}, "
        f"cached_partial={incomplete_count}/{gib(incomplete_size)}, need≈{gib(remaining_size)}, "
        f"local_expected={gib(local_expected_size)}/{gib(remote_size)})",
        file=sys.stderr,
    )
    for path, size in missing[:5]:
        print(f"          missing: {path} ({gib(size)})", file=sys.stderr)
    for path, remote_size, local_size in size_mismatch[:5]:
        print(
            f"          damaged: {path} local={local_size} remote={remote_size}",
            file=sys.stderr,
        )
    print("incomplete")
PYEOF
}

remove_damaged_files() {
    local repo=$1 dest=$2
    $PYTHON - "$repo" "$dest" "$MARKER_NAME" <<'PYEOF'
import sys
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

repo_id, dest_arg, marker_name = sys.argv[1:4]
dest = Path(dest_arg)
ignore = ["flax_model*", "tf_model*", "rust_model*", "onnx/*", "*.msgpack", "*.h5"]

api = HfApi()
remote = {}
for item in api.list_repo_tree(repo_id, repo_type="model", recursive=True, expand=True):
    if not isinstance(item, RepoFile):
        continue
    path = item.path
    if any(fnmatch(path, pattern) for pattern in ignore):
        continue
    remote[path] = int(item.size or 0)

for rel, remote_size in remote.items():
    path = dest / rel
    if path.is_file() and remote_size and path.stat().st_size != remote_size:
        print(f"  [REPAIR] removing damaged local file: {path}", flush=True)
        path.unlink()
PYEOF
}

prepare_destination() {
    local key=$1 repo=$2 dest=$3

    mkdir -p "$dest"
    write_marker "$key" "$repo" "$dest" "in_progress"
}

finalize_download() {
    local key=$1 repo=$2 dest=$3
    local state

    if ! state=$(check_model_state "$key" "$repo" "$dest"); then
        return 1
    fi
    if [[ "$state" == "complete" ]]; then
        write_marker "$key" "$repo" "$dest" "complete"
        return 0
    else
        echo "  [ERROR] ${key} is still incomplete after hf download" >&2
        return 1
    fi
}

# ── Download function with retry ──────────────────────────────────────────────
# Args: KEY REPO_ID LOCAL_DIR
download_model() {
    local key=$1 repo=$2 dest=$3
    local attempt delay
    local token_args=()

    [[ -x "$HF_CLI" ]] || die "hf CLI not found or not executable: $HF_CLI"
    if [[ -n "${HF_TOKEN:-}" ]]; then
        token_args=(--token "$HF_TOKEN")
    fi

    for (( attempt=1; attempt<=MAX_ATTEMPTS; attempt++ )); do
        echo "  attempt ${attempt}/${MAX_ATTEMPTS}: ${repo} → ${dest}"
        if "$HF_CLI" download "$repo" \
            --repo-type model \
            --local-dir "$dest" \
            --no-force-download \
            --exclude "flax_model*" \
            --exclude "tf_model*" \
            --exclude "rust_model*" \
            --exclude "onnx/*" \
            --exclude "*.msgpack" \
            --exclude "*.h5" \
            "${token_args[@]}"
        then
            return 0   # success
        fi

        if (( attempt < MAX_ATTEMPTS )); then
            delay=$(( BASE_DELAY * (1 << (attempt - 1)) ))
            echo "  [RETRY] waiting ${delay}s before attempt $((attempt+1))..."
            sleep "$delay"
        fi
    done

    echo "  [FAIL] ${key} exhausted ${MAX_ATTEMPTS} attempts"
    return 1
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "Proxy:   $PROXY"
echo "Dest:    $MODELS_DIR"
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "HF auth: token from HF_TOKEN"
else
    echo "HF auth: cached token (~/.cache/huggingface/token) or anonymous"
fi
echo

SELECTED=()
STATES=()
FAILED=()
SKIPPED=()
DOWNLOADED=()

for entry in "${CATALOGUE[@]}"; do
    IFS='|' read -r key repo tier local_dir <<< "$entry"

    # Apply filters
    if (( tier > TIER_FILTER )); then
        continue
    fi
    if [[ ${#EXPLICIT_KEYS[@]} -gt 0 ]]; then
        match=0
        for k in "${EXPLICIT_KEYS[@]}"; do [[ "$k" == "$key" ]] && match=1; done
        [[ $match -eq 0 ]] && continue
    fi

    SELECTED+=("$entry")
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
    echo "[ERROR] No catalogue entries matched the requested filters" >&2
    exit 1
fi

PREFLIGHT_FAILED=()
echo "Preflight:"
for entry in "${SELECTED[@]}"; do
    IFS='|' read -r key repo tier local_dir <<< "$entry"
    local_dir=${local_dir:-$key}
    dest="${MODELS_DIR}/${local_dir}"
    if state=$(check_model_state "$key" "$repo" "$dest"); then
        STATES+=("$state")
    else
        state=error
        STATES+=("$state")
    fi
    if [[ "$state" == "error" ]]; then
        PREFLIGHT_FAILED+=("$key")
    fi
done
echo

if [[ ${#PREFLIGHT_FAILED[@]} -gt 0 ]]; then
    echo "[ERROR] preflight failed; no downloads were started" >&2
    echo "  Failed: ${PREFLIGHT_FAILED[*]}" >&2
    exit 1
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "========================================"
    echo "Dry run. selected=${#SELECTED[@]}"
    exit 0
fi

for index in "${!SELECTED[@]}"; do
    entry="${SELECTED[$index]}"
    state="${STATES[$index]}"
    IFS='|' read -r key repo tier local_dir <<< "$entry"
    local_dir=${local_dir:-$key}
    dest="${MODELS_DIR}/${local_dir}"
    echo "[Tier ${tier}] ${key}"
    echo "         ${repo}"

    if [[ "$state" == "complete" ]]; then
        write_marker "$key" "$repo" "$dest" "complete"
        SKIPPED+=("$key")
        echo "  [SKIP]  already complete"
        echo
        continue
    fi

    # hf download resumes partial files and skips complete files. Local files with
    # a wrong size are removed first so they are redownloaded instead of reused.
    remove_damaged_files "$repo" "$dest"
    if ! prepare_destination "$key" "$repo" "$dest"; then
        FAILED+=("$key")
        echo
        continue
    fi
    if download_model "$key" "$repo" "$dest"; then
        if finalize_download "$key" "$repo" "$dest"; then
            DOWNLOADED+=("$key")
            echo "  [OK]    ${key}"
        else
            FAILED+=("$key")
        fi
    else
        FAILED+=("$key")
    fi
    echo
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "========================================"
echo "Done.  selected=${#SELECTED[@]}  skipped=${#SKIPPED[@]}  downloaded=${#DOWNLOADED[@]}  failed=${#FAILED[@]}"
[[ ${#SKIPPED[@]} -gt 0 ]] && echo "  Skipped:    ${SKIPPED[*]}"
[[ ${#DOWNLOADED[@]} -gt 0 ]] && echo "  Downloaded: ${DOWNLOADED[*]}"
[[ ${#FAILED[@]} -gt 0 ]]  && echo "  Failed:  ${FAILED[*]}" && exit 1
exit 0
