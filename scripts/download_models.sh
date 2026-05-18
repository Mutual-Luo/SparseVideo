#!/usr/bin/env bash
# Download all-backbone model weights from Hugging Face.
# Proxy: localhost:10001
# Usage:
#   bash scripts/download_models.sh [--tier 1|2|3] [--all] [--dry-run] [--adopt-existing] [KEY ...]
#
# Examples:
#   bash scripts/download_models.sh --dry-run
#   bash scripts/download_models.sh --tier 1
#   bash scripts/download_models.sh cogvideox-5b-t2v mochi-1
#   bash scripts/download_models.sh --all    # all tiers

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python
HF_CLI=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/hf
MODELS_DIR=/home/dataset-assist-0/luojy/models
PROXY=http://127.0.0.1:10001
MAX_ATTEMPTS=5
BASE_DELAY=10        # seconds; doubles each retry (10 20 40 80 160)
DRY_RUN=0
ALL=0
ADOPT_EXISTING=0
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
    # WanAnimate / WanVACE run on existing Wan weights — no separate download

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
        --adopt-existing) ADOPT_EXISTING=1; shift ;;
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

validate_marker() {
    local repo=$1 marker=$2
    $PYTHON - "$marker" "$repo" <<'PYEOF'
import json
import sys
from pathlib import Path

marker = Path(sys.argv[1])
repo_id = sys.argv[2]

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
PYEOF
}

validate_destination() {
    local key=$1 repo=$2 dest=$3
    local marker="${dest}/${MARKER_NAME}"

    if [[ -f "$marker" ]]; then
        validate_marker "$repo" "$marker"
    elif [[ -d "$dest" && -n "$(find "$dest" -mindepth 1 -maxdepth 1 ! -name ".cache" -print -quit)" ]]; then
        if [[ $ADOPT_EXISTING -eq 1 ]]; then
            echo "  [WARN] ${key}: will adopt existing unmarked directory as ${repo}"
        else
            echo "  [ERROR] ${dest} is non-empty but has no ${MARKER_NAME}" >&2
            echo "          Refusing to mix checkpoints. If this directory is definitely ${repo}," >&2
            echo "          rerun with --adopt-existing for this first fixed-script run." >&2
            return 1
        fi
    fi
}

prepare_destination() {
    local key=$1 repo=$2 dest=$3
    local marker="${dest}/${MARKER_NAME}"

    mkdir -p "$dest"
    if [[ -f "$marker" ]]; then
        write_marker "$key" "$repo" "$dest" "in_progress"
    elif [[ -n "$(find "$dest" -mindepth 1 -maxdepth 1 ! -name ".cache" -print -quit)" ]]; then
        if [[ $ADOPT_EXISTING -eq 1 ]]; then
            echo "  [WARN] adopting existing unmarked directory as ${repo}"
            write_marker "$key" "$repo" "$dest" "in_progress"
        else
            echo "  [ERROR] ${dest} is non-empty but has no ${MARKER_NAME}" >&2
            echo "          Refusing to mix checkpoints. If this directory is definitely ${repo}," >&2
            echo "          rerun with --adopt-existing for this first fixed-script run." >&2
            return 1
        fi
    else
        write_marker "$key" "$repo" "$dest" "in_progress"
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

FAILED=()
SELECTED=()

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

if [[ $DRY_RUN -eq 0 ]]; then
    PREFLIGHT_FAILED=()
    echo "Preflight:"
    for entry in "${SELECTED[@]}"; do
        IFS='|' read -r key repo tier local_dir <<< "$entry"
        local_dir=${local_dir:-$key}
        dest="${MODELS_DIR}/${local_dir}"
        if validate_destination "$key" "$repo" "$dest"; then
            echo "  [OK]    ${key} → ${dest}"
        else
            PREFLIGHT_FAILED+=("$key")
        fi
    done
    echo
    if [[ ${#PREFLIGHT_FAILED[@]} -gt 0 ]]; then
        echo "[ERROR] preflight failed; no downloads were started" >&2
        echo "  Failed: ${PREFLIGHT_FAILED[*]}" >&2
        exit 1
    fi
fi

for entry in "${SELECTED[@]}"; do
    IFS='|' read -r key repo tier local_dir <<< "$entry"
    local_dir=${local_dir:-$key}
    dest="${MODELS_DIR}/${local_dir}"
    echo "[Tier ${tier}] ${key}"
    echo "         ${repo}"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "         → ${dest}  (dry-run)"
        echo
        continue
    fi

    # hf download handles resume and skip of already-complete files internally.
    # No shell-level skip — it would false-positive on partially downloaded repos.
    # The marker prevents reusing a directory for a different repo after a catalogue edit.
    if ! prepare_destination "$key" "$repo" "$dest"; then
        FAILED+=("$key")
        echo
        continue
    fi
    if download_model "$key" "$repo" "$dest"; then
        write_marker "$key" "$repo" "$dest" "complete"
        echo "  [OK]    ${key}"
    else
        FAILED+=("$key")
    fi
    echo
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "========================================"
echo "Done.  selected=${#SELECTED[@]}  failed=${#FAILED[@]}"
[[ ${#FAILED[@]} -gt 0 ]]  && echo "  Failed:  ${FAILED[*]}" && exit 1
exit 0
