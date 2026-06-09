#!/usr/bin/env bash
# Download DiffSynth-Studio-native model bundles for SparseVideo integration.
# Usage:
#   bash scripts/download/download_diffsynth_models.sh [--list]
#   bash scripts/download/download_diffsynth_models.sh [--source auto|modelscope-first|huggingface-first|hf-first|huggingface|modelscope] [--hf-endpoint URL] [--proxy URL|--no-proxy] [--link-root PATH] [--tier 1|2|3] [--kind wan|mova|ltx2] [--all] [--dry-run] [KEY ...]
#
# Examples:
#   bash scripts/download/download_diffsynth_models.sh --list
#   bash scripts/download/download_diffsynth_models.sh --dry-run wan21-t2v-1.3b
#   bash scripts/download/download_diffsynth_models.sh --tier 1 --source huggingface
#   bash scripts/download/download_diffsynth_models.sh mova-720p --source modelscope
#   bash scripts/download/download_diffsynth_models.sh wan21-t2v-14b
#   # Only use proxy after confirming the mirror path cannot fetch the required files.
#   bash scripts/download/download_diffsynth_models.sh wan21-t2v-14b --proxy http://127.0.0.1:10000
#   bash scripts/download/download_diffsynth_models.sh --link-root /path/to/existing/models wan21-t2v-1.3b
# bash scripts/download/download_diffsynth_models.sh \
#     --all \
#     --source auto \
#     --hf-endpoint https://hf-mirror.com \
#     --no-proxy


set -euo pipefail
trap 'echo; echo "[ABORT] interrupted"; exit 130' INT TERM

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPTS_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python}
HF_CLI=${HF_CLI:-/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/hf}
MODELSCOPE_CLI=${MODELSCOPE_CLI:-/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/modelscope}
MODEL_ROOT=${MODEL_ROOT:-/home/dataset-assist-0/public-models}
PROXY=${PROXY:-}
SOURCE=${SOURCE:-auto}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
LINK_ROOT=
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
BASE_DELAY=${BASE_DELAY:-10}
MAX_WORKERS=${MAX_WORKERS:-8}

DRY_RUN=0
LIST=0
ALL=0
SKIP_PROXY=1
TIER_FILTER=
KIND_FILTER=
EXPLICIT_KEYS=()

WAN_TOKENIZER="Wan-AI/Wan2.1-T2V-1.3B::google/umt5-xxl/tokenizer.json;;Wan-AI/Wan2.1-T2V-1.3B::google/umt5-xxl/tokenizer_config.json;;Wan-AI/Wan2.1-T2V-1.3B::google/umt5-xxl/spiece.model;;Wan-AI/Wan2.1-T2V-1.3B::google/umt5-xxl/special_tokens_map.json"
WAN_T5="DiffSynth-Studio/Wan-Series-Converted-Safetensors::models_t5_umt5-xxl-enc-bf16.safetensors"
WAN_CLIP="DiffSynth-Studio/Wan-Series-Converted-Safetensors::models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"
WAN21_VAE="DiffSynth-Studio/Wan-Series-Converted-Safetensors::Wan2.1_VAE.safetensors"
WAN22_VAE="DiffSynth-Studio/Wan-Series-Converted-Safetensors::Wan2.2_VAE.safetensors"
WAN_COMMON_21="${WAN_TOKENIZER};;${WAN_T5};;${WAN21_VAE}"
WAN_COMMON_21_IMAGE="${WAN_COMMON_21};;${WAN_CLIP}"
WAN_COMMON_22="${WAN_TOKENIZER};;${WAN_T5};;${WAN22_VAE}"
WAN_COMMON_22_IMAGE="${WAN_COMMON_22};;${WAN_CLIP}"
LTX2_GEMMA="Lightricks/gemma-3-12b-it-qat-q4_0-unquantized::tokenizer.json;;Lightricks/gemma-3-12b-it-qat-q4_0-unquantized::tokenizer.model;;Lightricks/gemma-3-12b-it-qat-q4_0-unquantized::tokenizer_config.json;;Lightricks/gemma-3-12b-it-qat-q4_0-unquantized::preprocessor_config.json;;Lightricks/gemma-3-12b-it-qat-q4_0-unquantized::processor_config.json;;Lightricks/gemma-3-12b-it-qat-q4_0-unquantized::special_tokens_map.json;;Lightricks/gemma-3-12b-it-qat-q4_0-unquantized::added_tokens.json;;Lightricks/gemma-3-12b-it-qat-q4_0-unquantized::chat_template.json;;Lightricks/gemma-3-12b-it-qat-q4_0-unquantized::config.json;;Lightricks/gemma-3-12b-it-qat-q4_0-unquantized::generation_config.json;;Lightricks/gemma-3-12b-it-qat-q4_0-unquantized::model.safetensors.index.json;;Lightricks/gemma-3-12b-it-qat-q4_0-unquantized::model*.safetensors"
S2V_WAV2VEC="Wan-AI/Wan2.2-S2V-14B::wav2vec2-large-xlsr-53-english/model.safetensors;;Wan-AI/Wan2.2-S2V-14B::wav2vec2-large-xlsr-53-english/preprocessor_config.json;;Wan-AI/Wan2.2-S2V-14B::wav2vec2-large-xlsr-53-english/vocab.json;;Wan-AI/Wan2.2-S2V-14B::wav2vec2-large-xlsr-53-english/special_tokens_map.json"
MOVA_TOKENIZER="openmoss/MOVA-720p::tokenizer/tokenizer.json;;openmoss/MOVA-720p::tokenizer/tokenizer_config.json;;openmoss/MOVA-720p::tokenizer/special_tokens_map.json"

# Format: "KEY|TIER|KIND|DESCRIPTION|REPO::PATTERN;;REPO::PATTERN"
CATALOGUE=(
    "wan21-t2v-1.3b|1|wan|DiffSynth Wan2.1 text-to-video 1.3B|${WAN_COMMON_21};;Wan-AI/Wan2.1-T2V-1.3B::diffusion_pytorch_model*.safetensors"
    "wan21-speedcontrol-1.3b|1|wan|DiffSynth Wan2.1 1.3B speed-control motion controller|${WAN_COMMON_21};;Wan-AI/Wan2.1-T2V-1.3B::diffusion_pytorch_model*.safetensors;;DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1::model.safetensors"
    "wan21-t2v-14b|1|wan|DiffSynth Wan2.1 text-to-video 14B|${WAN_COMMON_21};;Wan-AI/Wan2.1-T2V-14B::diffusion_pytorch_model*.safetensors"
    "wan21-i2v-14b-480p|1|wan|DiffSynth Wan2.1 image-to-video 14B 480P|${WAN_COMMON_21_IMAGE};;Wan-AI/Wan2.1-I2V-14B-480P::diffusion_pytorch_model*.safetensors"
    "wan21-i2v-14b-720p|1|wan|DiffSynth Wan2.1 image-to-video 14B 720P|${WAN_COMMON_21_IMAGE};;Wan-AI/Wan2.1-I2V-14B-720P::diffusion_pytorch_model*.safetensors"
    "wan21-flf2v-14b-720p|1|wan|DiffSynth Wan2.1 first-last-frame-to-video 14B 720P|${WAN_COMMON_21_IMAGE};;Wan-AI/Wan2.1-FLF2V-14B-720P::diffusion_pytorch_model*.safetensors"
    "wan21-fun-1.3b-control|1|wan|DiffSynth Wan2.1-Fun 1.3B Control|${WAN_COMMON_21};;PAI/Wan2.1-Fun-1.3B-Control::diffusion_pytorch_model*.safetensors"
    "wan21-fun-1.3b-inp|1|wan|DiffSynth Wan2.1-Fun 1.3B InP|${WAN_COMMON_21_IMAGE};;PAI/Wan2.1-Fun-1.3B-InP::diffusion_pytorch_model*.safetensors"
    "wan21-fun-14b-control|1|wan|DiffSynth Wan2.1-Fun 14B Control|${WAN_COMMON_21};;PAI/Wan2.1-Fun-14B-Control::diffusion_pytorch_model*.safetensors"
    "wan21-fun-14b-inp|1|wan|DiffSynth Wan2.1-Fun 14B InP|${WAN_COMMON_21_IMAGE};;PAI/Wan2.1-Fun-14B-InP::diffusion_pytorch_model*.safetensors"
    "wan21-fun-v11-1.3b-control|1|wan|DiffSynth Wan2.1-Fun V1.1 1.3B Control|${WAN_COMMON_21_IMAGE};;PAI/Wan2.1-Fun-V1.1-1.3B-Control::diffusion_pytorch_model*.safetensors"
    "wan21-fun-v11-1.3b-control-camera|1|wan|DiffSynth Wan2.1-Fun V1.1 1.3B Control-Camera|${WAN_COMMON_21_IMAGE};;PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera::diffusion_pytorch_model*.safetensors"
    "wan21-fun-v11-14b-control|1|wan|DiffSynth Wan2.1-Fun V1.1 14B Control|${WAN_COMMON_21_IMAGE};;PAI/Wan2.1-Fun-V1.1-14B-Control::diffusion_pytorch_model*.safetensors"
    "wan21-fun-v11-14b-control-camera|1|wan|DiffSynth Wan2.1-Fun V1.1 14B Control-Camera|${WAN_COMMON_21_IMAGE};;PAI/Wan2.1-Fun-V1.1-14B-Control-Camera::diffusion_pytorch_model*.safetensors"
    "wan21-vace-1.3b|1|wan|DiffSynth VACE Wan2.1 1.3B preview|${WAN_COMMON_21};;iic/VACE-Wan2.1-1.3B-Preview::diffusion_pytorch_model*.safetensors"
    "wan21-vace-14b|1|wan|DiffSynth VACE Wan2.1 14B|${WAN_COMMON_21};;Wan-AI/Wan2.1-VACE-14B::diffusion_pytorch_model*.safetensors"
    "wan22-animate-14b|1|wan|DiffSynth Wan2.2 Animate 14B|${WAN_COMMON_21_IMAGE};;Wan-AI/Wan2.2-Animate-14B::diffusion_pytorch_model*.safetensors"
    "wan22-t2v-a14b|1|wan|DiffSynth Wan2.2 text-to-video A14B high/low-noise models|${WAN_COMMON_21};;Wan-AI/Wan2.2-T2V-A14B::high_noise_model/diffusion_pytorch_model*.safetensors;;Wan-AI/Wan2.2-T2V-A14B::low_noise_model/diffusion_pytorch_model*.safetensors"
    "wan22-i2v-a14b|1|wan|DiffSynth Wan2.2 image-to-video A14B high/low-noise models|${WAN_COMMON_21_IMAGE};;Wan-AI/Wan2.2-I2V-A14B::high_noise_model/diffusion_pytorch_model*.safetensors;;Wan-AI/Wan2.2-I2V-A14B::low_noise_model/diffusion_pytorch_model*.safetensors"
    "wan22-ti2v-5b|1|wan|DiffSynth Wan2.2 TI2V 5B|${WAN_COMMON_22};;Wan-AI/Wan2.2-TI2V-5B::diffusion_pytorch_model*.safetensors"
    "wan22-s2v-14b|1|wan|DiffSynth Wan2.2 speech-to-video 14B|${WAN_COMMON_21_IMAGE};;Wan-AI/Wan2.2-S2V-14B::diffusion_pytorch_model*.safetensors;;${S2V_WAV2VEC}"
    "wan22-fun-a14b-control|1|wan|DiffSynth Wan2.2-Fun A14B Control high/low-noise DiTs|${WAN_COMMON_21};;PAI/Wan2.2-Fun-A14B-Control::high_noise_model/diffusion_pytorch_model*.safetensors;;PAI/Wan2.2-Fun-A14B-Control::low_noise_model/diffusion_pytorch_model*.safetensors"
    "wan22-fun-a14b-control-camera|1|wan|DiffSynth Wan2.2-Fun A14B Control-Camera high/low-noise DiTs|${WAN_COMMON_21};;PAI/Wan2.2-Fun-A14B-Control-Camera::high_noise_model/diffusion_pytorch_model*.safetensors;;PAI/Wan2.2-Fun-A14B-Control-Camera::low_noise_model/diffusion_pytorch_model*.safetensors"
    "longcat-video|1|wan|DiffSynth LongCat-Video on WanVideoPipeline|${WAN_COMMON_21};;meituan-longcat/LongCat-Video::dit/diffusion_pytorch_model*.safetensors"
    "video-as-prompt-wan21-14b|1|wan|DiffSynth Video-as-Prompt Wan2.1 14B|${WAN_COMMON_21_IMAGE};;ByteDance/Video-As-Prompt-Wan2.1-14B::transformer/diffusion_pytorch_model*.safetensors"
    "krea-realtime-video|1|wan|DiffSynth Krea realtime video 14B|${WAN_COMMON_21};;krea/krea-realtime-video::krea-realtime-video-14b.safetensors"
    "mova-720p|2|mova|DiffSynth MOVA 720P plus Wan video backbone components|${WAN_COMMON_21};;Wan-AI/Wan2.1-T2V-14B::diffusion_pytorch_model*.safetensors;;${MOVA_TOKENIZER};;openmoss/MOVA-720p::audio_dit/diffusion_pytorch_model.safetensors;;openmoss/MOVA-720p::audio_vae/diffusion_pytorch_model.safetensors;;openmoss/MOVA-720p::dual_tower_bridge/diffusion_pytorch_model.safetensors"
    "ltx2|3|ltx2|DiffSynth LTX-2 repackaged audio-video components|${LTX2_GEMMA};;DiffSynth-Studio/LTX-2-Repackage::transformer.safetensors;;DiffSynth-Studio/LTX-2-Repackage::video_vae_encoder.safetensors;;DiffSynth-Studio/LTX-2-Repackage::video_vae_decoder.safetensors;;DiffSynth-Studio/LTX-2-Repackage::audio_vae_encoder.safetensors;;DiffSynth-Studio/LTX-2-Repackage::audio_vae_decoder.safetensors;;DiffSynth-Studio/LTX-2-Repackage::audio_vocoder.safetensors;;DiffSynth-Studio/LTX-2-Repackage::text_encoder_post_modules.safetensors"
    "ltx23|3|ltx2|DiffSynth LTX-2.3 source checkpoint plus latent upsampler components|${LTX2_GEMMA};;DiffSynth-Studio/LTX-2.3-Repackage::transformer.safetensors;;DiffSynth-Studio/LTX-2.3-Repackage::video_vae_encoder.safetensors;;DiffSynth-Studio/LTX-2.3-Repackage::video_vae_decoder.safetensors;;DiffSynth-Studio/LTX-2.3-Repackage::audio_vocoder.safetensors;;DiffSynth-Studio/LTX-2.3-Repackage::text_encoder_post_modules.safetensors;;Lightricks/LTX-2.3::ltx-2.3-22b-dev.safetensors;;Lightricks/LTX-2.3::ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
)

declare -A MODELSCOPE_ONLY_REPOS=(
    ["DiffSynth-Studio/Wan-Series-Converted-Safetensors"]=1
    ["DiffSynth-Studio/LTX-2-Repackage"]=1
    ["DiffSynth-Studio/LTX-2.3-Repackage"]=1
    ["iic/VACE-Wan2.1-1.3B-Preview"]=1
    ["openmoss/MOVA-720p"]=1
    ["PAI/Wan2.1-Fun-1.3B-Control"]=1
    ["PAI/Wan2.1-Fun-1.3B-InP"]=1
    ["PAI/Wan2.1-Fun-14B-Control"]=1
    ["PAI/Wan2.1-Fun-14B-InP"]=1
    ["PAI/Wan2.1-Fun-V1.1-1.3B-Control"]=1
    ["PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera"]=1
    ["PAI/Wan2.1-Fun-V1.1-14B-Control"]=1
    ["PAI/Wan2.1-Fun-V1.1-14B-Control-Camera"]=1
    ["PAI/Wan2.2-Fun-A14B-Control"]=1
    ["PAI/Wan2.2-Fun-A14B-Control-Camera"]=1
)

declare -A HF_ONLY_REPOS=(
    ["ByteDance/Video-As-Prompt-Wan2.1-14B"]=1
)

usage() {
    sed -n '2,11p' "$0" >&2
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

split_components() {
    local rest=$1
    local part
    while [[ -n "$rest" ]]; do
        if [[ "$rest" == *";;"* ]]; then
            part=${rest%%;;*}
            rest=${rest#*;;}
        else
            part=$rest
            rest=
        fi
        [[ -n "$part" ]] && printf '%s\n' "$part"
    done
}

include_pattern_for_cli() {
    local pattern=$1
    if [[ -z "$pattern" || "$pattern" == "*" ]]; then
        return 0
    fi
    if [[ "$pattern" == */ ]]; then
        printf '%s**' "$pattern"
    else
        printf '%s' "$pattern"
    fi
}

print_catalog() {
    local entry key tier pipeline_kind description components
    echo "Available DiffSynth-Studio bundles:"
    for entry in "${CATALOGUE[@]}"; do
        IFS='|' read -r key tier pipeline_kind description components <<< "$entry"
        printf '  %-30s tier=%s kind=%-5s %s\n' "$key" "$tier" "$pipeline_kind" "$description"
    done
    echo
    echo "Deferred/local-only DiffSynth models (not downloaded by this script):"
    printf '  %-30s kind=%-5s %s origin=%s:%s\n' \
        "wan22-dancer-14b" \
        "wan" \
        "DiffSynth Wan2.2-Dancer 14B WanToDance global model" \
        "Wan-AI/Wan2.2-Dancer-14B" \
        "global_model.safetensors"
}

print_available_keys() {
    local entry key tier pipeline_kind description components
    echo "Available keys:" >&2
    for entry in "${CATALOGUE[@]}"; do
        IFS='|' read -r key tier pipeline_kind description components <<< "$entry"
        echo "  $key" >&2
    done
}

is_known_key() {
    local requested=$1
    local entry key tier pipeline_kind description components
    for entry in "${CATALOGUE[@]}"; do
        IFS='|' read -r key tier pipeline_kind description components <<< "$entry"
        [[ "$requested" == "$key" ]] && return 0
    done
    return 1
}

print_bundle() {
    local entry=$1
    local key tier pipeline_kind description components component repo pattern display
    local -A seen=()
    IFS='|' read -r key tier pipeline_kind description components <<< "$entry"
    echo
    echo "[$key] $description"
    while IFS= read -r component; do
        [[ -n "${seen[$component]:-}" ]] && continue
        seen[$component]=1
        repo=${component%%::*}
        pattern=
        [[ "$component" == *"::"* ]] && pattern=${component#*::}
        display=${pattern:-*}
        echo "  - ${repo}:${display}"
    done < <(split_components "$components")
}

configure_proxy() {
    [[ -n "$PROXY" ]] || return 0
    export HTTP_PROXY=$PROXY
    export HTTPS_PROXY=$PROXY
    export ALL_PROXY=$PROXY
    export http_proxy=$PROXY
    export https_proxy=$PROXY
    export all_proxy=$PROXY
}

clear_proxy() {
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
    PROXY=
    export -n PROXY 2>/dev/null || true
}

resolve_tools() {
    if [[ "$SOURCE" == "huggingface" || "$SOURCE" == "auto" || "$SOURCE" == "modelscope-first" || "$SOURCE" == "huggingface-first" ]]; then
        if [[ ! -x "$HF_CLI" ]]; then
            HF_CLI=$(command -v hf || true)
        fi
        [[ -n "$HF_CLI" && -x "$HF_CLI" ]] || die "hf CLI not found"
    fi
    if [[ "$SOURCE" == "modelscope" || "$SOURCE" == "auto" || "$SOURCE" == "modelscope-first" || "$SOURCE" == "huggingface-first" ]]; then
        if [[ ! -x "$MODELSCOPE_CLI" ]]; then
            MODELSCOPE_CLI=$(command -v modelscope || true)
        fi
        [[ -n "$MODELSCOPE_CLI" && -x "$MODELSCOPE_CLI" ]] || die "modelscope CLI not found"
    fi
}

component_source() {
    local repo=$1
    if [[ "$SOURCE" != "auto" ]]; then
        if [[ "$SOURCE" == "modelscope-first" ]]; then
            if [[ -n "${HF_ONLY_REPOS[$repo]:-}" ]]; then
                printf 'huggingface'
            else
                printf 'modelscope'
            fi
            return 0
        fi
        if [[ "$SOURCE" == "huggingface-first" ]]; then
            if [[ -n "${MODELSCOPE_ONLY_REPOS[$repo]:-}" ]]; then
                printf 'modelscope'
            else
                printf 'huggingface'
            fi
            return 0
        fi
        printf '%s' "$SOURCE"
        return 0
    fi
    if [[ -n "${MODELSCOPE_ONLY_REPOS[$repo]:-}" ]]; then
        printf 'modelscope'
    else
        printf 'huggingface'
    fi
}

component_sources() {
    local repo=$1
    if [[ "$SOURCE" == "modelscope-first" ]]; then
        if [[ -n "${HF_ONLY_REPOS[$repo]:-}" ]]; then
            printf 'huggingface\n'
        elif [[ -n "${MODELSCOPE_ONLY_REPOS[$repo]:-}" ]]; then
            printf 'modelscope\n'
        else
            printf 'modelscope\n'
            printf 'huggingface\n'
        fi
        return 0
    fi
    if [[ "$SOURCE" == "huggingface-first" ]]; then
        if [[ -n "${MODELSCOPE_ONLY_REPOS[$repo]:-}" ]]; then
            printf 'modelscope\n'
        else
            printf 'huggingface\n'
            if [[ -z "${HF_ONLY_REPOS[$repo]:-}" ]]; then
                printf 'modelscope\n'
            fi
        fi
        return 0
    fi
    printf '%s\n' "$(component_source "$repo")"
}

join_sources() {
    local IFS=,
    printf '%s' "$*"
}

repo_dir_name() {
    local repo=$1
    printf '%s' "${repo##*/}"
}

repo_dest() {
    local repo=$1
    printf '%s/%s' "$MODEL_ROOT" "$(repo_dir_name "$repo")"
}

pattern_exists_in_dir() {
    local dest=$1
    local pattern=$2
    local directory_path
    local -a matches
    local -a file_matches=()
    local match

    [[ -e "$dest" ]] || return 1
    if [[ -z "$pattern" || "$pattern" == "*" ]]; then
        directory_has_files "$dest"
        return $?
    fi

    if [[ "$pattern" == */ ]]; then
        directory_path="$dest/${pattern%/}"
        [[ -d "$directory_path" ]] || return 1
        directory_has_files "$directory_path"
        return $?
    fi

    shopt -s nullglob
    matches=("$dest"/$pattern)
    shopt -u nullglob
    for match in "${matches[@]}"; do
        [[ -f "$match" && -s "$match" ]] && file_matches+=("$match")
    done
    [[ ${#file_matches[@]} -gt 0 ]] || return 1
    shard_matches_complete "${file_matches[@]}"
}

directory_has_files() {
    local directory_path=$1
    [[ -n "$(find "$directory_path" -type f -print -quit)" ]]
}

shard_matches_complete() {
    local path name shard_idx shard_total variant group_key total key shard_key
    local saw_shards=0
    local -A present=()
    local -A totals=()

    for path in "$@"; do
        name=${path##*/}
        if [[ "$name" =~ ^(.*)-([0-9]{5})-of-([0-9]{5})([^/]*)\.safetensors$ ]]; then
            saw_shards=1
            shard_idx=$((10#${BASH_REMATCH[2]}))
            shard_total=$((10#${BASH_REMATCH[3]}))
            variant=${BASH_REMATCH[4]}
            group_key="${BASH_REMATCH[1]}|${variant}"
            totals[$group_key]=$shard_total
            present["$group_key:$shard_idx"]=1
        fi
    done

    [[ $saw_shards -eq 1 ]] || return 0
    for key in "${!totals[@]}"; do
        total=${totals[$key]}
        for ((shard_idx = 1; shard_idx <= total; shard_idx++)); do
            shard_key="${key}:${shard_idx}"
            [[ -n "${present[$shard_key]:-}" ]] || continue 2
        done
        return 0
    done
    return 1
}

pattern_exists_in_root() {
    local root=$1
    local repo=$2
    local pattern=$3
    repo_source_in_root "$root" "$repo" "$pattern" >/dev/null
}

repo_source_in_root() {
    local root=$1
    local repo=$2
    local pattern=$3
    local flat_dest="${root}/$(repo_dir_name "$repo")"
    local repo_dest="${root}/${repo}"
    local leaf
    local path
    local -a candidates=()

    leaf=$(repo_dir_name "$repo")
    candidates=("$flat_dest" "$repo_dest")
    shopt -s nullglob
    candidates+=("$root"/*/"$leaf" "$root"/*/*/"$leaf")
    shopt -u nullglob

    for path in "${candidates[@]}"; do
        if pattern_exists_in_dir "$path" "$pattern"; then
            printf '%s' "$path"
            return 0
        fi
    done
    return 1
}

component_exists() {
    pattern_exists_in_root "$MODEL_ROOT" "$1" "$2"
}

legacy_wan_common_path_in_root() {
    local root=$1
    local repo=$2
    local pattern=$3
    local -a candidates=()
    local path

    [[ "$repo" == "DiffSynth-Studio/Wan-Series-Converted-Safetensors" ]] || return 1
    case "$pattern" in
        models_t5_umt5-xxl-enc-bf16.safetensors)
            candidates=(
                "$root/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
                "$root/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"
                "$root/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
                "$root/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"
            )
            ;;
        Wan2.1_VAE.safetensors)
            candidates=(
                "$root/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
                "$root/Wan2.1-T2V-14B/Wan2.1_VAE.pth"
                "$root/Wan2.2-T2V-A14B/Wan2.1_VAE.pth"
                "$root/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
                "$root/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth"
                "$root/Wan-AI/Wan2.2-T2V-A14B/Wan2.1_VAE.pth"
            )
            ;;
        Wan2.2_VAE.safetensors)
            candidates=(
                "$root/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
                "$root/Wan2.2-T2V-A14B/Wan2.2_VAE.pth"
                "$root/Wan2.2-I2V-A14B/Wan2.2_VAE.pth"
                "$root/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
                "$root/Wan-AI/Wan2.2-T2V-A14B/Wan2.2_VAE.pth"
                "$root/Wan-AI/Wan2.2-I2V-A14B/Wan2.2_VAE.pth"
            )
            ;;
        models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors)
            candidates=(
                "$root/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
                "$root/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
                "$root/Wan2.1-FLF2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
                "$root/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
                "$root/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
                "$root/Wan-AI/Wan2.1-FLF2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
            )
            ;;
        *)
            return 1
            ;;
    esac

    for path in "${candidates[@]}"; do
        if [[ -s "$path" ]]; then
            printf '%s' "$path"
            return 0
        fi
    done
    return 1
}

legacy_wan_common_path() {
    local path
    if path=$(legacy_wan_common_path_in_root "$MODEL_ROOT" "$1" "$2"); then
        printf '%s' "$path"
        return 0
    fi
    if [[ -n "$LINK_ROOT" ]] && path=$(legacy_wan_common_path_in_root "$LINK_ROOT" "$1" "$2"); then
        printf '%s' "$path"
        return 0
    fi
    return 1
}

link_existing_repo() {
    local repo=$1
    local pattern=$2
    local src dest

    [[ -n "$LINK_ROOT" ]] || return 1
    src=$(repo_source_in_root "$LINK_ROOT" "$repo" "$pattern") || return 1
    dest=$(repo_dest "$repo")

    if [[ -L "$dest" ]]; then
        if [[ "$(readlink -f "$dest")" == "$(readlink -f "$src")" ]]; then
            return 0
        fi
        echo "  [ERROR] existing symlink points elsewhere: $dest -> $(readlink "$dest")" >&2
        return 2
    fi
    if [[ -e "$dest" ]]; then
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    ln -s "$src" "$dest"
    echo "  linked ${repo} -> ${src}"
}

download_component() {
    local repo=$1
    local pattern=$2
    local dest
    local display=${pattern:-*}
    local include_pattern
    local selected_source selected_sources_display source_index
    local legacy_path
    local link_source
    local attempt delay status
    local -a cmd
    local -a selected_sources

    dest=$(repo_dest "$repo")
    include_pattern=$(include_pattern_for_cli "$pattern")
    mapfile -t selected_sources < <(component_sources "$repo")
    selected_sources_display=$(join_sources "${selected_sources[@]}")

    if [[ $DRY_RUN -eq 1 ]]; then
        if component_exists "$repo" "$pattern"; then
            echo "  would skip existing complete ${repo}:${display}"
            return 0
        fi
        if legacy_path=$(legacy_wan_common_path "$repo" "$pattern"); then
            echo "  would skip ${repo}:${display}; compatible local Wan common file exists: ${legacy_path}"
            return 0
        fi
        if [[ -n "$LINK_ROOT" ]] && link_source=$(repo_source_in_root "$LINK_ROOT" "$repo" "$pattern"); then
            echo "  would link/reuse ${repo}:${display} from ${link_source} -> ${dest}"
            return 0
        fi
        echo "  would download ${repo}:${display} via ${selected_sources_display} -> ${dest}"
        return 0
    fi

    if link_existing_repo "$repo" "$pattern"; then
        if component_exists "$repo" "$pattern"; then
            echo "  reuse linked ${repo}:${display}"
            return 0
        fi
        if [[ -L "$dest" ]]; then
            echo "  [ERROR] linked repo is missing required pattern, refusing to download into external source: ${repo}:${display}" >&2
            return 1
        fi
    fi
    if component_exists "$repo" "$pattern"; then
        echo "  reuse existing ${repo}:${display}"
        return 0
    fi
    if legacy_path=$(legacy_wan_common_path "$repo" "$pattern"); then
        echo "  skip ${repo}:${display}; compatible local Wan common file exists: ${legacy_path}"
        return 0
    fi

    mkdir -p "$dest"
    for source_index in "${!selected_sources[@]}"; do
        selected_source=${selected_sources[$source_index]}
        for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
            echo "  download ${repo}:${display} via ${selected_source} (attempt ${attempt}/${MAX_ATTEMPTS})"
            if [[ "$selected_source" == "huggingface" ]]; then
                cmd=("$HF_CLI" download "$repo" --repo-type model --local-dir "$dest" --no-force-download --max-workers "$MAX_WORKERS")
            else
                cmd=("$MODELSCOPE_CLI" download "$repo" --repo-type model --local_dir "$dest" --max-workers "$MAX_WORKERS")
            fi
            if [[ -n "$include_pattern" ]]; then
                cmd+=(--include "$include_pattern")
            fi

            if "${cmd[@]}"; then
                if component_exists "$repo" "$pattern"; then
                    return 0
                fi
                echo "  download command completed but required pattern is still missing or incomplete: ${repo}:${display}" >&2
                status=1
            else
                status=$?
            fi
            if [[ $attempt -eq $MAX_ATTEMPTS ]]; then
                break
            fi
            delay=$((BASE_DELAY * (2 ** (attempt - 1))))
            echo "  retry after ${delay}s" >&2
            sleep "$delay"
        done
        if (( source_index + 1 < ${#selected_sources[@]} )); then
            echo "  fallback after ${selected_source} failure: ${repo}:${display}" >&2
        else
            echo "  [ERROR] failed ${repo}:${display}" >&2
            return "$status"
        fi
    done
}

skip_catalog_component() {
    local key=$1
    local repo=$2
    local pattern=$3
    local display=${pattern:-*}

    if [[ "$key" == "ltx23" && "$repo" == "DiffSynth-Studio/LTX-2.3-Repackage" ]]; then
        echo "  skip ${repo}:${display}; ltx23 loader uses Lightricks/LTX-2.3 source checkpoint to avoid duplicate large-model loading"
        return 0
    fi
    return 1
}

verify_bundle_complete() {
    local key=$1

    [[ $DRY_RUN -eq 0 ]] || return 0
    MODEL_ROOT="$MODEL_ROOT" VERIFY_KEY="$key" PYTHONPATH="$SCRIPTS_ROOT" "$PYTHON" - <<'PY'
import os
import sys

from _infer_diffsynth.models import resolve_diffsynth_model_paths

key = os.environ["VERIFY_KEY"]
model_root = os.environ["MODEL_ROOT"]
resolved = resolve_diffsynth_model_paths(key, model_root=model_root)
if resolved.complete:
    print(f"  verified complete DiffSynth bundle: {key}")
    sys.exit(0)

print(f"  [ERROR] DiffSynth bundle is still incomplete after downloads: {key}", file=sys.stderr)
for item in resolved.missing:
    print(f"    - {item}", file=sys.stderr)
sys.exit(1)
PY
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            --list)
                LIST=1
                shift
                ;;
            --all)
                ALL=1
                shift
                ;;
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            --tier)
                [[ $# -ge 2 && "$2" != -* ]] || die "--tier requires one of: 1, 2, 3"
                case "$2" in
                    1|2|3) TIER_FILTER=$2 ;;
                    *) die "--tier must be one of: 1, 2, 3" ;;
                esac
                shift 2
                ;;
            --kind)
                [[ $# -ge 2 && "$2" != -* ]] || die "--kind requires one of: wan, mova, ltx2"
                KIND_FILTER=$2
                shift 2
                ;;
            --model-root)
                [[ $# -ge 2 && "$2" != -* ]] || die "--model-root requires a path"
                MODEL_ROOT=$2
                shift 2
                ;;
            --source)
                [[ $# -ge 2 && "$2" != -* ]] || die "--source requires auto, modelscope-first, huggingface-first, hf-first, huggingface, or modelscope"
                case "$2" in
                    auto|modelscope-first|huggingface-first|huggingface|modelscope) SOURCE=$2 ;;
                    hf-first) SOURCE=huggingface-first ;;
                    *) die "--source must be auto, modelscope-first, huggingface-first, hf-first, huggingface, or modelscope" ;;
                esac
                shift 2
                ;;
            --hf-endpoint)
                [[ $# -ge 2 && "$2" != -* ]] || die "--hf-endpoint requires a URL, for example https://hf-mirror.com"
                HF_ENDPOINT=$2
                shift 2
                ;;
            --link-root)
                [[ $# -ge 2 && "$2" != -* ]] || die "--link-root requires a path"
                LINK_ROOT=$2
                shift 2
                ;;
            --proxy)
                [[ $# -ge 2 && "$2" != -* ]] || die "--proxy requires a URL"
                PROXY=$2
                SKIP_PROXY=0
                shift 2
                ;;
            --no-proxy)
                SKIP_PROXY=1
                shift
                ;;
            --max-workers)
                [[ $# -ge 2 && "$2" != -* ]] || die "--max-workers requires a positive integer"
                [[ "$2" =~ ^[0-9]+$ && "$2" -gt 0 ]] || die "--max-workers requires a positive integer"
                MAX_WORKERS=$2
                shift 2
                ;;
            -*)
                die "unknown option: $1"
                ;;
            *)
                EXPLICIT_KEYS+=("$1")
                shift
                ;;
        esac
    done
}

parse_args "$@"

if [[ $LIST -eq 1 ]]; then
    print_catalog
    exit 0
fi

if [[ $ALL -eq 1 && ( ${#EXPLICIT_KEYS[@]} -gt 0 || -n "$TIER_FILTER" || -n "$KIND_FILTER" ) ]]; then
    die "--all cannot be combined with --tier, --kind, or explicit keys"
fi

declare -A EXPLICIT_MAP=()
for key in "${EXPLICIT_KEYS[@]}"; do
    if ! is_known_key "$key"; then
        echo "[ERROR] unknown key: $key" >&2
        print_available_keys
        exit 1
    fi
    EXPLICIT_MAP[$key]=1
done

SELECTED_ENTRIES=()
for entry in "${CATALOGUE[@]}"; do
    IFS='|' read -r key tier pipeline_kind description components <<< "$entry"
    if [[ $ALL -eq 1 ||
          -n "${EXPLICIT_MAP[$key]:-}" ||
          ( -n "$TIER_FILTER" && "$tier" == "$TIER_FILTER" ) ||
          ( -n "$KIND_FILTER" && "$pipeline_kind" == "$KIND_FILTER" ) ]]; then
        SELECTED_ENTRIES+=("$entry")
    fi
done

if [[ ${#SELECTED_ENTRIES[@]} -eq 0 ]]; then
    die "refusing to download by default; pass --all, --tier N, --kind NAME, or explicit keys"
fi

if [[ $SKIP_PROXY -eq 0 ]]; then
    configure_proxy
else
    clear_proxy
fi
if [[ "$SOURCE" != "modelscope" && -n "$HF_ENDPOINT" ]]; then
    export HF_ENDPOINT
fi
if [[ $DRY_RUN -eq 0 ]]; then
    resolve_tools
fi

export DIFFSYNTH_MODEL_BASE_PATH=$MODEL_ROOT
export DIFFSYNTH_DOWNLOAD_SOURCE=$SOURCE

echo "DiffSynth model root: $MODEL_ROOT"
echo "Download source: $SOURCE"
if [[ "$SOURCE" != "modelscope" && -n "$HF_ENDPOINT" ]]; then
    echo "HF endpoint: $HF_ENDPOINT"
fi
if [[ $SKIP_PROXY -eq 0 && -n "$PROXY" ]]; then
    echo "Proxy: $PROXY"
else
    echo "Proxy: disabled"
fi
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run: no files will be downloaded."
fi

declare -A SEEN_COMPONENTS=()
for entry in "${SELECTED_ENTRIES[@]}"; do
    IFS='|' read -r key tier pipeline_kind description components <<< "$entry"
    print_bundle "$entry"
    while IFS= read -r component; do
        repo=${component%%::*}
        pattern=
        [[ "$component" == *"::"* ]] && pattern=${component#*::}
        if skip_catalog_component "$key" "$repo" "$pattern"; then
            continue
        fi
        [[ -n "${SEEN_COMPONENTS[$component]:-}" ]] && continue
        SEEN_COMPONENTS[$component]=1
        download_component "$repo" "$pattern"
    done < <(split_components "$components")
    verify_bundle_complete "$key"
done
