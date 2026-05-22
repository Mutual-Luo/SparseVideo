from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_MODEL_ROOT = Path(
    os.environ.get("SPARSEVIDEO_DIFFSYNTH_MODEL_ROOT", "/home/dataset-assist-0/luojy/models")
)


@dataclass(frozen=True)
class DiffSynthModelSpec:
    key: str
    aliases: Tuple[str, ...]
    family: str
    pipeline: str
    description: str
    default_height: int
    default_width: int
    default_num_frames: int
    default_fps: int
    default_cfg_scale: float = 5.0
    default_sigma_shift: float = 5.0
    default_switch_dit_boundary: float = 0.875
    required_inputs: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedDiffSynthModel:
    spec: DiffSynthModelSpec
    model_root: Path
    components: Mapping[str, Tuple[Path, ...]]
    missing: Tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.spec.key,
            "family": self.spec.family,
            "pipeline": self.spec.pipeline,
            "description": self.spec.description,
            "required_inputs": list(self.spec.required_inputs),
            "model_root": str(self.model_root),
            "complete": self.complete,
            "components": {
                name: [str(path) for path in paths]
                for name, paths in sorted(self.components.items())
            },
            "missing": list(self.missing),
        }


@dataclass(frozen=True)
class DeferredDiffSynthModelSpec:
    key: str
    aliases: Tuple[str, ...]
    family: str
    pipeline: str
    description: str
    origin_repo: str
    origin_pattern: str
    required_inputs: Tuple[str, ...]
    deferred_reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.key,
            "aliases": list(self.aliases),
            "family": self.family,
            "pipeline": self.pipeline,
            "description": self.description,
            "origin_repo": self.origin_repo,
            "origin_pattern": self.origin_pattern,
            "required_inputs": list(self.required_inputs),
            "deferred_reason": self.deferred_reason,
        }


_SPECS: Tuple[DiffSynthModelSpec, ...] = (
    DiffSynthModelSpec(
        key="wan21-t2v-1.3b",
        aliases=("wan1.3b", "wan21-1.3b"),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.1 text-to-video 1.3B",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
    ),
    DiffSynthModelSpec(
        key="wan21-speedcontrol-1.3b",
        aliases=("wan21-speedcontrol", "speedcontrol"),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.1 1.3B speed-control motion controller",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
    ),
    DiffSynthModelSpec(
        key="wan21-t2v-14b",
        aliases=("wan14b", "wan21-14b"),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.1 text-to-video 14B",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
    ),
    DiffSynthModelSpec(
        key="wan21-i2v-14b-480p",
        aliases=("wan21-i2v-480p", "wan-i2v-480p"),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.1 image-to-video 14B 480P",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
        required_inputs=("input_image",),
    ),
    DiffSynthModelSpec(
        key="wan21-i2v-14b-720p",
        aliases=("wan21-i2v-14b", "wan21-i2v", "wan-i2v", "wan14b-i2v"),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.1 image-to-video 14B 720P",
        default_height=720,
        default_width=1280,
        default_num_frames=81,
        default_fps=15,
        required_inputs=("input_image",),
    ),
    DiffSynthModelSpec(
        key="wan21-flf2v-14b-720p",
        aliases=("wan21-flf2v", "wan-flf2v"),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.1 first-last-frame-to-video 14B 720P",
        default_height=720,
        default_width=1280,
        default_num_frames=81,
        default_fps=15,
        required_inputs=("input_image", "end_image"),
    ),
    DiffSynthModelSpec(
        key="wan21-fun-1.3b-control",
        aliases=("wan21-fun-control-1.3b",),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.1-Fun 1.3B Control",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
        required_inputs=("control_video",),
    ),
    DiffSynthModelSpec(
        key="wan21-fun-1.3b-inp",
        aliases=("wan21-fun-inp-1.3b",),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.1-Fun 1.3B InP",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
        required_inputs=("input_image", "end_image"),
    ),
    DiffSynthModelSpec(
        key="wan21-fun-14b-control",
        aliases=("wan21-fun-control-14b",),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.1-Fun 14B Control",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
        required_inputs=("control_video",),
    ),
    DiffSynthModelSpec(
        key="wan21-fun-14b-inp",
        aliases=("wan21-fun-inp-14b",),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.1-Fun 14B InP",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
        required_inputs=("input_image", "end_image"),
    ),
    DiffSynthModelSpec(
        key="wan21-fun-v11-1.3b-control",
        aliases=("wan21-fun-v1.1-1.3b-control",),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.1-Fun V1.1 1.3B Control",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
        required_inputs=("control_video", "reference_image"),
    ),
    DiffSynthModelSpec(
        key="wan21-fun-v11-1.3b-control-camera",
        aliases=("wan21-fun-v1.1-1.3b-control-camera",),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.1-Fun V1.1 1.3B Control-Camera",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
        required_inputs=("input_image", "camera_control_direction"),
    ),
    DiffSynthModelSpec(
        key="wan21-fun-v11-14b-control",
        aliases=("wan21-fun-v1.1-14b-control",),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.1-Fun V1.1 14B Control",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
        required_inputs=("control_video", "reference_image"),
    ),
    DiffSynthModelSpec(
        key="wan21-fun-v11-14b-control-camera",
        aliases=("wan21-fun-v1.1-14b-control-camera",),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.1-Fun V1.1 14B Control-Camera",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
        required_inputs=("input_image", "camera_control_direction"),
    ),
    DiffSynthModelSpec(
        key="wan21-vace-1.3b",
        aliases=("vace", "wan-vace", "wan21-vace"),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth VACE Wan2.1 1.3B preview",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
    ),
    DiffSynthModelSpec(
        key="wan21-vace-14b",
        aliases=(),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth VACE Wan2.1 14B",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
    ),
    DiffSynthModelSpec(
        key="wan22-animate-14b",
        aliases=("wananimate", "wan-animate", "wan22-animate"),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.2 Animate 14B",
        default_height=480,
        default_width=832,
        default_num_frames=77,
        default_fps=16,
        default_cfg_scale=1.0,
    ),
    DiffSynthModelSpec(
        key="wan22-t2v-a14b",
        aliases=("wan22", "wan22-a14b"),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.2 text-to-video A14B high/low-noise models",
        default_height=704,
        default_width=1248,
        default_num_frames=121,
        default_fps=15,
    ),
    DiffSynthModelSpec(
        key="wan22-i2v-a14b",
        aliases=("wan22-i2v",),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.2 image-to-video A14B high/low-noise models",
        default_height=704,
        default_width=1248,
        default_num_frames=121,
        default_fps=15,
        required_inputs=("input_image",),
    ),
    DiffSynthModelSpec(
        key="wan22-ti2v-5b",
        aliases=("wan22-ti2v",),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.2 text/image-to-video 5B",
        default_height=704,
        default_width=1248,
        default_num_frames=121,
        default_fps=15,
    ),
    DiffSynthModelSpec(
        key="wan22-s2v-14b",
        aliases=("wan22-s2v",),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.2 speech-to-video 14B",
        default_height=704,
        default_width=1248,
        default_num_frames=121,
        default_fps=16,
        required_inputs=("input_audio",),
    ),
    DiffSynthModelSpec(
        key="wan22-fun-a14b-control",
        aliases=("wan22-fun-control-a14b",),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.2-Fun A14B Control high-noise DiT",
        default_height=704,
        default_width=1248,
        default_num_frames=121,
        default_fps=15,
        required_inputs=("control_video", "reference_image"),
    ),
    DiffSynthModelSpec(
        key="wan22-fun-a14b-control-camera",
        aliases=("wan22-fun-control-camera-a14b",),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.2-Fun A14B Control-Camera high-noise DiT",
        default_height=704,
        default_width=1248,
        default_num_frames=121,
        default_fps=15,
        required_inputs=("input_image", "camera_control_direction"),
    ),
    DiffSynthModelSpec(
        key="longcat-video",
        aliases=("longcat",),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth LongCat-Video on WanVideoPipeline",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
        required_inputs=("longcat_video",),
    ),
    DiffSynthModelSpec(
        key="video-as-prompt-wan21-14b",
        aliases=("video-as-prompt", "vap-wan21-14b"),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Video-as-Prompt Wan2.1 14B",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
        required_inputs=("vap_video",),
    ),
    DiffSynthModelSpec(
        key="krea-realtime-video",
        aliases=("krea-video",),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Krea realtime video 14B",
        default_height=480,
        default_width=832,
        default_num_frames=81,
        default_fps=15,
    ),
    DiffSynthModelSpec(
        key="mova-720p",
        aliases=("mova",),
        family="mova",
        pipeline="MovaAudioVideoPipeline",
        description="DiffSynth MOVA 720P with Wan video DiT backbone",
        default_height=720,
        default_width=1280,
        default_num_frames=81,
        default_fps=24,
        default_switch_dit_boundary=0.9,
    ),
    DiffSynthModelSpec(
        key="ltx2",
        aliases=("ltx-2",),
        family="ltx2",
        pipeline="LTX2AudioVideoPipeline",
        description="DiffSynth LTX-2 repackaged audio-video components",
        default_height=512,
        default_width=768,
        default_num_frames=121,
        default_fps=24,
        default_cfg_scale=3.0,
    ),
    DiffSynthModelSpec(
        key="ltx23",
        aliases=("ltx2.3", "ltx-2.3"),
        family="ltx2",
        pipeline="LTX2AudioVideoPipeline",
        description="DiffSynth LTX-2.3 source checkpoint plus latent upsampler components",
        default_height=512,
        default_width=768,
        default_num_frames=121,
        default_fps=24,
        default_cfg_scale=3.0,
    ),
)

_DEFERRED_SPECS: Tuple[DeferredDiffSynthModelSpec, ...] = (
    DeferredDiffSynthModelSpec(
        key="wan22-dancer-14b",
        aliases=("wan22-dancer", "wantodance"),
        family="wan",
        pipeline="WanVideoPipeline",
        description="DiffSynth Wan2.2-Dancer 14B WanToDance global model",
        origin_repo="Wan-AI/Wan2.2-Dancer-14B",
        origin_pattern="global_model.safetensors",
        required_inputs=(
            "wantodance_music_path",
            "wantodance_reference_image",
            "wantodance_fps",
            "wantodance_keyframes",
            "wantodance_keyframes_mask",
        ),
        deferred_reason=(
            "DiffSynth 2.0.12 model_configs.py references this checkpoint, but the "
            "repo is not safe to include in the resumable --all downloader until a "
            "stable Hugging Face or ModelScope source is confirmed. SparseVideo still "
            "reports WanToDance music-injector and music-encoder attention as "
            "unpatched if a local pipeline is loaded."
        ),
    ),
)

_SPEC_BY_NAME = {spec.key: spec for spec in _SPECS}
for _spec in _SPECS:
    for _alias in _spec.aliases:
        _SPEC_BY_NAME[_alias] = _spec

_SHARD_RE = re.compile(r"^(?P<prefix>.*)-(?P<idx>\d{5})-of-(?P<total>\d{5})(?P<variant>.*)\.safetensors$")

_WAN21_FUN_REPOS = {
    "wan21-fun-1.3b-control": "Wan2.1-Fun-1.3B-Control",
    "wan21-fun-1.3b-inp": "Wan2.1-Fun-1.3B-InP",
    "wan21-fun-14b-control": "Wan2.1-Fun-14B-Control",
    "wan21-fun-14b-inp": "Wan2.1-Fun-14B-InP",
    "wan21-fun-v11-1.3b-control": "Wan2.1-Fun-V1.1-1.3B-Control",
    "wan21-fun-v11-1.3b-control-camera": "Wan2.1-Fun-V1.1-1.3B-Control-Camera",
    "wan21-fun-v11-14b-control": "Wan2.1-Fun-V1.1-14B-Control",
    "wan21-fun-v11-14b-control-camera": "Wan2.1-Fun-V1.1-14B-Control-Camera",
}

_WAN21_FUN_IMAGE_REPOS = {
    "wan21-fun-1.3b-inp",
    "wan21-fun-14b-inp",
    "wan21-fun-v11-1.3b-control",
    "wan21-fun-v11-1.3b-control-camera",
    "wan21-fun-v11-14b-control",
    "wan21-fun-v11-14b-control-camera",
}

_WAN22_FUN_REPOS = {
    "wan22-fun-a14b-control": "Wan2.2-Fun-A14B-Control",
    "wan22-fun-a14b-control-camera": "Wan2.2-Fun-A14B-Control-Camera",
}


def list_diffsynth_model_specs() -> Tuple[DiffSynthModelSpec, ...]:
    return _SPECS


def list_deferred_diffsynth_model_specs() -> Tuple[DeferredDiffSynthModelSpec, ...]:
    return _DEFERRED_SPECS


def diffsynth_model_list_lines(*, include_deferred: bool = True) -> Tuple[str, ...]:
    lines = []
    for spec in _SPECS:
        aliases = f" aliases={','.join(spec.aliases)}" if spec.aliases else ""
        lines.append(f"{spec.key}: {spec.description}{aliases}")

    if include_deferred and _DEFERRED_SPECS:
        lines.append("Deferred/local-only DiffSynth models:")
        for spec in _DEFERRED_SPECS:
            aliases = f" aliases={','.join(spec.aliases)}" if spec.aliases else ""
            lines.append(
                f"{spec.key}: {spec.description}{aliases} "
                f"origin={spec.origin_repo}:{spec.origin_pattern}"
            )
    return tuple(lines)


def get_diffsynth_model_spec(model: str) -> DiffSynthModelSpec:
    try:
        return _SPEC_BY_NAME[model]
    except KeyError as exc:
        supported = ", ".join(spec.key for spec in _SPECS)
        deferred_names = [
            spec.key
            for spec in _DEFERRED_SPECS
            if model == spec.key or model in spec.aliases
        ]
        if deferred_names:
            deferred = deferred_names[0]
            raise ValueError(
                f"DiffSynth model '{model}' is currently deferred/local-only as '{deferred}'. "
                "It is documented in list_deferred_diffsynth_model_specs(), but is not "
                "included in the active downloader until its source is confirmed."
            ) from exc
        deferred = ", ".join(spec.key for spec in _DEFERRED_SPECS)
        suffix = f" Deferred/local-only models: {deferred}" if deferred else ""
        raise ValueError(f"Unsupported DiffSynth model '{model}'. Supported models: {supported}.{suffix}") from exc


def resolve_diffsynth_model_paths(
    model: str,
    model_root: Optional[os.PathLike[str] | str] = None,
) -> ResolvedDiffSynthModel:
    spec = get_diffsynth_model_spec(model)
    root = Path(model_root) if model_root is not None else DEFAULT_MODEL_ROOT
    root = root.expanduser()
    components: Dict[str, Tuple[Path, ...]] = {}
    missing: List[str] = []

    def add(
        name: str,
        candidates: Sequence[Tuple[str, str]],
        *,
        directory: bool = False,
        required_files: Sequence[str] = (),
    ) -> None:
        result, notes = _find_first_complete(
            root,
            candidates,
            directory=directory,
            required_files=required_files,
        )
        if result:
            components[name] = result
        else:
            detail = "; ".join(notes) if notes else ", ".join(f"{repo}:{pattern}" for repo, pattern in candidates)
            missing.append(f"{name}: {detail}")

    if spec.key == "wan21-t2v-1.3b":
        _add_wan_common_components(add, dit_repo="Wan2.1-T2V-1.3B", vae_version="21")
        add("dit", (("Wan2.1-T2V-1.3B", "diffusion_pytorch_model*.safetensors"),))
    elif spec.key == "wan21-speedcontrol-1.3b":
        _add_wan_common_components(add, dit_repo="Wan2.1-T2V-1.3B", vae_version="21")
        add("dit", (("Wan2.1-T2V-1.3B", "diffusion_pytorch_model*.safetensors"),))
        add("motion_controller", (("Wan2.1-1.3b-speedcontrol-v1", "model.safetensors"),))
    elif spec.key == "wan21-t2v-14b":
        _add_wan_common_components(add, dit_repo="Wan2.1-T2V-14B", vae_version="21")
        add("dit", (("Wan2.1-T2V-14B", "diffusion_pytorch_model*.safetensors"),))
    elif spec.key == "wan21-i2v-14b-480p":
        _add_wan_image_components(add, dit_repo="Wan2.1-I2V-14B-480P", vae_version="21")
        add("dit", (("Wan2.1-I2V-14B-480P", "diffusion_pytorch_model*.safetensors"),))
    elif spec.key == "wan21-i2v-14b-720p":
        _add_wan_image_components(add, dit_repo="Wan2.1-I2V-14B-720P", vae_version="21")
        add("dit", (("Wan2.1-I2V-14B-720P", "diffusion_pytorch_model*.safetensors"),))
    elif spec.key == "wan21-flf2v-14b-720p":
        _add_wan_image_components(add, dit_repo="Wan2.1-FLF2V-14B-720P", vae_version="21")
        add("dit", (("Wan2.1-FLF2V-14B-720P", "diffusion_pytorch_model*.safetensors"),))
    elif spec.key in _WAN21_FUN_REPOS:
        repo = _WAN21_FUN_REPOS[spec.key]
        if spec.key in _WAN21_FUN_IMAGE_REPOS:
            _add_wan_image_components(add, dit_repo=repo, vae_version="21")
        else:
            _add_wan_common_components(add, dit_repo=repo, vae_version="21")
        add("dit", ((repo, "diffusion_pytorch_model*.safetensors"),))
    elif spec.key == "wan21-vace-1.3b":
        _add_wan_common_components(add, dit_repo="VACE-Wan2.1-1.3B-Preview", vae_version="21")
        vace_paths = (("VACE-Wan2.1-1.3B-Preview", "diffusion_pytorch_model*.safetensors"),)
        add("dit", vace_paths)
        add("vace", vace_paths)
    elif spec.key == "wan21-vace-14b":
        _add_wan_common_components(add, dit_repo="Wan2.1-VACE-14B", vae_version="21")
        vace_paths = (("Wan2.1-VACE-14B", "diffusion_pytorch_model*.safetensors"),)
        add("dit", vace_paths)
        add("vace", vace_paths)
    elif spec.key == "wan22-animate-14b":
        _add_wan_image_components(add, dit_repo="Wan2.2-Animate-14B", vae_version="22")
        animate_paths = (("Wan2.2-Animate-14B", "diffusion_pytorch_model*.safetensors"),)
        add("dit", animate_paths)
        add("animate_adapter", animate_paths)
    elif spec.key == "wan22-t2v-a14b":
        _add_wan_common_components(add, dit_repo="Wan2.2-T2V-A14B", vae_version="21")
        add("dit_high_noise", (("Wan2.2-T2V-A14B", "high_noise_model/diffusion_pytorch_model*.safetensors"),))
        add("dit_low_noise", (("Wan2.2-T2V-A14B", "low_noise_model/diffusion_pytorch_model*.safetensors"),))
    elif spec.key == "wan22-i2v-a14b":
        _add_wan_image_components(add, dit_repo="Wan2.2-I2V-A14B", vae_version="22")
        add("dit_high_noise", (("Wan2.2-I2V-A14B", "high_noise_model/diffusion_pytorch_model*.safetensors"),))
        add("dit_low_noise", (("Wan2.2-I2V-A14B", "low_noise_model/diffusion_pytorch_model*.safetensors"),))
    elif spec.key == "wan22-ti2v-5b":
        _add_wan_common_components(add, dit_repo="Wan2.2-TI2V-5B", vae_version="22")
        add("dit", (("Wan2.2-TI2V-5B", "diffusion_pytorch_model*.safetensors"),))
    elif spec.key == "wan22-s2v-14b":
        _add_wan_image_components(add, dit_repo="Wan2.2-S2V-14B", vae_version="22")
        add("dit", (("Wan2.2-S2V-14B", "diffusion_pytorch_model*.safetensors"),))
        add("audio_encoder", (("Wan2.2-S2V-14B", "wav2vec2-large-xlsr-53-english/model.safetensors"),))
        add(
            "audio_processor",
            (("Wan2.2-S2V-14B", "wav2vec2-large-xlsr-53-english"),),
            directory=True,
            required_files=("preprocessor_config.json", "vocab.json"),
        )
    elif spec.key in _WAN22_FUN_REPOS:
        repo = _WAN22_FUN_REPOS[spec.key]
        _add_wan_common_components(add, dit_repo=repo, vae_version="21")
        add("dit", ((repo, "high_noise_model/diffusion_pytorch_model*.safetensors"),))
    elif spec.key == "longcat-video":
        _add_wan_common_components(add, dit_repo="LongCat-Video", vae_version="21")
        add("dit", (("LongCat-Video", "dit/diffusion_pytorch_model*.safetensors"),))
    elif spec.key == "video-as-prompt-wan21-14b":
        _add_wan_image_components(add, dit_repo="Video-As-Prompt-Wan2.1-14B", vae_version="21")
        vap_paths = (("Video-As-Prompt-Wan2.1-14B", "transformer/diffusion_pytorch_model*.safetensors"),)
        add("dit", vap_paths)
        add("vap", vap_paths)
    elif spec.key == "krea-realtime-video":
        _add_wan_common_components(add, dit_repo="krea-realtime-video", vae_version="21")
        add("dit", (("krea-realtime-video", "krea-realtime-video-14b.safetensors"),))
    elif spec.key == "mova-720p":
        _add_wan_weight_components(add, dit_repo="Wan2.1-T2V-14B", vae_version="21")
        add("dit", (("Wan2.1-T2V-14B", "diffusion_pytorch_model*.safetensors"),))
        add(
            "mova_tokenizer",
            (("MOVA-720p", "tokenizer"),),
            directory=True,
            required_files=("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"),
        )
        add("audio_dit", (("MOVA-720p", "audio_dit/diffusion_pytorch_model*.safetensors"),))
        add("audio_vae", (("MOVA-720p", "audio_vae/diffusion_pytorch_model*.safetensors"),))
        add("dual_tower_bridge", (("MOVA-720p", "dual_tower_bridge/diffusion_pytorch_model*.safetensors"),))
    elif spec.key == "ltx2":
        _add_ltx2_components(add, repo="LTX-2-Repackage")
        add("audio_vae_encoder", (("LTX-2-Repackage", "audio_vae_encoder.safetensors"),))
        add("audio_vae_decoder", (("LTX-2-Repackage", "audio_vae_decoder.safetensors"),))
    elif spec.key == "ltx23":
        _add_ltx2_text_components(add)
        add("source_checkpoint", (("LTX-2.3", "ltx-2.3-22b-dev.safetensors"),))
        add("latent_upsampler", (("LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),))
    else:
        raise AssertionError(f"Unhandled DiffSynth model spec: {spec.key}")

    return ResolvedDiffSynthModel(
        spec=spec,
        model_root=root,
        components=components,
        missing=tuple(missing),
    )


def load_diffsynth_pipeline(
    model: str,
    *,
    model_root: Optional[os.PathLike[str] | str] = None,
    torch_dtype: Any = None,
    device: str = "cuda",
    offload_device: Optional[str] = "cpu",
    vram_limit: Optional[float] = None,
    use_usp: bool = False,
    enable_vram_management: bool = True,
):
    resolved = resolve_diffsynth_model_paths(model, model_root=model_root)
    if not resolved.complete:
        missing = "\n  - ".join(resolved.missing)
        raise FileNotFoundError(
            f"DiffSynth model '{resolved.spec.key}' is incomplete under {resolved.model_root}.\n"
            f"  - {missing}\n"
            "Run scripts/download_diffsynth_models.sh for the missing native DiffSynth files."
        )

    if torch_dtype is None:
        import torch

        torch_dtype = torch.bfloat16

    if resolved.spec.family == "wan":
        from diffsynth import ModelConfig
        from diffsynth.pipelines.wan_video import WanVideoPipeline

        audio_processor_path = (
            _single_path(resolved, "audio_processor")
            if "audio_processor" in resolved.components
            else None
        )
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=_model_configs_from_components(
                resolved,
                _wan_model_config_order(resolved),
                ModelConfig,
                offload_device=offload_device,
            ),
            tokenizer_config=ModelConfig(path=str(_single_path(resolved, "tokenizer"))),
            audio_processor_config=None,
            redirect_common_files=False,
            use_usp=use_usp,
            vram_limit=vram_limit,
        )
        if audio_processor_path is not None:
            pipe.audio_processor = _load_wav2vec2_processor(audio_processor_path)
    elif resolved.spec.family == "mova":
        from diffsynth import ModelConfig
        from diffsynth.pipelines.mova_audio_video import MovaAudioVideoPipeline

        pipe = MovaAudioVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=_model_configs_from_components(
                resolved,
                ("text_encoder", "dit", "vae", "audio_dit", "audio_vae", "dual_tower_bridge"),
                ModelConfig,
                offload_device=offload_device,
            ),
            tokenizer_config=ModelConfig(path=str(_single_path(resolved, "mova_tokenizer"))),
            use_usp=use_usp,
            vram_limit=vram_limit,
        )
    elif resolved.spec.family == "ltx2":
        if use_usp:
            raise NotImplementedError("DiffSynth LTX2AudioVideoPipeline does not expose SparseVideo USP support.")
        from diffsynth import ModelConfig
        from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline

        pipe = LTX2AudioVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=_model_configs_from_components(
                resolved,
                _ltx2_model_config_order(resolved),
                ModelConfig,
                offload_device=offload_device,
            ),
            tokenizer_config=ModelConfig(path=str(_single_path(resolved, "tokenizer"))),
            vram_limit=vram_limit,
        )
    else:
        raise ValueError(f"Unsupported DiffSynth pipeline family: {resolved.spec.family}")

    if enable_vram_management and hasattr(pipe, "enable_vram_management"):
        pipe.enable_vram_management()
    pipe._sparsevideo_model_key = resolved.spec.key
    return pipe, resolved


def save_diffsynth_output(
    output: Any,
    output_file: os.PathLike[str] | str,
    *,
    fps: int,
    quality: int = 5,
    audio_sample_rate: int = 48000,
) -> Dict[str, Any]:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video, audio = split_diffsynth_output(output)
    audio_path = output_path.with_suffix(".wav") if audio is not None else None

    _unlink_existing_output(output_path)
    if audio_path is not None:
        _unlink_existing_output(audio_path)

    save_video = _diffsynth_save_video()
    save_video(video, str(output_path), fps=fps, quality=quality)
    if not output_path.is_file():
        raise RuntimeError(f"DiffSynth video export did not create output file: {output_path}")

    metadata: Dict[str, Any] = {
        "output_file": str(output_path),
        "output_type": "video_audio" if audio is not None else "video",
        "num_output_frames": len(video) if hasattr(video, "__len__") else None,
    }
    metadata["output_size"] = output_path.stat().st_size

    if audio is not None:
        assert audio_path is not None
        metadata["audio_file"] = str(audio_path)
        metadata["audio_sample_rate"] = int(audio_sample_rate)
        _save_diffsynth_audio(audio, int(audio_sample_rate), audio_path)
        if not audio_path.is_file():
            raise RuntimeError(f"DiffSynth audio export did not create output file: {audio_path}")
        metadata["audio_size"] = audio_path.stat().st_size
    return metadata


def diffsynth_output_audio_sample_rate(pipe: Any, default: int = 48000) -> int:
    for module_name in ("audio_vocoder", "audio_vae_decoder", "audio_vae", "audio_processor"):
        module = getattr(pipe, module_name, None)
        if module is None:
            continue
        for attr_name in ("output_sampling_rate", "sample_rate"):
            value = getattr(module, attr_name, None)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
    return int(default)


def split_diffsynth_output(output: Any) -> Tuple[Any, Any | None]:
    if isinstance(output, tuple):
        if len(output) != 2:
            raise ValueError(f"Unsupported DiffSynth tuple output length: {len(output)}")
        return output[0], output[1]
    return output, None


def _unlink_existing_output(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except IsADirectoryError as exc:
        raise RuntimeError(f"DiffSynth output path is a directory: {path}") from exc


def _add_wan_common_components(add, *, dit_repo: str, vae_version: str) -> None:
    _add_wan_tokenizer_component(add, dit_repo=dit_repo)
    _add_wan_weight_components(add, dit_repo=dit_repo, vae_version=vae_version)


def _add_wan_image_components(add, *, dit_repo: str, vae_version: str) -> None:
    _add_wan_common_components(add, dit_repo=dit_repo, vae_version=vae_version)
    _add_wan_clip_component(add, dit_repo=dit_repo)


def _add_wan_tokenizer_component(add, *, dit_repo: str) -> None:
    add(
        "tokenizer",
        (
            ("Wan2.1-T2V-1.3B", "google/umt5-xxl"),
            (dit_repo, "google/umt5-xxl"),
        ),
        directory=True,
        required_files=("tokenizer.json", "tokenizer_config.json", "spiece.model"),
    )


def _add_wan_clip_component(add, *, dit_repo: str) -> None:
    add(
        "image_encoder",
        (
            (dit_repo, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            ("Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
        ),
    )


def _add_wan_weight_components(add, *, dit_repo: str, vae_version: str) -> None:
    add(
        "text_encoder",
        (
            (dit_repo, "models_t5_umt5-xxl-enc-bf16.pth"),
            ("Wan2.1-T2V-1.3B", "models_t5_umt5-xxl-enc-bf16.pth"),
            ("Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
        ),
    )
    vae_name = "Wan2.2_VAE" if vae_version == "22" else "Wan2.1_VAE"
    if vae_version == "22":
        vae_candidates = (
            (dit_repo, f"{vae_name}.pth"),
            ("Wan2.2-TI2V-5B", f"{vae_name}.pth"),
            ("Wan2.2-T2V-A14B", f"{vae_name}.pth"),
            ("Wan2.2-I2V-A14B", f"{vae_name}.pth"),
            ("Wan-Series-Converted-Safetensors", f"{vae_name}.safetensors"),
        )
    else:
        vae_candidates = (
            (dit_repo, f"{vae_name}.pth"),
            ("Wan2.1-T2V-1.3B", f"{vae_name}.pth"),
            ("Wan2.1-T2V-14B", f"{vae_name}.pth"),
            ("Wan2.2-T2V-A14B", f"{vae_name}.pth"),
            ("Wan-Series-Converted-Safetensors", f"{vae_name}.safetensors"),
        )
    add(
        "vae",
        vae_candidates,
    )


def _add_ltx2_components(add, *, repo: str) -> None:
    _add_ltx2_text_components(add)
    add("dit", ((repo, "transformer.safetensors"),))
    add("video_vae_encoder", ((repo, "video_vae_encoder.safetensors"),))
    add("video_vae_decoder", ((repo, "video_vae_decoder.safetensors"),))
    add("audio_vocoder", ((repo, "audio_vocoder.safetensors"),))
    add("text_encoder_post_modules", ((repo, "text_encoder_post_modules.safetensors"),))


def _add_ltx2_text_components(add) -> None:
    add(
        "tokenizer",
        (("google/gemma-3-12b-it-qat-q4_0-unquantized", ""),),
        directory=True,
        required_files=(
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.json",
            "config.json",
            "generation_config.json",
        ),
    )
    add("text_encoder", (("google/gemma-3-12b-it-qat-q4_0-unquantized", "model*.safetensors"),))


def _find_first_complete(
    model_root: Path,
    candidates: Sequence[Tuple[str, str]],
    *,
    directory: bool,
    required_files: Sequence[str] = (),
) -> Tuple[Tuple[Path, ...], List[str]]:
    notes: List[str] = []
    for repo_dir, pattern in _dedupe_candidates(candidates):
        checked = []
        for repo_base in _repo_dir_candidates(model_root, repo_dir):
            base = repo_base / pattern
            checked.append(str(base))
            if directory:
                if base.is_dir():
                    missing_required = [
                        rel_path
                        for rel_path in required_files
                        if not (base / rel_path).is_file()
                    ]
                    if missing_required:
                        notes.append(
                            f"missing required file(s) in directory {base}: "
                            f"{', '.join(missing_required)}"
                        )
                        continue
                    if _directory_has_files(base):
                        return (base,), notes
                    notes.append(f"empty directory {base}")
                continue

            paths = tuple(
                sorted(
                    path
                    for path in base.parent.glob(base.name)
                    if path.is_file() and path.stat().st_size > 0
                )
            )
            if not paths:
                continue
            complete_paths, incomplete_note = _select_complete_file_set(paths)
            if incomplete_note:
                notes.append(f"incomplete shards for {base}: {incomplete_note}")
                continue
            return complete_paths, notes
        if directory:
            notes.append(f"missing directory {' or '.join(checked)}")
            continue

        notes.append(f"missing files {' or '.join(checked)}")
    return (), notes


def _directory_has_files(path: Path) -> bool:
    return any(item.is_file() for item in path.rglob("*"))


def _repo_dir_candidates(model_root: Path, repo_dir: str) -> Tuple[Path, ...]:
    candidates = [model_root / repo_dir]
    leaf_dir = repo_dir.rsplit("/", 1)[-1]
    if "/" in repo_dir:
        candidates.append(model_root / leaf_dir)
    else:
        candidates.extend(sorted(path for path in model_root.glob(f"*/{repo_dir}") if path.is_dir()))
    candidates.extend(sorted(path for path in model_root.glob(f"*/{leaf_dir}") if path.is_dir()))
    candidates.extend(sorted(path for path in model_root.glob(f"*/*/{leaf_dir}") if path.is_dir()))
    return tuple(_dedupe_paths(candidates))


def _dedupe_candidates(candidates: Sequence[Tuple[str, str]]) -> Tuple[Tuple[str, str], ...]:
    seen = set()
    result = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return tuple(result)


def _dedupe_paths(paths: Sequence[Path]) -> Tuple[Path, ...]:
    seen = set()
    result = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return tuple(result)


def _select_complete_file_set(paths: Sequence[Path]) -> Tuple[Tuple[Path, ...], Optional[str]]:
    unsharded: List[Path] = []
    groups: Dict[Tuple[str, int, str], Dict[int, Path]] = {}
    for path in paths:
        match = _SHARD_RE.match(path.name)
        if not match:
            unsharded.append(path)
            continue
        shard_idx = int(match.group("idx"))
        total = int(match.group("total"))
        group_key = (match.group("prefix"), total, match.group("variant"))
        groups.setdefault(group_key, {})[shard_idx] = path

    complete_groups = []
    for (prefix, total, variant), present in groups.items():
        expected = set(range(1, total + 1))
        if expected <= set(present):
            complete_groups.append((prefix, total, variant, tuple(present[idx] for idx in range(1, total + 1))))

    if complete_groups:
        complete_groups.sort(key=_shard_group_preference)
        return complete_groups[0][3], None

    if unsharded:
        return tuple(sorted(unsharded)), None

    return (), _incomplete_shard_note_from_groups(groups)


def _shard_group_preference(group) -> Tuple[int, str, int]:
    _prefix, _total, variant, paths = group
    if variant == "-bf16":
        rank = 0
    elif variant == "":
        rank = 1
    else:
        rank = 2
    return rank, variant, sum(path.stat().st_size for path in paths)


def _incomplete_shard_note_from_groups(groups: Mapping[Tuple[str, int, str], Mapping[int, Path]]) -> str:
    notes = []
    for (_prefix, shard_total, variant), present in sorted(groups.items()):
        expected = set(range(1, shard_total + 1))
        missing = sorted(expected - set(present))
        if not missing:
            continue
        notes.append(_format_missing_shards(missing, shard_total, variant))
    return "; ".join(notes) if notes else "mixed or incomplete shard set"


def _format_missing_shards(missing: Sequence[int], shard_total: int, variant: str) -> str:
    preview = ", ".join(f"{idx:05d}" for idx in missing[:8])
    if len(missing) > 8:
        preview = f"{preview}, ..."
    variant_note = f" for variant {variant}" if variant else ""
    return f"missing shard indexes {preview} of {shard_total:05d}{variant_note}"


def _model_configs_from_components(
    resolved: ResolvedDiffSynthModel,
    names: Iterable[str],
    model_config_cls,
    *,
    offload_device: Optional[str],
) -> List[Any]:
    configs = []
    seen_paths = set()
    for name in names:
        path = _model_config_path(resolved.components[name])
        path_key = tuple(path) if isinstance(path, list) else (path,)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        kwargs = {"path": path}
        if offload_device:
            kwargs["offload_device"] = offload_device
        configs.append(model_config_cls(**kwargs))
    return configs


def _model_config_path(paths: Sequence[Path]) -> str | List[str]:
    if len(paths) == 1:
        return str(paths[0])
    return [str(path) for path in paths]


def _single_path(resolved: ResolvedDiffSynthModel, name: str) -> Path:
    paths = resolved.components[name]
    if len(paths) != 1:
        raise ValueError(f"Expected one path for DiffSynth component {name}, got {len(paths)}")
    return paths[0]


def _optional_model_config(resolved: ResolvedDiffSynthModel, name: str, model_config_cls):
    if name not in resolved.components:
        return None
    return model_config_cls(path=str(_single_path(resolved, name)))


def _load_wav2vec2_processor(path: Path):
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(str(path))
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(path))
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def _wan_model_config_order(resolved: ResolvedDiffSynthModel) -> Tuple[str, ...]:
    names = []
    if "dit_high_noise" in resolved.components:
        names.extend(("dit_high_noise", "dit_low_noise"))
    else:
        names.append("dit")
    for optional in ("motion_controller", "vace", "animate_adapter", "vap", "audio_encoder"):
        if optional in resolved.components:
            names.append(optional)
    names.append("text_encoder")
    if "image_encoder" in resolved.components:
        names.append("image_encoder")
    names.append("vae")
    return tuple(names)


def _ltx2_model_config_order(resolved: ResolvedDiffSynthModel) -> Tuple[str, ...]:
    if resolved.spec.key == "ltx23":
        return tuple(
            name
            for name in ("text_encoder", "source_checkpoint", "latent_upsampler")
            if name in resolved.components
        )

    names = [
        "text_encoder",
        "dit",
        "video_vae_encoder",
        "video_vae_decoder",
        "audio_vae_decoder",
        "audio_vocoder",
        "audio_vae_encoder",
        "text_encoder_post_modules",
    ]
    if "latent_upsampler" in resolved.components:
        names.append("latent_upsampler")
    return tuple(name for name in names if name in resolved.components)


def _diffsynth_save_video():
    try:
        from diffsynth import save_video

        return save_video
    except ImportError:
        from diffsynth.utils.data import save_video

        return save_video


def _diffsynth_save_audio():
    from diffsynth.utils.data.audio import save_audio

    return save_audio


def _save_diffsynth_audio(audio: Any, sample_rate: int, audio_path: Path) -> None:
    try:
        _diffsynth_save_audio()(audio, sample_rate, str(audio_path))
        return
    except ModuleNotFoundError as exc:
        if exc.name != "torchcodec":
            raise
        torchcodec_error = exc
    except ImportError as exc:
        if "torchcodec" not in str(exc):
            raise
        torchcodec_error = exc

    try:
        import torch
        import torchaudio

        waveform = torch.as_tensor(audio).detach().cpu()
        if waveform.ndim == 3:
            waveform = waveform[0]
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(f"expected audio tensor with shape [channels, samples], got {tuple(waveform.shape)}")
        torchaudio.save(str(audio_path), waveform.float(), sample_rate)
    except Exception as exc:
        raise RuntimeError(
            "DiffSynth returned audio, but audio export failed through both "
            f"DiffSynth torchcodec save_audio and torchaudio fallback: {exc}"
        ) from torchcodec_error
