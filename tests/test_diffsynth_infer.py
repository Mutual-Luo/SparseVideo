from __future__ import annotations

import json
import importlib.util
import inspect
import os
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest
import sparsevideo


REPO_ROOT = Path(__file__).resolve().parents[1]
LTX_GEMMA_REPO = "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"
SCRIPT = REPO_ROOT / "scripts" / "infer_diffsynth.py"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _infer_diffsynth.models import (
    DEFAULT_MODEL_ROOT,
    diffsynth_output_audio_sample_rate,
    diffsynth_model_list_lines,
    get_diffsynth_model_spec,
    list_deferred_diffsynth_model_specs,
    list_diffsynth_model_specs,
    load_diffsynth_pipeline,
    resolve_diffsynth_model_paths,
    save_diffsynth_output,
    split_diffsynth_output,
)


def test_diffsynth_inference_helpers_stay_out_of_sparsevideo_package():
    assert not (REPO_ROOT / "src/sparsevideo/_diffsynth_infer.py").exists()
    assert not list((REPO_ROOT / "src/sparsevideo").glob("*diffsynth*infer*.py"))

    forbidden = (
        "_infer_diffsynth.models",
        "load_diffsynth_pipeline",
        "resolve_diffsynth_model_paths",
        "save_diffsynth_output",
        "ModelConfig(",
    )
    for path in (REPO_ROOT / "src/sparsevideo").glob("*diffsynth*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.relative_to(REPO_ROOT)} leaked script inference helper {token}"


def test_default_diffsynth_model_root_matches_download_script_layout():
    assert DEFAULT_MODEL_ROOT == Path("/home/dataset-assist-0/public-models")


def test_generic_sparsevideo_model_root_env_does_not_override_diffsynth_default():
    env = os.environ.copy()
    env["SPARSEVIDEO_MODEL_ROOT"] = "/tmp/wrong-public-models"
    env.pop("SPARSEVIDEO_DIFFSYNTH_MODEL_ROOT", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'scripts'); "
            "from _infer_diffsynth.models import DEFAULT_MODEL_ROOT; print(DEFAULT_MODEL_ROOT)",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "/home/dataset-assist-0/public-models"


def test_diffsynth_model_root_env_override_is_specific():
    env = os.environ.copy()
    env["SPARSEVIDEO_DIFFSYNTH_MODEL_ROOT"] = "/tmp/diffsynth-models"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'scripts'); "
            "from _infer_diffsynth.models import DEFAULT_MODEL_ROOT; print(DEFAULT_MODEL_ROOT)",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "/tmp/diffsynth-models"



def test_split_diffsynth_output_accepts_video_and_video_audio():
    video = [object()]
    audio = object()

    assert split_diffsynth_output(video) == (video, None)
    assert split_diffsynth_output((video, audio)) == (video, audio)


def test_split_diffsynth_output_rejects_unknown_tuple_shape():
    with pytest.raises(ValueError, match="Unsupported DiffSynth tuple output length"):
        split_diffsynth_output((1, 2, 3))


def test_diffsynth_output_audio_sample_rate_prefers_pipeline_output_rate():
    pipe = type(
        "Pipe",
        (),
        {
            "audio_vocoder": type("Vocoder", (), {"output_sampling_rate": 24000})(),
            "audio_vae_decoder": type("Decoder", (), {"sample_rate": 16000})(),
        },
    )()

    assert diffsynth_output_audio_sample_rate(pipe) == 24000


def test_diffsynth_output_audio_sample_rate_falls_back_to_mova_audio_vae():
    pipe = type("Pipe", (), {"audio_vae": type("AudioVAE", (), {"sample_rate": 48000})()})()

    assert diffsynth_output_audio_sample_rate(pipe, default=44100) == 48000
    assert diffsynth_output_audio_sample_rate(object(), default=44100) == 44100


def test_save_diffsynth_output_uses_torchaudio_when_torchcodec_is_missing(monkeypatch, tmp_path):
    import torch

    models = sys.modules["_infer_diffsynth.models"]
    calls = {}

    def fake_save_video(frames, save_path, *, fps, quality):
        calls["video"] = {"frames": frames, "fps": fps, "quality": quality}
        Path(save_path).write_bytes(b"video")

    def fake_save_audio(audio, sample_rate, save_path):
        raise ModuleNotFoundError("No module named 'torchcodec'", name="torchcodec")

    fake_torchaudio = type(sys)("torchaudio")

    def fake_torchaudio_save(save_path, waveform, sample_rate):
        calls["audio"] = {
            "shape": tuple(waveform.shape),
            "sample_rate": sample_rate,
        }
        Path(save_path).write_bytes(b"audio")

    fake_torchaudio.save = fake_torchaudio_save
    monkeypatch.setattr(models, "_diffsynth_save_video", lambda: fake_save_video)
    monkeypatch.setattr(models, "_diffsynth_save_audio", lambda: fake_save_audio)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)

    metadata = save_diffsynth_output(
        ([object()], torch.zeros(1, 16)),
        tmp_path / "out.mp4",
        fps=24,
        quality=6,
        audio_sample_rate=44100,
    )

    assert calls["video"]["fps"] == 24
    assert calls["video"]["quality"] == 6
    assert calls["audio"] == {"shape": (1, 16), "sample_rate": 44100}
    assert metadata["output_type"] == "video_audio"
    assert metadata["audio_file"] == str(tmp_path / "out.wav")
    assert metadata["audio_size"] == 5
    assert "audio_save_error" not in metadata


def test_save_diffsynth_output_rejects_missing_video_file(monkeypatch, tmp_path):
    models = sys.modules["_infer_diffsynth.models"]
    monkeypatch.setattr(models, "_diffsynth_save_video", lambda: lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="video export did not create output file"):
        save_diffsynth_output([object()], tmp_path / "missing.mp4", fps=24)


def test_save_diffsynth_output_rejects_stale_existing_video_file(monkeypatch, tmp_path):
    output_file = tmp_path / "stale.mp4"
    output_file.write_bytes(b"old")
    models = sys.modules["_infer_diffsynth.models"]
    monkeypatch.setattr(models, "_diffsynth_save_video", lambda: lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="video export did not create output file"):
        save_diffsynth_output([object()], output_file, fps=24)
    assert not output_file.exists()


def test_save_diffsynth_output_rejects_missing_audio_file(monkeypatch, tmp_path):
    import torch

    models = sys.modules["_infer_diffsynth.models"]

    def fake_save_video(frames, save_path, *, fps, quality):
        Path(save_path).write_bytes(b"video")

    monkeypatch.setattr(models, "_diffsynth_save_video", lambda: fake_save_video)
    monkeypatch.setattr(models, "_diffsynth_save_audio", lambda: lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="audio export did not create output file"):
        save_diffsynth_output(([object()], torch.zeros(1, 16)), tmp_path / "missing_audio.mp4", fps=24)


def test_save_diffsynth_output_rejects_stale_existing_audio_file(monkeypatch, tmp_path):
    import torch

    output_file = tmp_path / "stale_audio.mp4"
    audio_file = tmp_path / "stale_audio.wav"
    audio_file.write_bytes(b"old")
    models = sys.modules["_infer_diffsynth.models"]

    def fake_save_video(frames, save_path, *, fps, quality):
        Path(save_path).write_bytes(b"video")

    monkeypatch.setattr(models, "_diffsynth_save_video", lambda: fake_save_video)
    monkeypatch.setattr(models, "_save_diffsynth_audio", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="audio export did not create output file"):
        save_diffsynth_output(([object()], torch.zeros(1, 16)), output_file, fps=24)
    assert not audio_file.exists()

def _load_infer_module():
    spec = importlib.util.spec_from_file_location("sparsevideo_infer_diffsynth", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _call_signature_params(pipeline_cls) -> set[str]:
    params = set(inspect.signature(pipeline_cls.__call__).parameters)
    params.discard("self")
    return params


def _installed_diffsynth_video_model_config_examples() -> set[tuple[str, str]]:
    pytest.importorskip("diffsynth")
    from diffsynth.configs import model_configs

    source = Path(model_configs.__file__).read_text(encoding="utf-8")
    sections = list(re.finditer(r"^(\w+_series)\s*=\s*\[", source, re.MULTILINE))
    examples: set[tuple[str, str]] = set()
    for index, match in enumerate(sections):
        if match.group(1) not in {"wan_series", "mova_series", "ltx2_series"}:
            continue
        end = sections[index + 1].start() if index + 1 < len(sections) else len(source)
        section = source[match.start():end]
        examples.update(
            re.findall(
                r'# Example: ModelConfig\(model_id="([^"]+)", origin_file_pattern="([^"]+)"',
                section,
            )
        )
    return examples


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _make_wan21_common(root: Path) -> None:
    _touch(root / "Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json")
    _touch(root / "Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json")
    _touch(root / "Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model")
    _touch(root / "Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json")
    _touch(root / "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    _touch(root / "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")


def _make_wan21_image_common(root: Path, repo: str) -> None:
    _make_wan21_common(root)
    _touch(root / repo / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")


def _make_wan22_image_common(root: Path, repo: str) -> None:
    _make_wan21_common(root)
    _touch(root / "Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    _touch(root / repo / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")


def _make_ltx2_common(root: Path, repo: str) -> None:
    _make_ltx2_text(root)
    _touch(root / f"{repo}/transformer.safetensors")
    _touch(root / f"{repo}/video_vae_encoder.safetensors")
    _touch(root / f"{repo}/video_vae_decoder.safetensors")
    _touch(root / f"{repo}/audio_vocoder.safetensors")
    _touch(root / f"{repo}/text_encoder_post_modules.safetensors")


def _make_ltx2_text(root: Path) -> None:
    _touch(root / "gemma-3-12b-it-qat-q4_0-unquantized/tokenizer.json")
    _touch(root / "gemma-3-12b-it-qat-q4_0-unquantized/tokenizer.model")
    _touch(root / "gemma-3-12b-it-qat-q4_0-unquantized/tokenizer_config.json")
    _touch(root / "gemma-3-12b-it-qat-q4_0-unquantized/preprocessor_config.json")
    _touch(root / "gemma-3-12b-it-qat-q4_0-unquantized/processor_config.json")
    _touch(root / "gemma-3-12b-it-qat-q4_0-unquantized/special_tokens_map.json")
    _touch(root / "gemma-3-12b-it-qat-q4_0-unquantized/added_tokens.json")
    _touch(root / "gemma-3-12b-it-qat-q4_0-unquantized/chat_template.json")
    _touch(root / "gemma-3-12b-it-qat-q4_0-unquantized/config.json")
    _touch(root / "gemma-3-12b-it-qat-q4_0-unquantized/generation_config.json")
    _touch(root / "gemma-3-12b-it-qat-q4_0-unquantized/model.safetensors")


def _make_mova_common(root: Path) -> None:
    _make_wan21_common(root)
    _touch(root / "Wan2.1-T2V-14B/diffusion_pytorch_model.safetensors")
    _touch(root / "MOVA-720p/tokenizer/tokenizer.json")
    _touch(root / "MOVA-720p/tokenizer/tokenizer_config.json")
    _touch(root / "MOVA-720p/tokenizer/special_tokens_map.json")
    _touch(root / "MOVA-720p/audio_dit/diffusion_pytorch_model.safetensors")
    _touch(root / "MOVA-720p/audio_vae/diffusion_pytorch_model.safetensors")
    _touch(root / "MOVA-720p/dual_tower_bridge/diffusion_pytorch_model.safetensors")


def _make_wan22_t2v_a14b_common(root: Path) -> None:
    _make_wan21_common(root)
    _touch(root / "Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model.safetensors")
    _touch(root / "Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model.safetensors")


def test_resolve_diffsynth_wan_flat_model_layout(tmp_path):
    _make_wan21_common(tmp_path)
    _touch(tmp_path / "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")

    resolved = resolve_diffsynth_model_paths("wan21-t2v-1.3b", model_root=tmp_path)

    assert resolved.complete
    assert resolved.components["tokenizer"] == (tmp_path / "Wan2.1-T2V-1.3B/google/umt5-xxl",)
    assert resolved.components["dit"] == (tmp_path / "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",)


def test_resolve_diffsynth_rejects_empty_checkpoint_file(tmp_path):
    _make_wan21_common(tmp_path)
    path = tmp_path / "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")

    resolved = resolve_diffsynth_model_paths("wan21-t2v-1.3b", model_root=tmp_path)

    assert not resolved.complete
    assert any("dit: missing files" in item for item in resolved.missing)


def test_resolve_diffsynth_speedcontrol_adds_motion_controller(tmp_path):
    _make_wan21_common(tmp_path)
    _touch(tmp_path / "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    _touch(tmp_path / "DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1/model.safetensors")

    resolved = resolve_diffsynth_model_paths("speedcontrol", model_root=tmp_path)

    assert resolved.complete
    assert resolved.spec.key == "wan21-speedcontrol-1.3b"
    assert resolved.components["dit"] == (tmp_path / "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",)
    assert resolved.components["motion_controller"] == (
        tmp_path / "DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1/model.safetensors",
    )


def test_resolve_diffsynth_wan_nested_repo_layout(tmp_path):
    _touch(tmp_path / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json")
    _touch(tmp_path / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json")
    _touch(tmp_path / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model")
    _touch(tmp_path / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json")
    _touch(tmp_path / "Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    _touch(tmp_path / "Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    _touch(tmp_path / "Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")

    resolved = resolve_diffsynth_model_paths("wan21-t2v-1.3b", model_root=tmp_path)

    assert resolved.complete
    assert resolved.components["dit"] == (
        tmp_path / "Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    )


def test_resolve_diffsynth_rejects_empty_directory_components(tmp_path):
    (tmp_path / "Wan2.1-T2V-1.3B/google/umt5-xxl").mkdir(parents=True)
    _touch(tmp_path / "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    _touch(tmp_path / "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    _touch(tmp_path / "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")

    resolved = resolve_diffsynth_model_paths("wan21-t2v-1.3b", model_root=tmp_path)

    assert not resolved.complete
    assert any("missing required file(s)" in item for item in resolved.missing)


def test_resolve_diffsynth_rejects_incomplete_wan_tokenizer_dir(tmp_path):
    _touch(tmp_path / "Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json")
    _touch(tmp_path / "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    _touch(tmp_path / "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    _touch(tmp_path / "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")

    resolved = resolve_diffsynth_model_paths("wan21-t2v-1.3b", model_root=tmp_path)

    assert not resolved.complete
    assert any("tokenizer_config.json" in item for item in resolved.missing)
    assert any("spiece.model" in item for item in resolved.missing)


def test_resolve_diffsynth_reuses_two_level_nested_longcat_layout(tmp_path):
    _make_wan21_common(tmp_path)
    for idx in range(1, 7):
        _touch(
            tmp_path
            / "Longcat/weights/LongCat-Video/dit"
            / f"diffusion_pytorch_model-{idx:05d}-of-00006.safetensors"
        )

    resolved = resolve_diffsynth_model_paths("longcat-video", model_root=tmp_path)

    assert resolved.complete
    assert resolved.components["dit"][0] == (
        tmp_path
        / "Longcat/weights/LongCat-Video/dit/diffusion_pytorch_model-00001-of-00006.safetensors"
    )


def test_resolve_diffsynth_image_model_uses_clip_and_requires_input(tmp_path):
    _make_wan21_image_common(tmp_path, "Wan2.1-I2V-14B-720P")
    _touch(tmp_path / "Wan2.1-I2V-14B-720P/diffusion_pytorch_model.safetensors")

    resolved = resolve_diffsynth_model_paths("wan21-i2v", model_root=tmp_path)

    assert resolved.complete
    assert resolved.spec.key == "wan21-i2v-14b-720p"
    assert resolved.spec.required_inputs == ("input_image",)
    assert resolved.components["image_encoder"] == (
        tmp_path / "Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    )


def test_resolve_diffsynth_flf_model_uses_clip_and_requires_first_last_images(tmp_path):
    _make_wan21_image_common(tmp_path, "Wan2.1-FLF2V-14B-720P")
    _touch(tmp_path / "Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model.safetensors")

    resolved = resolve_diffsynth_model_paths("wan21-flf2v-14b-720p", model_root=tmp_path)

    assert resolved.complete
    assert resolved.spec.required_inputs == ("input_image", "end_image")
    assert resolved.components["image_encoder"] == (
        tmp_path / "Wan2.1-FLF2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    )


def test_resolve_diffsynth_wan21_fun_control_uses_control_video_without_clip(tmp_path):
    _make_wan21_common(tmp_path)
    _touch(tmp_path / "Wan2.1-Fun-1.3B-Control/diffusion_pytorch_model.safetensors")

    resolved = resolve_diffsynth_model_paths("wan21-fun-1.3b-control", model_root=tmp_path)

    assert resolved.complete
    assert resolved.spec.required_inputs == ("control_video",)
    assert resolved.components["dit"] == (
        tmp_path / "Wan2.1-Fun-1.3B-Control/diffusion_pytorch_model.safetensors",
    )
    assert "image_encoder" not in resolved.components


def test_resolve_diffsynth_wan21_fun_inp_uses_clip_and_first_last_images(tmp_path):
    _make_wan21_image_common(tmp_path, "Wan2.1-Fun-14B-InP")
    _touch(tmp_path / "Wan2.1-Fun-14B-InP/diffusion_pytorch_model.safetensors")

    resolved = resolve_diffsynth_model_paths("wan21-fun-inp-14b", model_root=tmp_path)

    assert resolved.complete
    assert resolved.spec.key == "wan21-fun-14b-inp"
    assert resolved.spec.required_inputs == ("input_image", "end_image")
    assert resolved.components["image_encoder"] == (
        tmp_path / "Wan2.1-Fun-14B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    )


def test_resolve_diffsynth_wan21_fun_control_camera_requires_image_and_camera_direction(tmp_path):
    _make_wan21_image_common(tmp_path, "Wan2.1-Fun-V1.1-1.3B-Control-Camera")
    _touch(tmp_path / "Wan2.1-Fun-V1.1-1.3B-Control-Camera/diffusion_pytorch_model.safetensors")

    resolved = resolve_diffsynth_model_paths("wan21-fun-v1.1-1.3b-control-camera", model_root=tmp_path)

    assert resolved.complete
    assert resolved.spec.key == "wan21-fun-v11-1.3b-control-camera"
    assert resolved.spec.required_inputs == ("input_image", "camera_control_direction")
    assert resolved.components["dit"] == (
        tmp_path / "Wan2.1-Fun-V1.1-1.3B-Control-Camera/diffusion_pytorch_model.safetensors",
    )
    assert resolved.components["image_encoder"] == (
        tmp_path
        / "Wan2.1-Fun-V1.1-1.3B-Control-Camera/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    )


def test_resolve_diffsynth_wan22_fun_control_uses_high_noise_dit(tmp_path):
    _make_wan21_common(tmp_path)
    _touch(tmp_path / "Wan2.2-Fun-A14B-Control/high_noise_model/diffusion_pytorch_model.safetensors")

    resolved = resolve_diffsynth_model_paths("wan22-fun-a14b-control", model_root=tmp_path)

    assert resolved.complete
    assert resolved.spec.required_inputs == ("control_video", "reference_image")
    assert resolved.components["dit"] == (
        tmp_path / "Wan2.2-Fun-A14B-Control/high_noise_model/diffusion_pytorch_model.safetensors",
    )
    assert "image_encoder" not in resolved.components


def test_resolve_diffsynth_wan22_animate_uses_wan21_vae(tmp_path):
    _make_wan21_common(tmp_path)
    _touch(tmp_path / "Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    _touch(tmp_path / "Wan2.2-Animate-14B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
    _touch(tmp_path / "Wan2.2-Animate-14B/diffusion_pytorch_model.safetensors")

    resolved = resolve_diffsynth_model_paths("wan22-animate-14b", model_root=tmp_path)

    assert resolved.complete
    assert resolved.components["vae"] == (tmp_path / "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",)


def test_load_diffsynth_wan_shared_checkpoint_roles_are_not_duplicated(monkeypatch, tmp_path):
    _make_wan21_common(tmp_path)
    _touch(tmp_path / "VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors")

    calls = {}

    class FakeModelConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeWanVideoPipeline:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.update(kwargs)
            return FakeWanVideoPipeline()

        def enable_vram_management(self):
            calls["enabled_vram_management"] = True

    fake_diffsynth = type(sys)("diffsynth")
    fake_diffsynth.ModelConfig = FakeModelConfig
    fake_pipeline_module = type(sys)("diffsynth.pipelines.wan_video")
    fake_pipeline_module.WanVideoPipeline = FakeWanVideoPipeline
    monkeypatch.setitem(sys.modules, "diffsynth", fake_diffsynth)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines", type(sys)("diffsynth.pipelines"))
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines.wan_video", fake_pipeline_module)

    pipe, resolved = load_diffsynth_pipeline(
        "wan21-vace-1.3b",
        model_root=tmp_path,
        torch_dtype="bf16",
        device="cpu",
    )

    assert isinstance(pipe, FakeWanVideoPipeline)
    assert resolved.components["dit"] == resolved.components["vace"]
    assert [Path(config.kwargs["path"]).name for config in calls["model_configs"]] == [
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
    ]
    assert calls["enabled_vram_management"] is True
    assert pipe._sparsevideo_model_key == "wan21-vace-1.3b"


def test_load_diffsynth_speedcontrol_loads_motion_controller(monkeypatch, tmp_path):
    _make_wan21_common(tmp_path)
    _touch(tmp_path / "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    _touch(tmp_path / "Wan2.1-1.3b-speedcontrol-v1/model.safetensors")
    calls = {}

    class FakeModelConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeWanVideoPipeline:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.update(kwargs)
            return FakeWanVideoPipeline()

        def enable_vram_management(self):
            calls["enabled_vram_management"] = True

    fake_diffsynth = type(sys)("diffsynth")
    fake_diffsynth.ModelConfig = FakeModelConfig
    fake_pipeline_module = type(sys)("diffsynth.pipelines.wan_video")
    fake_pipeline_module.WanVideoPipeline = FakeWanVideoPipeline
    monkeypatch.setitem(sys.modules, "diffsynth", fake_diffsynth)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines", type(sys)("diffsynth.pipelines"))
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines.wan_video", fake_pipeline_module)

    pipe, resolved = load_diffsynth_pipeline(
        "wan21-speedcontrol",
        model_root=tmp_path,
        torch_dtype="bf16",
        device="cpu",
    )

    assert isinstance(pipe, FakeWanVideoPipeline)
    assert resolved.spec.key == "wan21-speedcontrol-1.3b"
    assert [
        Path(config.kwargs["path"]).relative_to(tmp_path).as_posix()
        for config in calls["model_configs"]
    ] == [
        "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "Wan2.1-1.3b-speedcontrol-v1/model.safetensors",
        "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ]
    assert calls["enabled_vram_management"] is True
    assert pipe._sparsevideo_model_key == "wan21-speedcontrol-1.3b"


def test_load_diffsynth_animate_shared_checkpoint_roles_are_not_duplicated(monkeypatch, tmp_path):
    _make_wan22_image_common(tmp_path, "Wan2.2-Animate-14B")
    _touch(tmp_path / "Wan2.2-Animate-14B/diffusion_pytorch_model.safetensors")
    calls = {}

    class FakeModelConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeWanVideoPipeline:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.update(kwargs)
            return FakeWanVideoPipeline()

        def enable_vram_management(self):
            calls["enabled_vram_management"] = True

    fake_diffsynth = type(sys)("diffsynth")
    fake_diffsynth.ModelConfig = FakeModelConfig
    fake_pipeline_module = type(sys)("diffsynth.pipelines.wan_video")
    fake_pipeline_module.WanVideoPipeline = FakeWanVideoPipeline
    monkeypatch.setitem(sys.modules, "diffsynth", fake_diffsynth)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines", type(sys)("diffsynth.pipelines"))
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines.wan_video", fake_pipeline_module)

    pipe, resolved = load_diffsynth_pipeline("wan22-animate-14b", model_root=tmp_path, torch_dtype="bf16", device="cpu")

    assert isinstance(pipe, FakeWanVideoPipeline)
    assert resolved.components["dit"] == resolved.components["animate_adapter"]
    assert [
        Path(config.kwargs["path"]).relative_to(tmp_path).as_posix()
        for config in calls["model_configs"]
    ] == [
        "Wan2.2-Animate-14B/diffusion_pytorch_model.safetensors",
        "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.2-Animate-14B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ]
    assert calls["enabled_vram_management"] is True
    assert pipe._sparsevideo_model_key == "wan22-animate-14b"


def test_load_diffsynth_vap_shared_checkpoint_roles_are_not_duplicated(monkeypatch, tmp_path):
    _make_wan21_image_common(tmp_path, "Video-As-Prompt-Wan2.1-14B")
    _touch(tmp_path / "Video-As-Prompt-Wan2.1-14B/transformer/diffusion_pytorch_model.safetensors")
    calls = {}

    class FakeModelConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeWanVideoPipeline:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.update(kwargs)
            return FakeWanVideoPipeline()

        def enable_vram_management(self):
            calls["enabled_vram_management"] = True

    fake_diffsynth = type(sys)("diffsynth")
    fake_diffsynth.ModelConfig = FakeModelConfig
    fake_pipeline_module = type(sys)("diffsynth.pipelines.wan_video")
    fake_pipeline_module.WanVideoPipeline = FakeWanVideoPipeline
    monkeypatch.setitem(sys.modules, "diffsynth", fake_diffsynth)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines", type(sys)("diffsynth.pipelines"))
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines.wan_video", fake_pipeline_module)

    pipe, resolved = load_diffsynth_pipeline(
        "video-as-prompt-wan21-14b",
        model_root=tmp_path,
        torch_dtype="bf16",
        device="cpu",
    )

    assert isinstance(pipe, FakeWanVideoPipeline)
    assert resolved.components["dit"] == resolved.components["vap"]
    assert [
        Path(config.kwargs["path"]).relative_to(tmp_path).as_posix()
        for config in calls["model_configs"]
    ] == [
        "Video-As-Prompt-Wan2.1-14B/transformer/diffusion_pytorch_model.safetensors",
        "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "Video-As-Prompt-Wan2.1-14B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ]
    assert calls["enabled_vram_management"] is True
    assert pipe._sparsevideo_model_key == "video-as-prompt-wan21-14b"


def test_resolve_diffsynth_s2v_uses_audio_encoder_and_processor(tmp_path):
    _make_wan22_image_common(tmp_path, "Wan2.2-S2V-14B")
    _touch(tmp_path / "Wan2.2-S2V-14B/diffusion_pytorch_model.safetensors")
    _touch(tmp_path / "Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors")
    _touch(tmp_path / "Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/preprocessor_config.json")
    _touch(tmp_path / "Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/vocab.json")

    resolved = resolve_diffsynth_model_paths("wan22-s2v", model_root=tmp_path)

    assert resolved.complete
    assert resolved.spec.key == "wan22-s2v-14b"
    assert resolved.components["audio_encoder"] == (
        tmp_path / "Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors",
    )
    assert resolved.components["audio_processor"] == (
        tmp_path / "Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english",
    )


def test_resolve_diffsynth_s2v_rejects_incomplete_audio_processor_dir(tmp_path):
    _make_wan22_image_common(tmp_path, "Wan2.2-S2V-14B")
    _touch(tmp_path / "Wan2.2-S2V-14B/diffusion_pytorch_model.safetensors")
    _touch(tmp_path / "Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors")
    _touch(tmp_path / "Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/preprocessor_config.json")
    _touch(tmp_path / "Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/README.md")

    resolved = resolve_diffsynth_model_paths("wan22-s2v", model_root=tmp_path)

    assert not resolved.complete
    assert any(
        "audio_processor: missing required file(s)" in item and "vocab.json" in item
        for item in resolved.missing
    )


def test_load_diffsynth_s2v_loads_audio_processor_after_pipeline(monkeypatch, tmp_path):
    _make_wan22_image_common(tmp_path, "Wan2.2-S2V-14B")
    _touch(tmp_path / "Wan2.2-S2V-14B/diffusion_pytorch_model.safetensors")
    _touch(tmp_path / "Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors")
    _touch(tmp_path / "Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/preprocessor_config.json")
    _touch(tmp_path / "Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/vocab.json")
    calls = {}
    models = sys.modules["_infer_diffsynth.models"]

    class FakeModelConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeWanVideoPipeline:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.update(kwargs)
            return FakeWanVideoPipeline()

        def enable_vram_management(self):
            calls["enabled_vram_management"] = True

    fake_diffsynth = type(sys)("diffsynth")
    fake_diffsynth.ModelConfig = FakeModelConfig
    fake_pipeline_module = type(sys)("diffsynth.pipelines.wan_video")
    fake_pipeline_module.WanVideoPipeline = FakeWanVideoPipeline
    monkeypatch.setitem(sys.modules, "diffsynth", fake_diffsynth)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines", type(sys)("diffsynth.pipelines"))
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines.wan_video", fake_pipeline_module)
    monkeypatch.setattr(
        models,
        "_load_wav2vec2_processor",
        lambda path: f"processor:{Path(path).relative_to(tmp_path).as_posix()}",
    )

    pipe, resolved = load_diffsynth_pipeline("wan22-s2v-14b", model_root=tmp_path, torch_dtype="bf16", device="cpu")

    assert isinstance(pipe, FakeWanVideoPipeline)
    assert resolved.spec.key == "wan22-s2v-14b"
    assert [
        Path(config.kwargs["path"]).relative_to(tmp_path).as_posix()
        for config in calls["model_configs"]
    ] == [
        "Wan2.2-S2V-14B/diffusion_pytorch_model.safetensors",
        "Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors",
        "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.2-S2V-14B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ]
    assert calls["audio_processor_config"] is None
    assert pipe.audio_processor == "processor:Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english"
    assert calls["enabled_vram_management"] is True
    assert pipe._sparsevideo_model_key == "wan22-s2v-14b"


def test_resolve_diffsynth_wan22_a14b_dual_dit_components(tmp_path):
    _make_wan22_t2v_a14b_common(tmp_path)

    resolved = resolve_diffsynth_model_paths("wan22", model_root=tmp_path)

    assert resolved.complete
    assert resolved.spec.key == "wan22-t2v-a14b"
    assert resolved.components["dit_high_noise"] == (
        tmp_path / "Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model.safetensors",
    )
    assert resolved.components["dit_low_noise"] == (
        tmp_path / "Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model.safetensors",
    )


def test_load_diffsynth_wan22_a14b_uses_high_low_noise_config_order(monkeypatch, tmp_path):
    _make_wan22_t2v_a14b_common(tmp_path)
    calls = {}

    class FakeModelConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeWanVideoPipeline:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.update(kwargs)
            return FakeWanVideoPipeline()

        def enable_vram_management(self):
            calls["enabled_vram_management"] = True

    fake_diffsynth = type(sys)("diffsynth")
    fake_diffsynth.ModelConfig = FakeModelConfig
    fake_pipeline_module = type(sys)("diffsynth.pipelines.wan_video")
    fake_pipeline_module.WanVideoPipeline = FakeWanVideoPipeline
    monkeypatch.setitem(sys.modules, "diffsynth", fake_diffsynth)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines", type(sys)("diffsynth.pipelines"))
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines.wan_video", fake_pipeline_module)

    pipe, resolved = load_diffsynth_pipeline("wan22-t2v-a14b", model_root=tmp_path, torch_dtype="bf16", device="cpu")

    assert isinstance(pipe, FakeWanVideoPipeline)
    assert resolved.spec.key == "wan22-t2v-a14b"
    assert [
        Path(config.kwargs["path"]).relative_to(tmp_path).as_posix()
        for config in calls["model_configs"]
    ] == [
        "Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model.safetensors",
        "Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model.safetensors",
        "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ]
    assert calls["enabled_vram_management"] is True
    assert pipe._sparsevideo_model_key == "wan22-t2v-a14b"


def test_resolve_diffsynth_prefers_single_bf16_shard_set(tmp_path):
    _make_wan21_common(tmp_path)
    _touch(tmp_path / "Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    for idx in range(1, 4):
        _touch(tmp_path / f"Wan2.2-TI2V-5B/diffusion_pytorch_model-{idx:05d}-of-00003.safetensors")
        _touch(tmp_path / f"Wan2.2-TI2V-5B/diffusion_pytorch_model-{idx:05d}-of-00003-bf16.safetensors")

    resolved = resolve_diffsynth_model_paths("wan22-ti2v-5b", model_root=tmp_path)

    assert resolved.complete
    assert [path.name for path in resolved.components["dit"]] == [
        "diffusion_pytorch_model-00001-of-00003-bf16.safetensors",
        "diffusion_pytorch_model-00002-of-00003-bf16.safetensors",
        "diffusion_pytorch_model-00003-of-00003-bf16.safetensors",
    ]


def test_resolve_diffsynth_reports_incomplete_sharded_model(tmp_path):
    _make_wan21_common(tmp_path)
    _touch(tmp_path / "Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00003.safetensors")
    _touch(tmp_path / "Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00003.safetensors")

    resolved = resolve_diffsynth_model_paths("wan21-t2v-14b", model_root=tmp_path)

    assert not resolved.complete
    assert any("missing shard indexes 00002" in item for item in resolved.missing)


def test_resolve_diffsynth_reports_incomplete_sharded_variant(tmp_path):
    _make_wan21_common(tmp_path)
    _touch(tmp_path / "Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    _touch(tmp_path / "Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003-bf16.safetensors")
    _touch(tmp_path / "Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003-bf16.safetensors")

    resolved = resolve_diffsynth_model_paths("wan22-ti2v-5b", model_root=tmp_path)

    assert not resolved.complete
    assert any("missing shard indexes 00002" in item and "variant -bf16" in item for item in resolved.missing)


def test_resolve_diffsynth_ltx2_flat_model_layout(tmp_path):
    _make_ltx2_common(tmp_path, "LTX-2-Repackage")
    _touch(tmp_path / "LTX-2-Repackage/audio_vae_encoder.safetensors")
    _touch(tmp_path / "LTX-2-Repackage/audio_vae_decoder.safetensors")

    resolved = resolve_diffsynth_model_paths("ltx-2", model_root=tmp_path)

    assert resolved.complete
    assert resolved.spec.key == "ltx2"
    assert resolved.components["tokenizer"] == (
        tmp_path / "gemma-3-12b-it-qat-q4_0-unquantized",
    )
    assert resolved.components["text_encoder"] == (
        tmp_path / "gemma-3-12b-it-qat-q4_0-unquantized/model.safetensors",
    )
    assert resolved.components["dit"] == (tmp_path / "LTX-2-Repackage/transformer.safetensors",)
    assert resolved.components["audio_vae_encoder"] == (
        tmp_path / "LTX-2-Repackage/audio_vae_encoder.safetensors",
    )


def test_resolve_diffsynth_ltx2_rejects_incomplete_gemma_tokenizer_dir(tmp_path):
    _touch(tmp_path / "gemma-3-12b-it-qat-q4_0-unquantized/tokenizer.json")
    _touch(tmp_path / "gemma-3-12b-it-qat-q4_0-unquantized/model.safetensors")
    _touch(tmp_path / "LTX-2-Repackage/transformer.safetensors")
    _touch(tmp_path / "LTX-2-Repackage/video_vae_encoder.safetensors")
    _touch(tmp_path / "LTX-2-Repackage/video_vae_decoder.safetensors")
    _touch(tmp_path / "LTX-2-Repackage/audio_vocoder.safetensors")
    _touch(tmp_path / "LTX-2-Repackage/text_encoder_post_modules.safetensors")
    _touch(tmp_path / "LTX-2-Repackage/audio_vae_encoder.safetensors")
    _touch(tmp_path / "LTX-2-Repackage/audio_vae_decoder.safetensors")

    resolved = resolve_diffsynth_model_paths("ltx2", model_root=tmp_path)

    assert not resolved.complete
    assert any("tokenizer.model" in item for item in resolved.missing)
    assert any("preprocessor_config.json" in item for item in resolved.missing)
    assert any("processor_config.json" in item for item in resolved.missing)
    assert any("tokenizer_config.json" in item for item in resolved.missing)
    assert any("special_tokens_map.json" in item for item in resolved.missing)
    assert any("config.json" in item for item in resolved.missing)
    assert any("generation_config.json" in item for item in resolved.missing)


def test_resolve_diffsynth_ltx23_uses_source_checkpoint_without_repackage(tmp_path):
    _make_ltx2_text(tmp_path)
    _touch(tmp_path / "LTX-2.3/ltx-2.3-22b-dev.safetensors")
    _touch(tmp_path / "LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

    resolved = resolve_diffsynth_model_paths("ltx2.3", model_root=tmp_path)

    assert resolved.complete
    assert resolved.spec.key == "ltx23"
    assert resolved.components["source_checkpoint"] == (
        tmp_path / "LTX-2.3/ltx-2.3-22b-dev.safetensors",
    )
    assert resolved.components["latent_upsampler"] == (
        tmp_path / "LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    )
    assert "dit" not in resolved.components


def test_resolve_diffsynth_mova_uses_video_backbone_and_audio_components(tmp_path):
    _make_mova_common(tmp_path)

    resolved = resolve_diffsynth_model_paths("mova", model_root=tmp_path)

    assert resolved.complete
    assert resolved.spec.key == "mova-720p"
    assert resolved.components["mova_tokenizer"] == (tmp_path / "MOVA-720p/tokenizer",)
    assert resolved.components["dit"] == (
        tmp_path / "Wan2.1-T2V-14B/diffusion_pytorch_model.safetensors",
    )
    assert resolved.components["audio_dit"] == (
        tmp_path / "MOVA-720p/audio_dit/diffusion_pytorch_model.safetensors",
    )
    assert resolved.components["audio_vae"] == (
        tmp_path / "MOVA-720p/audio_vae/diffusion_pytorch_model.safetensors",
    )
    assert resolved.components["dual_tower_bridge"] == (
        tmp_path / "MOVA-720p/dual_tower_bridge/diffusion_pytorch_model.safetensors",
    )


def test_resolve_diffsynth_mova_rejects_incomplete_tokenizer_dir(tmp_path):
    _make_wan21_common(tmp_path)
    _touch(tmp_path / "Wan2.1-T2V-14B/diffusion_pytorch_model.safetensors")
    _touch(tmp_path / "MOVA-720p/tokenizer/tokenizer.json")
    _touch(tmp_path / "MOVA-720p/audio_dit/diffusion_pytorch_model.safetensors")
    _touch(tmp_path / "MOVA-720p/audio_vae/diffusion_pytorch_model.safetensors")
    _touch(tmp_path / "MOVA-720p/dual_tower_bridge/diffusion_pytorch_model.safetensors")

    resolved = resolve_diffsynth_model_paths("mova", model_root=tmp_path)

    assert not resolved.complete
    assert any("tokenizer_config.json" in item for item in resolved.missing)
    assert any("special_tokens_map.json" in item for item in resolved.missing)


def test_load_diffsynth_mova_uses_video_audio_component_model_configs(monkeypatch, tmp_path):
    _make_mova_common(tmp_path)
    calls = {}

    class FakeModelConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeMovaAudioVideoPipeline:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.update(kwargs)
            return FakeMovaAudioVideoPipeline()

        def enable_vram_management(self):
            calls["enabled_vram_management"] = True

    fake_diffsynth = type(sys)("diffsynth")
    fake_diffsynth.ModelConfig = FakeModelConfig
    fake_pipeline_module = type(sys)("diffsynth.pipelines.mova_audio_video")
    fake_pipeline_module.MovaAudioVideoPipeline = FakeMovaAudioVideoPipeline
    monkeypatch.setitem(sys.modules, "diffsynth", fake_diffsynth)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines", type(sys)("diffsynth.pipelines"))
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines.mova_audio_video", fake_pipeline_module)

    pipe, resolved = load_diffsynth_pipeline("mova-720p", model_root=tmp_path, torch_dtype="bf16", device="cpu")

    assert isinstance(pipe, FakeMovaAudioVideoPipeline)
    assert resolved.spec.key == "mova-720p"
    assert calls["tokenizer_config"].kwargs == {"path": str(tmp_path / "MOVA-720p/tokenizer")}
    assert [
        Path(config.kwargs["path"]).relative_to(tmp_path).as_posix()
        for config in calls["model_configs"]
    ] == [
        "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1-T2V-14B/diffusion_pytorch_model.safetensors",
        "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        "MOVA-720p/audio_dit/diffusion_pytorch_model.safetensors",
        "MOVA-720p/audio_vae/diffusion_pytorch_model.safetensors",
        "MOVA-720p/dual_tower_bridge/diffusion_pytorch_model.safetensors",
    ]
    assert calls["enabled_vram_management"] is True
    assert pipe._sparsevideo_model_key == "mova-720p"


def test_load_diffsynth_ltx2_uses_component_model_configs(monkeypatch, tmp_path):
    _make_ltx2_common(tmp_path, "LTX-2-Repackage")
    _touch(tmp_path / "LTX-2-Repackage/audio_vae_encoder.safetensors")
    _touch(tmp_path / "LTX-2-Repackage/audio_vae_decoder.safetensors")

    calls = {}

    class FakeModelConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLTX2AudioVideoPipeline:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.update(kwargs)
            return FakeLTX2AudioVideoPipeline()

        def enable_vram_management(self):
            calls["enabled_vram_management"] = True

    fake_diffsynth = type(sys)("diffsynth")
    fake_diffsynth.ModelConfig = FakeModelConfig
    fake_pipeline_module = type(sys)("diffsynth.pipelines.ltx2_audio_video")
    fake_pipeline_module.LTX2AudioVideoPipeline = FakeLTX2AudioVideoPipeline
    monkeypatch.setitem(sys.modules, "diffsynth", fake_diffsynth)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines", type(sys)("diffsynth.pipelines"))
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines.ltx2_audio_video", fake_pipeline_module)

    pipe, resolved = load_diffsynth_pipeline("ltx2", model_root=tmp_path, torch_dtype="bf16", device="cpu")

    assert isinstance(pipe, FakeLTX2AudioVideoPipeline)
    assert resolved.spec.key == "ltx2"
    assert calls["tokenizer_config"].kwargs == {
        "path": str(tmp_path / "gemma-3-12b-it-qat-q4_0-unquantized"),
    }
    assert [Path(config.kwargs["path"]).name for config in calls["model_configs"]] == [
        "model.safetensors",
        "transformer.safetensors",
        "video_vae_encoder.safetensors",
        "video_vae_decoder.safetensors",
        "audio_vae_decoder.safetensors",
        "audio_vocoder.safetensors",
        "audio_vae_encoder.safetensors",
        "text_encoder_post_modules.safetensors",
    ]
    assert calls["enabled_vram_management"] is True
    assert pipe._sparsevideo_model_key == "ltx2"


def test_load_diffsynth_ltx23_uses_source_checkpoint_without_repackage_duplicates(monkeypatch, tmp_path):
    _make_ltx2_text(tmp_path)
    _touch(tmp_path / "LTX-2.3/ltx-2.3-22b-dev.safetensors")
    _touch(tmp_path / "LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

    calls = {}

    class FakeModelConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLTX2AudioVideoPipeline:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.update(kwargs)
            return FakeLTX2AudioVideoPipeline()

        def enable_vram_management(self):
            calls["enabled_vram_management"] = True

    fake_diffsynth = type(sys)("diffsynth")
    fake_diffsynth.ModelConfig = FakeModelConfig
    fake_pipeline_module = type(sys)("diffsynth.pipelines.ltx2_audio_video")
    fake_pipeline_module.LTX2AudioVideoPipeline = FakeLTX2AudioVideoPipeline
    monkeypatch.setitem(sys.modules, "diffsynth", fake_diffsynth)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines", type(sys)("diffsynth.pipelines"))
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines.ltx2_audio_video", fake_pipeline_module)

    pipe, resolved = load_diffsynth_pipeline("ltx23", model_root=tmp_path, torch_dtype="bf16", device="cpu")

    assert isinstance(pipe, FakeLTX2AudioVideoPipeline)
    assert resolved.spec.key == "ltx23"
    assert [Path(config.kwargs["path"]).name for config in calls["model_configs"]] == [
        "model.safetensors",
        "ltx-2.3-22b-dev.safetensors",
        "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    ]
    assert all("LTX-2.3-Repackage" not in config.kwargs["path"] for config in calls["model_configs"])
    assert calls["enabled_vram_management"] is True
    assert pipe._sparsevideo_model_key == "ltx23"


def test_download_diffsynth_dry_run_skips_ltx23_repackage_duplicates(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--model-root",
            str(tmp_path),
            "ltx23",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "skip DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors" in result.stdout
    assert "duplicate large-model loading" in result.stdout
    assert "would download DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors" not in result.stdout
    assert "would download Lightricks/LTX-2.3:ltx-2.3-22b-dev.safetensors" in result.stdout


def test_download_diffsynth_dry_run_does_not_skip_incomplete_shards(tmp_path):
    _make_wan21_common(tmp_path)
    _touch(tmp_path / "Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00003.safetensors")
    _touch(tmp_path / "Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00003.safetensors")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--model-root",
            str(tmp_path),
            "wan21-t2v-14b",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "would download Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors" in result.stdout
    assert "would skip existing complete Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors" not in result.stdout


def test_download_diffsynth_dry_run_does_not_skip_incomplete_shard_variant(tmp_path):
    _make_wan21_common(tmp_path)
    _touch(tmp_path / "Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    _touch(tmp_path / "Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003-bf16.safetensors")
    _touch(tmp_path / "Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003-bf16.safetensors")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--model-root",
            str(tmp_path),
            "wan22-ti2v-5b",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "would download Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors" in result.stdout
    assert "would skip existing complete Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors" not in result.stdout


def test_download_diffsynth_dry_run_skips_two_level_nested_longcat_layout(tmp_path):
    _make_wan21_common(tmp_path)
    for idx in range(1, 7):
        _touch(
            tmp_path
            / "Longcat/weights/LongCat-Video/dit"
            / f"diffusion_pytorch_model-{idx:05d}-of-00006.safetensors"
        )

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--model-root",
            str(tmp_path),
            "longcat-video",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "would skip existing complete meituan-longcat/LongCat-Video:dit/diffusion_pytorch_model*.safetensors" in result.stdout
    assert "would download meituan-longcat/LongCat-Video:dit/diffusion_pytorch_model*.safetensors" not in result.stdout


def test_download_diffsynth_modelscope_first_dry_run_reports_hf_fallback(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--source",
            "modelscope-first",
            "--model-root",
            str(tmp_path),
            "wan21-t2v-14b",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "would download Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors via modelscope,huggingface" in result.stdout


def test_download_diffsynth_modelscope_first_keeps_modelscope_only_repos_on_modelscope(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--source",
            "modelscope-first",
            "--model-root",
            str(tmp_path),
            "wan21-t2v-14b",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    converted_repo = "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
    assert f"would download {converted_repo}:Wan2.1_VAE.safetensors via modelscope ->" in result.stdout
    assert f"would download {converted_repo}:Wan2.1_VAE.safetensors via modelscope,huggingface" not in result.stdout


def test_download_diffsynth_dry_run_includes_speedcontrol_motion_controller(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--source",
            "hf-first",
            "--model-root",
            str(tmp_path),
            "wan21-speedcontrol-1.3b",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        "would download DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1:model.safetensors "
        "via huggingface,modelscope"
    ) in result.stdout


def test_download_diffsynth_hf_first_keeps_pai_fun_repos_on_modelscope(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--source",
            "hf-first",
            "--model-root",
            str(tmp_path),
            "wan21-fun-1.3b-control",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        "would download PAI/Wan2.1-Fun-1.3B-Control:diffusion_pytorch_model*.safetensors "
        "via modelscope ->"
    ) in result.stdout
    assert (
        "would download PAI/Wan2.1-Fun-1.3B-Control:diffusion_pytorch_model*.safetensors "
        "via huggingface"
    ) not in result.stdout


def test_download_diffsynth_modelscope_first_keeps_hf_only_repos_on_hf(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--source",
            "modelscope-first",
            "--model-root",
            str(tmp_path),
            "video-as-prompt-wan21-14b",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "would download ByteDance/Video-As-Prompt-Wan2.1-14B:transformer/diffusion_pytorch_model*.safetensors via huggingface" in result.stdout


def test_download_diffsynth_defaults_to_hf_mirror_without_proxy(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "HTTP_PROXY": "http://127.0.0.1:10000",
            "HTTPS_PROXY": "http://127.0.0.1:10000",
            "ALL_PROXY": "http://127.0.0.1:10000",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--model-root",
            str(tmp_path),
            "wan21-t2v-14b",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "HF endpoint: https://hf-mirror.com" in result.stdout
    assert "Proxy: disabled" in result.stdout
    assert "Proxy: http://127.0.0.1:10000" not in result.stdout


def test_download_diffsynth_no_proxy_clears_proxy_env_for_hf_cli(tmp_path):
    model_root = tmp_path / "models"
    env_log = tmp_path / "env.log"
    fake_hf = tmp_path / "hf"
    _make_wan21_common(model_root)
    fake_hf.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
local_dir=
include=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-dir)
            local_dir=$2
            shift 2
            ;;
        --include)
            include=$2
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done
{
    printf 'HTTP_PROXY=%s\n' "${HTTP_PROXY-}"
    printf 'HTTPS_PROXY=%s\n' "${HTTPS_PROXY-}"
    printf 'ALL_PROXY=%s\n' "${ALL_PROXY-}"
    printf 'PROXY=%s\n' "${PROXY-}"
} >> "$ENV_LOG"
mkdir -p "$local_dir"
case "$include" in
    diffusion_pytorch_model*.safetensors)
        printf x > "$local_dir/diffusion_pytorch_model-00001-of-00001.safetensors"
        ;;
    *)
        printf 'unexpected include: %s\n' "$include" >&2
        exit 2
        ;;
esac
""",
        encoding="utf-8",
    )
    fake_hf.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "HF_CLI": str(fake_hf),
            "ENV_LOG": str(env_log),
            "HTTP_PROXY": "http://127.0.0.1:10000",
            "HTTPS_PROXY": "http://127.0.0.1:10000",
            "ALL_PROXY": "http://127.0.0.1:10000",
            "PROXY": "http://127.0.0.1:10000",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--source",
            "huggingface",
            "--model-root",
            str(model_root),
            "wan21-t2v-14b",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Proxy: disabled" in result.stdout
    assert env_log.read_text(encoding="utf-8") == "HTTP_PROXY=\nHTTPS_PROXY=\nALL_PROXY=\nPROXY=\n"


def test_download_diffsynth_huggingface_first_dry_run_reports_modelscope_fallback(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--source",
            "hf-first",
            "--model-root",
            str(tmp_path),
            "wan21-t2v-14b",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "would download Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors via huggingface,modelscope" in result.stdout
    assert "would download DiffSynth-Studio/Wan-Series-Converted-Safetensors:Wan2.1_VAE.safetensors via modelscope" in result.stdout


def test_download_diffsynth_link_root_reuses_nested_longcat_layout(tmp_path):
    model_root = tmp_path / "models"
    link_root = tmp_path / "existing"
    _make_wan21_common(model_root)
    for idx in range(1, 7):
        _touch(
            link_root
            / "Longcat/weights/LongCat-Video/dit"
            / f"diffusion_pytorch_model-{idx:05d}-of-00006.safetensors"
        )

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--no-proxy",
            "--model-root",
            str(model_root),
            "--link-root",
            str(link_root),
            "longcat-video",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    linked = model_root / "LongCat-Video"
    assert linked.is_symlink()
    assert linked.resolve() == (link_root / "Longcat/weights/LongCat-Video").resolve()
    assert "reuse linked meituan-longcat/LongCat-Video:dit/diffusion_pytorch_model*.safetensors" in result.stdout
    assert "download meituan-longcat/LongCat-Video:dit/diffusion_pytorch_model*.safetensors" not in result.stdout


def test_download_diffsynth_dry_run_does_not_skip_incomplete_wan_tokenizer(tmp_path):
    model_root = tmp_path / "models"
    (model_root / "Wan2.1-T2V-1.3B/google/umt5-xxl").mkdir(parents=True)

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--model-root",
            str(model_root),
            "wan21-t2v-1.3b",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "would download Wan-AI/Wan2.1-T2V-1.3B:google/umt5-xxl/tokenizer.json" in result.stdout
    assert "would download Wan-AI/Wan2.1-T2V-1.3B:google/umt5-xxl/tokenizer_config.json" in result.stdout
    assert "would download Wan-AI/Wan2.1-T2V-1.3B:google/umt5-xxl/spiece.model" in result.stdout
    assert "would skip existing complete Wan-AI/Wan2.1-T2V-1.3B:google/umt5-xxl/tokenizer.json" not in result.stdout


def test_download_diffsynth_verifies_pattern_after_successful_cli(tmp_path):
    model_root = tmp_path / "models"
    _make_wan21_common(model_root)
    fake_hf = tmp_path / "hf"
    fake_hf.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_hf.chmod(0o755)
    env = {
        **os.environ,
        "HF_CLI": str(fake_hf),
        "MAX_ATTEMPTS": "1",
        "BASE_DELAY": "0",
    }

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--no-proxy",
            "--source",
            "huggingface",
            "--model-root",
            str(model_root),
            "wan21-t2v-14b",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "download Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors via huggingface" in result.stdout
    assert "required pattern is still missing or incomplete" in result.stderr


def test_download_diffsynth_lists_explicit_s2v_wav2vec_components(tmp_path):
    model_root = tmp_path / "models"
    _make_wan22_image_common(model_root, "Wan2.2-S2V-14B")
    _touch(model_root / "Wan-Series-Converted-Safetensors/models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors")
    _touch(model_root / "Wan2.2-S2V-14B/diffusion_pytorch_model.safetensors")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--model-root",
            str(model_root),
            "wan22-s2v-14b",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wav2vec2-large-xlsr-53-english/model.safetensors" in result.stdout
    assert "wav2vec2-large-xlsr-53-english/preprocessor_config.json" in result.stdout
    assert "wav2vec2-large-xlsr-53-english/vocab.json" in result.stdout
    assert "wav2vec2-large-xlsr-53-english/special_tokens_map.json" in result.stdout
    assert not any(
        line.strip().endswith("Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/")
        for line in result.stdout.splitlines()
    )


def test_download_diffsynth_ltx_text_download_uses_explicit_patterns(tmp_path):
    model_root = tmp_path / "models"
    (model_root / "gemma-3-12b-it-qat-q4_0-unquantized").mkdir(parents=True)

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--model-root",
            str(model_root),
            "ltx2",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"{LTX_GEMMA_REPO}:tokenizer.json" in result.stdout
    assert f"{LTX_GEMMA_REPO}:tokenizer.model" in result.stdout
    assert f"{LTX_GEMMA_REPO}:tokenizer_config.json" in result.stdout
    assert f"{LTX_GEMMA_REPO}:preprocessor_config.json" in result.stdout
    assert f"{LTX_GEMMA_REPO}:processor_config.json" in result.stdout
    assert f"{LTX_GEMMA_REPO}:special_tokens_map.json" in result.stdout
    assert f"{LTX_GEMMA_REPO}:added_tokens.json" in result.stdout
    assert f"{LTX_GEMMA_REPO}:chat_template.json" in result.stdout
    assert f"{LTX_GEMMA_REPO}:config.json" in result.stdout
    assert f"{LTX_GEMMA_REPO}:generation_config.json" in result.stdout
    assert f"{LTX_GEMMA_REPO}:model.safetensors.index.json" in result.stdout
    assert f"{LTX_GEMMA_REPO}:model*.safetensors" in result.stdout
    assert f"{LTX_GEMMA_REPO}:*.json" not in result.stdout
    assert not any(
        line.strip() == f"- {LTX_GEMMA_REPO}:*"
        for line in result.stdout.splitlines()
    )


def test_download_diffsynth_mova_tokenizer_uses_explicit_files(tmp_path):
    model_root = tmp_path / "models"
    _touch(model_root / "MOVA-720p/tokenizer/tokenizer.json")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--model-root",
            str(model_root),
            "mova-720p",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "openmoss/MOVA-720p:tokenizer/tokenizer.json" in result.stdout
    assert "openmoss/MOVA-720p:tokenizer/tokenizer_config.json" in result.stdout
    assert "openmoss/MOVA-720p:tokenizer/special_tokens_map.json" in result.stdout
    assert not any(
        line.strip().endswith("openmoss/MOVA-720p:tokenizer/")
        for line in result.stdout.splitlines()
    )
    assert "would download openmoss/MOVA-720p:tokenizer/ " not in result.stdout
    assert "would skip existing complete openmoss/MOVA-720p:tokenizer/tokenizer.json" in result.stdout
    assert "would download openmoss/MOVA-720p:tokenizer/tokenizer_config.json" in result.stdout


def test_download_diffsynth_file_pattern_does_not_match_directory(tmp_path):
    model_root = tmp_path / "models"
    (model_root / "gemma-3-12b-it-qat-q4_0-unquantized/tokenizer_config").mkdir(parents=True)

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--model-root",
            str(model_root),
            "ltx2",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"would download {LTX_GEMMA_REPO}:tokenizer.json" in result.stdout
    assert f"would skip existing complete {LTX_GEMMA_REPO}:tokenizer.json" not in result.stdout


def test_download_diffsynth_dry_run_does_not_skip_empty_checkpoint_file(tmp_path):
    model_root = tmp_path / "models"
    _make_wan21_common(model_root)
    path = model_root / "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--model-root",
            str(model_root),
            "wan21-t2v-1.3b",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "would download Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors" in result.stdout
    assert "would skip existing complete Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors" not in result.stdout


def test_download_diffsynth_ltx_text_partial_json_dir_downloads_missing_configs(tmp_path):
    model_root = tmp_path / "models"
    _touch(model_root / "gemma-3-12b-it-qat-q4_0-unquantized/tokenizer.json")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--model-root",
            str(model_root),
            "ltx2",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"would skip existing complete {LTX_GEMMA_REPO}:tokenizer.json" in result.stdout
    assert f"would download {LTX_GEMMA_REPO}:tokenizer.model" in result.stdout
    assert f"would download {LTX_GEMMA_REPO}:tokenizer_config.json" in result.stdout
    assert f"would download {LTX_GEMMA_REPO}:preprocessor_config.json" in result.stdout
    assert f"would download {LTX_GEMMA_REPO}:processor_config.json" in result.stdout
    assert f"would download {LTX_GEMMA_REPO}:config.json" in result.stdout
    assert f"would download {LTX_GEMMA_REPO}:generation_config.json" in result.stdout


def test_infer_diffsynth_requires_i2v_input_image_for_generation():
    infer = _load_infer_module()
    args = infer.build_parser().parse_args(["--model", "wan21-i2v-14b-720p"])

    with pytest.raises(ValueError, match="requires --input-image"):
        infer._build_call_kwargs(args, "prompt")


def test_infer_diffsynth_requires_flf_first_and_last_images_for_generation(tmp_path):
    infer = _load_infer_module()
    args = infer.build_parser().parse_args(["--model", "wan21-flf2v-14b-720p"])

    with pytest.raises(ValueError, match="requires --input-image, --end-image"):
        infer._build_call_kwargs(args, "prompt")

    with_input = infer.build_parser().parse_args(
        [
            "--model",
            "wan21-flf2v-14b-720p",
            "--input-image",
            str(tmp_path / "first.png"),
        ]
    )
    with pytest.raises(ValueError, match="requires --end-image"):
        infer._build_call_kwargs(with_input, "prompt")


def test_infer_diffsynth_requires_s2v_audio_for_generation():
    infer = _load_infer_module()
    args = infer.build_parser().parse_args(["--model", "wan22-s2v-14b"])

    with pytest.raises(ValueError, match="requires --input-audio"):
        infer._build_call_kwargs(args, "prompt")


def test_infer_diffsynth_rejects_shape_that_diffsynth_would_round():
    infer = _load_infer_module()
    wan_bad_frames = infer.build_parser().parse_args(["--model", "wan21-t2v-1.3b", "--num-frames", "82"])
    ltx_bad_width = infer.build_parser().parse_args(["--model", "ltx2", "--width", "770"])

    with pytest.raises(ValueError, match=r"num_frames % 4 == 1"):
        infer._build_call_kwargs(wan_bad_frames, "prompt")
    with pytest.raises(ValueError, match="height/width multiples of 32"):
        infer._build_call_kwargs(ltx_bad_width, "prompt")


def test_infer_diffsynth_builds_s2v_audio_and_video_kwargs(monkeypatch, tmp_path):
    infer = _load_infer_module()
    audio = tmp_path / "speech.wav"
    pose = tmp_path / "pose"
    motion = tmp_path / "motion"
    audio_calls = {}

    def fake_load_audio(path, *, start_time=0.0, duration=None, as_numpy=True):
        audio_calls.update(
            {
                "path": Path(path),
                "start_time": start_time,
                "duration": duration,
                "as_numpy": as_numpy,
            }
        )
        return [0.0, 1.0], 22050

    monkeypatch.setattr(infer, "_load_audio_input", fake_load_audio)
    monkeypatch.setattr(
        infer,
        "_load_video_frames",
        lambda path, *, height, width: [f"{Path(path).name}:{height}x{width}"],
    )

    args = infer.build_parser().parse_args(
        [
            "--model",
            "wan22-s2v-14b",
            "--input-audio",
            str(audio),
            "--s2v-pose-video",
            str(pose),
            "--motion-video",
            str(motion),
            "--height",
            "128",
            "--width",
            "256",
            "--audio-start-time",
            "1.5",
            "--audio-duration",
            "2.0",
        ]
    )

    kwargs = infer._build_call_kwargs(args, "prompt")

    assert audio_calls == {
        "path": audio,
        "start_time": 1.5,
        "duration": 2.0,
        "as_numpy": True,
    }
    assert kwargs["input_audio"] == [0.0, 1.0]
    assert kwargs["audio_sample_rate"] == 22050
    assert kwargs["s2v_pose_video"] == ["pose:128x256"]
    assert kwargs["motion_video"] == ["motion:128x256"]


def test_infer_diffsynth_audio_input_uses_torchaudio_when_torchcodec_is_missing(monkeypatch, tmp_path):
    pytest.importorskip("diffsynth")
    import torch
    import diffsynth.utils.data.audio as audio_mod

    infer = _load_infer_module()
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"fake")
    calls = {}

    def fake_read_audio(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'torchcodec'", name="torchcodec")

    fake_torchaudio = type(sys)("torchaudio")
    fake_torchaudio.info = lambda path: type("Info", (), {"sample_rate": 22050})()

    def fake_load(path, *, frame_offset=0, num_frames=-1):
        calls["load"] = {
            "path": path,
            "frame_offset": frame_offset,
            "num_frames": num_frames,
        }
        return torch.ones(1, 16), 22050

    fake_torchaudio.load = fake_load
    monkeypatch.setattr(audio_mod, "read_audio", fake_read_audio)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)

    waveform, sample_rate = infer._load_audio_input(
        audio,
        start_time=1.5,
        duration=2.0,
        as_numpy=False,
    )

    assert tuple(waveform.shape) == (1, 16)
    assert sample_rate == 22050
    assert calls["load"] == {
        "path": str(audio),
        "frame_offset": 33075,
        "num_frames": 44100,
    }


def test_infer_diffsynth_requires_vap_video_and_builds_kwargs(monkeypatch, tmp_path):
    infer = _load_infer_module()

    missing = infer.build_parser().parse_args(["--model", "video-as-prompt-wan21-14b"])
    with pytest.raises(ValueError, match="requires --vap-video"):
        infer._build_call_kwargs(missing, "prompt")

    monkeypatch.setattr(
        infer,
        "_load_video_frames",
        lambda path, *, height, width: [f"{Path(path).name}:{height}x{width}"],
    )
    args = infer.build_parser().parse_args(
        [
            "--model",
            "video-as-prompt-wan21-14b",
            "--vap-video",
            str(tmp_path / "vap.mp4"),
            "--vap-prompt",
            "vap prompt",
            "--negative-vap-prompt",
            "bad vap",
        ]
    )

    kwargs = infer._build_call_kwargs(args, "prompt")

    assert kwargs["vap_video"] == ["vap.mp4:480x832"]
    assert kwargs["vap_prompt"] == "vap prompt"
    assert kwargs["negative_vap_prompt"] == "bad vap"


def test_infer_diffsynth_builds_vace_mask_as_repeated_image_and_wan_options(monkeypatch, tmp_path):
    pytest.importorskip("diffsynth")
    from diffsynth.pipelines.wan_video import WanVideoPipeline

    infer = _load_infer_module()

    class FakeImage:
        def __init__(self, name):
            self.name = name

        def copy(self):
            return f"copy:{self.name}"

    def fake_load_image(path):
        name = Path(path).name
        if name == "mask.png":
            return FakeImage(name)
        return f"image:{name}"

    monkeypatch.setattr(infer, "_load_image", fake_load_image)
    monkeypatch.setattr(
        infer,
        "_load_video_frames",
        lambda path, *, height, width: [f"{Path(path).name}:{height}x{width}"],
    )
    args = infer.build_parser().parse_args(
        [
            "--model",
            "wan21-vace-14b",
            "--vace-video",
            str(tmp_path / "vace.mp4"),
            "--vace-video-mask",
            str(tmp_path / "mask.png"),
            "--tile-size",
            "16,32",
            "--tile-stride",
            "[8,16]",
            "--camera-control-direction",
            "Left",
            "--camera-control-speed",
            "0.25",
            "--camera-control-origin",
            "0,1,2",
            "--cfg-merge",
            "--motion-bucket-id",
            "9",
            "--sliding-window-size",
            "17",
            "--sliding-window-stride",
            "5",
            "--tea-cache-l1-thresh",
            "0.2",
            "--tea-cache-model-id",
            "wan-test",
            "--wantodance-music-path",
            str(tmp_path / "music.wav"),
            "--wantodance-reference-image",
            str(tmp_path / "dance.png"),
            "--wantodance-fps",
            "30",
            "--wantodance-keyframes",
            str(tmp_path / "keyframes"),
            "--wantodance-keyframes-mask",
            "1,0,1",
            "--framewise-decoding",
            "--output-type",
            "floatpoint",
        ]
    )

    kwargs = infer._build_call_kwargs(args, "prompt")

    assert kwargs["vace_video"] == ["vace.mp4:480x832"]
    assert kwargs["vace_video_mask"] == ["copy:mask.png"] * 81
    assert kwargs["tile_size"] == (16, 32)
    assert kwargs["tile_stride"] == (8, 16)
    assert kwargs["camera_control_direction"] == "Left"
    assert kwargs["camera_control_speed"] == 0.25
    assert kwargs["camera_control_origin"] == (0.0, 1.0, 2.0)
    assert kwargs["cfg_merge"] is True
    assert kwargs["motion_bucket_id"] == 9
    assert kwargs["sliding_window_size"] == 17
    assert kwargs["sliding_window_stride"] == 5
    assert kwargs["tea_cache_l1_thresh"] == 0.2
    assert kwargs["tea_cache_model_id"] == "wan-test"
    assert kwargs["wantodance_music_path"] == str(tmp_path / "music.wav")
    assert kwargs["wantodance_reference_image"] == "image:dance.png"
    assert kwargs["wantodance_fps"] == 30
    assert kwargs["wantodance_keyframes"] == ["keyframes:480x832"]
    assert kwargs["wantodance_keyframes_mask"] == [1, 0, 1]
    assert kwargs["framewise_decoding"] is True
    assert kwargs["output_type"] == "floatpoint"
    assert set(kwargs) <= _call_signature_params(WanVideoPipeline)


def test_infer_diffsynth_builds_vace_mask_video_as_frames(monkeypatch, tmp_path):
    infer = _load_infer_module()
    calls = []

    def fake_load_video_frames(path, *, height, width, num_frames=None):
        calls.append((Path(path).name, height, width, num_frames))
        return [f"{Path(path).name}:{height}x{width}:{num_frames}"]

    monkeypatch.setattr(infer, "_load_video_frames", fake_load_video_frames)
    args = infer.build_parser().parse_args(
        [
            "--model",
            "wan21-vace-14b",
            "--vace-video",
            str(tmp_path / "vace.mp4"),
            "--vace-video-mask",
            str(tmp_path / "mask.mp4"),
        ]
    )

    kwargs = infer._build_call_kwargs(args, "prompt")

    assert kwargs["vace_video"] == ["vace.mp4:480x832:81"]
    assert kwargs["vace_video_mask"] == ["mask.mp4:480x832:81"]
    assert calls == [
        ("mask.mp4", 480, 832, 81),
        ("vace.mp4", 480, 832, 81),
    ]


def test_infer_diffsynth_requires_longcat_video_for_generation():
    infer = _load_infer_module()
    args = infer.build_parser().parse_args(["--model", "longcat-video"])

    with pytest.raises(ValueError, match="requires --longcat-video"):
        infer._build_call_kwargs(args, "prompt")


def test_infer_diffsynth_cli_validates_required_media_before_model_resolution(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "longcat-video",
            "--model-root",
            str(tmp_path),
            "--method",
            "svg2",
            "--print-json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "requires --longcat-video" in result.stdout
    assert "incomplete under" not in result.stdout
    assert "missing files" not in result.stdout


def test_infer_diffsynth_cli_validates_bad_media_path_before_model_resolution(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "longcat-video",
            "--model-root",
            str(tmp_path),
            "--method",
            "svg2",
            "--longcat-video",
            str(tmp_path / "missing.mp4"),
            "--print-json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "input path(s) do not exist" in result.stdout
    assert "--longcat-video=" in result.stdout
    assert "incomplete under" not in result.stdout
    assert "missing files" not in result.stdout


def test_infer_diffsynth_cli_validates_optional_media_path_before_model_resolution(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "wan21-t2v-1.3b",
            "--model-root",
            str(tmp_path),
            "--method",
            "svg2",
            "--input-video",
            str(tmp_path / "missing.mp4"),
            "--print-json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "input path(s) do not exist" in result.stdout
    assert "--input-video=" in result.stdout
    assert "incomplete under" not in result.stdout
    assert "missing files" not in result.stdout


def test_infer_diffsynth_cli_validates_bad_shape_before_model_resolution(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "wan21-t2v-1.3b",
            "--model-root",
            str(tmp_path),
            "--method",
            "svg2",
            "--num-frames",
            "82",
            "--print-json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "num_frames % 4 == 1" in result.stdout
    assert "incomplete under" not in result.stdout
    assert "missing files" not in result.stdout


def test_infer_diffsynth_records_incomplete_resolved_model_before_loading(monkeypatch, tmp_path):
    infer = _load_infer_module()
    resolved_model = {
        "model": "wan21-t2v-1.3b",
        "complete": False,
        "components": {},
        "missing": ["dit: missing files"],
    }

    class FakeResolved:
        complete = False
        model_root = tmp_path
        missing = ("dit: missing files",)

        class spec:
            key = "wan21-t2v-1.3b"

        def as_dict(self):
            return resolved_model

    payloads = []
    monkeypatch.setattr(infer, "resolve_diffsynth_model_paths", lambda *args, **kwargs: FakeResolved())
    monkeypatch.setattr(
        infer,
        "load_diffsynth_pipeline",
        lambda *args, **kwargs: pytest.fail("incomplete inference should not load pipeline"),
    )
    monkeypatch.setattr(infer, "_emit_payload", lambda args, payload: payloads.append(dict(payload)))

    rc = infer.main(["--apply-only", "--model-root", str(tmp_path)])

    assert rc == 1
    assert payloads[0]["status"] == "failed"
    assert payloads[0]["error_type"] == "FileNotFoundError"
    assert payloads[0]["resolved_model"] == resolved_model


def test_infer_diffsynth_cli_records_incomplete_resolved_model(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "wan21-t2v-1.3b",
            "--model-root",
            str(tmp_path),
            "--apply-only",
            "--print-json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert payload["status"] == "failed"
    assert payload["error_type"] == "FileNotFoundError"
    assert payload["resolved_model"]["complete"] is False
    assert payload["resolved_model"]["missing"]


def test_infer_diffsynth_sparse_summary_validation_rejects_no_patch_or_no_sparse_dispatch():
    infer = _load_infer_module()
    sparse_summary = {
        "pipeline_backend": "diffsynth",
        "diffsynth_version": "2.0.12",
        "num_self_attn_layers": 30,
        "patched_attention_count": 30,
        "patched_attention_paths": [f"dit.blocks.{idx}.self_attn.attn" for idx in range(30)],
        "method_runtime": {"dispatch_counts": {"sparse": 60}},
    }

    infer._validate_sparse_apply_summary("svg2", sparse_summary)
    infer._validate_sparse_generate_summary("svg2", sparse_summary)
    infer._validate_sparse_backend_summary(
        "svg2",
        {
            **sparse_summary,
            "method_runtime": {
                "dispatch_counts": {"sparse": 60},
                "backend_counts": {"flashinfer": 60},
            },
        },
    )
    infer._validate_sparse_apply_summary(
        "dense",
        {
            "pipeline_backend": "diffsynth",
            "diffsynth_version": "2.0.12",
            "num_self_attn_layers": 30,
            "patched_attention_count": 0,
            "patched_attention_paths": [],
        },
    )
    infer._validate_sparse_generate_summary(
        "dense",
        {
            "pipeline_backend": "diffsynth",
            "diffsynth_version": "2.0.12",
            "num_self_attn_layers": 30,
            "patched_attention_count": 0,
            "patched_attention_paths": [],
            "method_runtime": {"dispatch_counts": {}},
        },
    )

    with pytest.raises(RuntimeError, match="expected pipeline_backend"):
        infer._validate_sparse_apply_summary("svg2", {"pipeline_backend": "diffusers", "patched_attention_count": 30})

    with pytest.raises(RuntimeError, match="dense inference unexpectedly patched"):
        infer._validate_sparse_apply_summary(
            "dense",
            {
                "pipeline_backend": "diffsynth",
                "num_self_attn_layers": 30,
                "patched_attention_count": 1,
            },
        )

    with pytest.raises(RuntimeError, match="did not patch"):
        infer._validate_sparse_apply_summary(
            "svg2",
            {
                "pipeline_backend": "diffsynth",
                "num_self_attn_layers": 30,
                "patched_attention_count": 0,
            },
        )

    with pytest.raises(RuntimeError, match="did not report"):
        infer._validate_sparse_apply_summary("svg2", {"pipeline_backend": "diffsynth", "patched_attention_count": 30})

    with pytest.raises(RuntimeError, match="patched only part"):
        infer._validate_sparse_apply_summary(
            "svg2",
            {
                "pipeline_backend": "diffsynth",
                "num_self_attn_layers": 30,
                "patched_attention_count": 29,
            },
        )

    with pytest.raises(RuntimeError, match="patch path count"):
        infer._validate_sparse_apply_summary(
            "svg2",
            {
                "pipeline_backend": "diffsynth",
                "num_self_attn_layers": 30,
                "patched_attention_count": 30,
                "patched_attention_paths": ["dit.blocks.0.self_attn.attn"],
            },
        )

    with pytest.raises(RuntimeError, match="did not record the installed diffsynth version"):
        infer._validate_sparse_apply_summary(
            "svg2",
            {
                "pipeline_backend": "diffsynth",
                "num_self_attn_layers": 30,
                "patched_attention_count": 30,
                "patched_attention_paths": [f"dit.blocks.{idx}.self_attn.attn" for idx in range(30)],
            },
        )

    with pytest.raises(RuntimeError, match="without sparse dispatch"):
        infer._validate_sparse_generate_summary(
            "svg2",
            {
                "pipeline_backend": "diffsynth",
                "diffsynth_version": "2.0.12",
                "num_self_attn_layers": 30,
                "patched_attention_count": 30,
                "patched_attention_paths": [f"dit.blocks.{idx}.self_attn.attn" for idx in range(30)],
                "method_runtime": {"dispatch_counts": {"dense": 60}},
            },
        )

    with pytest.raises(RuntimeError, match="without backend evidence"):
        infer._validate_sparse_backend_summary(
            "svg2",
            sparse_summary,
        )

    with pytest.raises(RuntimeError, match="debug fallback"):
        infer._validate_sparse_backend_summary(
            "svg2",
            {
                **sparse_summary,
                "method_runtime": {
                    "dispatch_counts": {"sparse": 60},
                    "backend_counts": {"triton_debug_fallback": 60},
                },
            },
        )

    with pytest.raises(RuntimeError, match="unexpected sparse backend"):
        infer._validate_sparse_backend_summary(
            "svg2",
            {
                **sparse_summary,
                "method_runtime": {
                    "dispatch_counts": {"sparse": 60},
                    "backend_counts": {"torch_sdpa": 60},
                },
            },
        )


def test_infer_diffsynth_restores_when_apply_validation_fails(monkeypatch, tmp_path):
    infer = _load_infer_module()

    class FakeResolved:
        complete = True

        def as_dict(self):
            return {"complete": True}

    class FakeHandle:
        def __init__(self):
            self.restored = False

        def summary(self):
            return {
                "pipeline_backend": "diffsynth",
                "diffsynth_version": "2.0.12",
                "model_key": "wan21-t2v-1.3b",
                "num_self_attn_layers": 30,
                "patched_attention_count": 0,
                "patched_attention_paths": [],
                "restored": self.restored,
            }

        def restore(self):
            self.restored = True

    handle = FakeHandle()
    payloads = []
    monkeypatch.setattr(infer, "resolve_diffsynth_model_paths", lambda *args, **kwargs: FakeResolved())
    monkeypatch.setattr(infer, "load_diffsynth_pipeline", lambda *args, **kwargs: (object(), FakeResolved()))
    monkeypatch.setattr(infer, "apply_sparse_attention", lambda *args, **kwargs: handle)
    monkeypatch.setattr(infer, "_emit_payload", lambda args, payload: payloads.append(dict(payload)))

    rc = infer.main(["--apply-only", "--method", "svg2", "--model-root", str(tmp_path)])

    assert rc == 1
    assert handle.restored is True
    assert payloads[0]["status"] == "failed"
    assert "did not patch" in payloads[0]["error"]
    assert payloads[0]["restore_summary"]["restored"] is True


def test_infer_diffsynth_apply_only_records_method_config_timing_and_cuda(monkeypatch, tmp_path):
    infer = _load_infer_module()

    class FakeResolved:
        complete = True

        def as_dict(self):
            return {"model": "wan21-t2v-1.3b", "complete": True, "missing": []}

    class FakeHandle:
        def __init__(self):
            self.restored = False

        def summary(self):
            return {
                "pipeline_backend": "diffsynth",
                "diffsynth_version": "2.0.12",
                "model_key": "wan21-t2v-1.3b",
                "num_self_attn_layers": 30,
                "patched_attention_count": 30,
                "patched_attention_paths": [f"dit.blocks.{idx}.self_attn.attn" for idx in range(30)],
                "restored": self.restored,
            }

        def restore(self):
            self.restored = True

    payloads = []
    monkeypatch.setattr(infer, "resolve_diffsynth_model_paths", lambda *args, **kwargs: FakeResolved())
    monkeypatch.setattr(infer, "load_diffsynth_pipeline", lambda *args, **kwargs: (object(), FakeResolved()))
    monkeypatch.setattr(infer, "apply_sparse_attention", lambda *args, **kwargs: FakeHandle())
    monkeypatch.setattr(infer, "_reset_cuda_memory", lambda device: None)
    monkeypatch.setattr(infer, "_cuda_memory", lambda device: {"device": device, "available": False})
    monkeypatch.setattr(infer, "_emit_payload", lambda args, payload: payloads.append(dict(payload)))

    rc = infer.main(
        [
            "--apply-only",
            "--method",
            "svg2",
            "--method-config",
            "num_q_centroids=3",
            "--method-config",
            "dense_warmup_step_ratio=0",
            "--device",
            "cpu",
            "--model-root",
            str(tmp_path),
        ]
    )

    payload = payloads[0]
    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["mode"] == "apply_only"
    assert payload["method_config"]["num_q_centroids"] == 3
    assert payload["method_config"]["dense_warmup_step_ratio"] == 0
    assert "allow_triton_fallback" not in payload["method_config"]
    assert payload["sparse_attention_handle"] == payload["apply_summary"]
    assert payload["sparse_attention_handle_after_restore"] == payload["restore_summary"]
    assert payload["timings"]["load_apply_sec"] >= 0
    assert payload["cuda"] == {"device": "cpu", "available": False}
    assert payload["restore_summary"]["restored"] is True


def test_infer_diffsynth_build_method_config_does_not_add_fallback_keys():
    infer = _load_infer_module()
    args = infer.build_parser().parse_args(
        [
            "--method",
            "svg2",
        ]
    )
    spec = infer.get_diffsynth_model_spec(args.model)

    method_config = infer._build_method_config(args, spec)

    assert "allow_triton_fallback" not in method_config


def test_infer_diffsynth_cpu_offload_flags_map_to_diffsynth_vram_management():
    infer = _load_infer_module()

    args = infer.build_parser().parse_args(["--cpu-offload", "--cpu-offload-mode", "model"])
    runtime_options = infer._resolve_runtime_options(args)

    assert runtime_options == {
        "cpu_offload": True,
        "cpu_offload_mode": "model",
        "offload_device": "cpu",
        "enable_vram_management": True,
    }


def test_infer_diffsynth_rejects_unsupported_sequential_cpu_offload_mode():
    infer = _load_infer_module()
    args = infer.build_parser().parse_args(["--cpu-offload", "--cpu-offload-mode", "sequential"])

    with pytest.raises(ValueError, match="sequential is not supported"):
        infer._resolve_runtime_options(args)


def test_infer_diffsynth_cpu_offload_flags_are_passed_to_loader(monkeypatch, tmp_path):
    infer = _load_infer_module()
    load_calls = {}

    class FakeResolved:
        complete = True

        def as_dict(self):
            return {"model": "wan21-t2v-1.3b", "complete": True, "missing": []}

    class FakeHandle:
        def __init__(self):
            self.restored = False

        def summary(self):
            return {
                "pipeline_backend": "diffsynth",
                "diffsynth_version": "2.0.12",
                "model_key": "wan21-t2v-1.3b",
                "num_self_attn_layers": 30,
                "patched_attention_count": 0,
                "patched_attention_paths": [],
                "restored": self.restored,
            }

        def restore(self):
            self.restored = True

    def fake_load_pipeline(*args, **kwargs):
        load_calls.update(kwargs)
        return object(), FakeResolved()

    payloads = []
    monkeypatch.setattr(infer, "resolve_diffsynth_model_paths", lambda *args, **kwargs: FakeResolved())
    monkeypatch.setattr(infer, "load_diffsynth_pipeline", fake_load_pipeline)
    monkeypatch.setattr(infer, "apply_sparse_attention", lambda *args, **kwargs: FakeHandle())
    monkeypatch.setattr(infer, "_reset_cuda_memory", lambda device: None)
    monkeypatch.setattr(infer, "_cuda_memory", lambda device: {"device": device, "available": False})
    monkeypatch.setattr(infer, "_emit_payload", lambda args, payload: payloads.append(dict(payload)))

    rc = infer.main(
        [
            "--apply-only",
            "--method",
            "dense",
            "--device",
            "cpu",
            "--model-root",
            str(tmp_path),
            "--cpu-offload",
            "--cpu-offload-mode",
            "model",
            "--vram-limit",
            "40",
        ]
    )

    payload = payloads[0]
    assert rc == 0
    assert load_calls["offload_device"] == "cpu"
    assert load_calls["vram_limit"] == 40
    assert load_calls["enable_vram_management"] is True
    assert payload["cpu_offload"] is True
    assert payload["cpu_offload_mode"] == "model"
    assert payload["offload_device"] == "cpu"
    assert payload["vram_limit"] == 40
    assert payload["vram_management"] is True


def test_infer_diffsynth_generation_uses_pipeline_audio_sample_rate(monkeypatch, tmp_path):
    infer = _load_infer_module()
    captured = {}

    class FakeResolved:
        complete = True

        def as_dict(self):
            return {"model": "ltx2", "complete": True, "missing": []}

    class FakePipe:
        audio_vocoder = type("Vocoder", (), {"output_sampling_rate": 24000})()

        def __call__(self, **kwargs):
            captured["call_kwargs"] = kwargs
            return ([object()], object())

    class FakeHandle:
        def __init__(self):
            self.restored = False

        def summary(self):
            return {
                "pipeline_backend": "diffsynth",
                "diffsynth_version": "2.0.12",
                "model_key": "ltx2",
                "num_self_attn_layers": 1,
                "patched_attention_count": 0,
                "patched_attention_paths": [],
                "method_runtime": {"dispatch_counts": {}, "backend_counts": {}},
                "restored": self.restored,
            }

        def restore(self):
            self.restored = True

    def fake_save_output(output, output_file, *, fps, quality, audio_sample_rate):
        captured["output"] = output
        captured["audio_sample_rate"] = audio_sample_rate
        return {
            "output_file": str(output_file),
            "output_type": "video_audio",
            "audio_sample_rate": audio_sample_rate,
        }

    payloads = []
    monkeypatch.setattr(infer, "resolve_diffsynth_model_paths", lambda *args, **kwargs: FakeResolved())
    monkeypatch.setattr(infer, "load_diffsynth_pipeline", lambda *args, **kwargs: (FakePipe(), FakeResolved()))
    monkeypatch.setattr(infer, "apply_sparse_attention", lambda *args, **kwargs: FakeHandle())
    monkeypatch.setattr(infer, "save_diffsynth_output", fake_save_output)
    monkeypatch.setattr(infer, "_reset_cuda_memory", lambda device: None)
    monkeypatch.setattr(infer, "_cuda_memory", lambda device: {"device": device, "available": False})
    monkeypatch.setattr(infer, "_emit_payload", lambda args, payload: payloads.append(dict(payload)))

    rc = infer.main(
        [
            "--model",
            "ltx2",
            "--method",
            "dense",
            "--device",
            "cpu",
            "--model-root",
            str(tmp_path),
            "--output-file",
            str(tmp_path / "out.mp4"),
        ]
    )

    payload = payloads[0]
    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["mode"] == "generate"
    assert payload["model_arg"] == "ltx2"
    assert payload["height"] == 512
    assert payload["width"] == 768
    assert payload["num_frames"] == 121
    assert payload["fps"] == 24
    assert payload["num_inference_steps"] == 50
    assert payload["seed"] == 0
    assert payload["sparse_attention_handle"] == payload["generate_summary"]
    assert payload["sparse_attention_handle_after_restore"] == payload["restore_summary"]
    assert payload["audio_sample_rate"] == 24000
    assert captured["audio_sample_rate"] == 24000
    assert captured["call_kwargs"]["frame_rate"] == 24
    assert payload["restore_summary"]["restored"] is True


def test_infer_diffsynth_cuda_memory_uses_selected_device(monkeypatch):
    infer = _load_infer_module()
    calls = []

    fake_torch = type(sys)("torch")
    fake_torch.cuda = type(
        "Cuda",
        (),
        {
            "is_available": staticmethod(lambda: True),
            "reset_peak_memory_stats": staticmethod(lambda device=None: calls.append(("reset", device))),
            "max_memory_allocated": staticmethod(lambda device=None: calls.append(("allocated", device)) or 1024**3),
            "max_memory_reserved": staticmethod(lambda device=None: calls.append(("reserved", device)) or 2 * 1024**3),
        },
    )()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    infer._reset_cuda_memory("cuda:2")
    assert infer._cuda_memory("cuda:2") == {
        "device": "cuda:2",
        "available": True,
        "peak_allocated_gb": 1.0,
        "peak_reserved_gb": 2.0,
    }
    infer._reset_cuda_memory("cuda")

    assert calls == [
        ("reset", "cuda:2"),
        ("allocated", "cuda:2"),
        ("reserved", "cuda:2"),
        ("reset", None),
    ]


def test_infer_diffsynth_restore_summary_validation_rejects_unrestored_handle():
    infer = _load_infer_module()

    infer._validate_restore_summary({"restored": True})
    with pytest.raises(RuntimeError, match="did not restore"):
        infer._validate_restore_summary({"restored": False})


def test_infer_diffsynth_default_kwargs_match_installed_pipeline_signatures():
    pytest.importorskip("diffsynth")
    from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline
    from diffsynth.pipelines.mova_audio_video import MovaAudioVideoPipeline
    from diffsynth.pipelines.wan_video import WanVideoPipeline

    infer = _load_infer_module()
    pipeline_classes = {
        "wan21-t2v-1.3b": WanVideoPipeline,
        "mova-720p": MovaAudioVideoPipeline,
        "ltx2": LTX2AudioVideoPipeline,
    }
    for model, pipeline_cls in pipeline_classes.items():
        args = infer.build_parser().parse_args(["--model", model])
        kwargs = infer._build_call_kwargs(args, "prompt")
        assert set(kwargs) <= _call_signature_params(pipeline_cls)


def test_infer_diffsynth_wan22_a14b_defaults_match_diffusers_benchmark_shape():
    infer = _load_infer_module()
    for model in ("wan22-t2v-a14b", "wan22-i2v-a14b"):
        args_list = ["--model", model]
        if model == "wan22-i2v-a14b":
            args_list += ["--input-image", "example/i2v/1.jpg"]
        args = infer.build_parser().parse_args(args_list)

        kwargs = infer._build_call_kwargs(args, "prompt")

        assert kwargs["height"] == 480
        assert kwargs["width"] == 832
        assert kwargs["num_frames"] == 81
        assert kwargs["num_inference_steps"] == 40


def test_infer_diffsynth_krea_defaults_match_diffsynth_studio_example():
    pytest.importorskip("diffsynth")
    from diffsynth.pipelines.wan_video import WanVideoPipeline

    infer = _load_infer_module()
    args = infer.build_parser().parse_args(["--model", "krea-realtime-video"])

    kwargs = infer._build_call_kwargs(args, "prompt")

    assert kwargs["num_inference_steps"] == 6
    assert kwargs["cfg_scale"] == 1.0
    assert kwargs["sigma_shift"] == 20.0
    assert kwargs["height"] == 480
    assert kwargs["width"] == 832
    assert kwargs["num_frames"] == 81
    assert kwargs["tiled"] is True
    assert set(kwargs) <= _call_signature_params(WanVideoPipeline)


def test_infer_diffsynth_krea_sparse_config_uses_wan14b_backbone_defaults():
    infer = _load_infer_module()
    spec = get_diffsynth_model_spec("krea-realtime-video")
    args = infer.build_parser().parse_args(["--model", "krea-realtime-video", "--method", "sta"])

    method_config = infer._build_method_config(args, spec)

    assert method_config["mask_strategy_file_path"].endswith("mask_strategy_wan21_t2v_14b.json")


def test_infer_diffsynth_krea_cli_overrides_diffsynth_studio_defaults():
    infer = _load_infer_module()
    args = infer.build_parser().parse_args(
        [
            "--model",
            "krea-realtime-video",
            "--num-inference-steps",
            "8",
            "--cfg-scale",
            "2.5",
            "--sigma-shift",
            "7.5",
        ]
    )

    kwargs = infer._build_call_kwargs(args, "prompt")

    assert kwargs["num_inference_steps"] == 8
    assert kwargs["cfg_scale"] == 2.5
    assert kwargs["sigma_shift"] == 7.5


def test_installed_diffsynth_video_pipeline_surface_is_documented():
    pytest.importorskip("diffsynth")
    import diffsynth.pipelines as pipelines

    video_pipeline_classes = set()
    for module_info in pkgutil.iter_modules(pipelines.__path__):
        if "video" not in module_info.name:
            continue
        module = __import__(f"diffsynth.pipelines.{module_info.name}", fromlist=["*"])
        for name, obj in vars(module).items():
            if inspect.isclass(obj) and name.endswith("Pipeline") and name != "BasePipeline":
                video_pipeline_classes.add(f"{module_info.name}.{name}")

    assert video_pipeline_classes == {
        "ltx2_audio_video.LTX2AudioVideoPipeline",
        "mova_audio_video.MovaAudioVideoPipeline",
        "wan_video.WanVideoPipeline",
    }


def test_download_diffsynth_catalog_tracks_installed_video_model_config_examples(tmp_path):
    examples = _installed_diffsynth_video_model_config_examples()
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download" / "download_diffsynth_models.sh"),
            "--dry-run",
            "--no-proxy",
            "--model-root",
            str(tmp_path),
            "--all",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    dry_run = result.stdout
    deferred = {
        (spec.origin_repo, spec.origin_pattern)
        for spec in list_deferred_diffsynth_model_specs()
    }
    documented_alternatives = {
        ("Lightricks/LTX-2", "ltx-2-19b-dev.safetensors"): "DiffSynth-Studio/LTX-2-Repackage:transformer.safetensors",
        (
            "Wan-AI/Wan2.1-I2V-14B-480P",
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ): "DiffSynth-Studio/Wan-Series-Converted-Safetensors:models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
        (
            "Wan-AI/Wan2.1-T2V-14B",
            "models_t5_umt5-xxl-enc-bf16.pth",
        ): "DiffSynth-Studio/Wan-Series-Converted-Safetensors:models_t5_umt5-xxl-enc-bf16.safetensors",
        (
            "Wan-AI/Wan2.1-T2V-14B",
            "Wan2.1_VAE.pth",
        ): "DiffSynth-Studio/Wan-Series-Converted-Safetensors:Wan2.1_VAE.safetensors",
        (
            "Wan-AI/Wan2.2-TI2V-5B",
            "Wan2.2_VAE.pth",
        ): "DiffSynth-Studio/Wan-Series-Converted-Safetensors:Wan2.2_VAE.safetensors",
        (
            "google/gemma-3-12b-it-qat-q4_0-unquantized",
            "model-*.safetensors",
        ): f"{LTX_GEMMA_REPO}:model*.safetensors",
    }

    missing = []
    for repo, pattern in sorted(examples):
        token = f"{repo}:{pattern}"
        if token in dry_run:
            continue
        if (repo, pattern) in deferred:
            continue
        alternative = documented_alternatives.get((repo, pattern))
        if alternative is not None and alternative in dry_run:
            continue
        missing.append(token)

    assert missing == []


def test_infer_diffsynth_builds_mova_optional_kwargs_match_signature():
    pytest.importorskip("diffsynth")
    from diffsynth.pipelines.mova_audio_video import MovaAudioVideoPipeline

    infer = _load_infer_module()
    args = infer.build_parser().parse_args(
        [
            "--model",
            "mova-720p",
            "--fps",
            "12",
            "--sigma-shift",
            "4.5",
            "--switch-dit-boundary",
            "0.8",
            "--tile-size",
            "16,32",
            "--tile-stride",
            "8,16",
        ]
    )

    kwargs = infer._build_call_kwargs(args, "prompt")

    assert kwargs["frame_rate"] == 12
    assert kwargs["sigma_shift"] == 4.5
    assert kwargs["switch_DiT_boundary"] == 0.8
    assert kwargs["tile_size"] == (16, 32)
    assert kwargs["tile_stride"] == (8, 16)
    assert set(kwargs) <= _call_signature_params(MovaAudioVideoPipeline)


def test_infer_diffsynth_builds_ltx2_specific_kwargs(monkeypatch, tmp_path):
    pytest.importorskip("diffsynth")
    from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline

    infer = _load_infer_module()
    image = tmp_path / "image.png"
    video = tmp_path / "video.mp4"
    audio = tmp_path / "audio.wav"
    audio_calls = {}

    monkeypatch.setattr(infer, "_load_image", lambda path: f"image:{Path(path).name}")
    monkeypatch.setattr(
        infer,
        "_load_video_frames",
        lambda path, *, height, width: [f"{Path(path).name}:{height}x{width}"],
    )
    def fake_load_audio(path, *, start_time=0.0, duration=None, as_numpy=True):
        audio_calls.update(
            {
                "path": Path(path),
                "start_time": start_time,
                "duration": duration,
                "as_numpy": as_numpy,
            }
        )
        return [0.0], 48000

    monkeypatch.setattr(infer, "_load_audio_input", fake_load_audio)
    args = infer.build_parser().parse_args(
        [
            "--model",
            "ltx2",
            "--input-image",
            str(image),
            "--input-video",
            str(video),
            "--input-audio",
            str(audio),
            "--input-images-indexes",
            "0,10",
            "--input-images-strength",
            "0.75",
            "--in-context-video",
            str(tmp_path / "context_a.mp4"),
            "--in-context-video",
            str(tmp_path / "context_b.mp4"),
            "--in-context-downsample-factor",
            "4",
            "--retake-video-regions",
            "[[0.1, 0.4], [0.6, 0.9]]",
            "--retake-audio-regions",
            "0.2,0.8",
            "--tile-size-in-pixels",
            "256",
            "--tile-overlap-in-pixels",
            "64",
            "--tile-size-in-frames",
            "32",
            "--tile-overlap-in-frames",
            "8",
            "--use-two-stage-pipeline",
            "--stage2-spatial-upsample-factor",
            "3",
            "--clear-lora-before-stage-two",
            "--use-distilled-pipeline",
        ]
    )

    kwargs = infer._build_call_kwargs(args, "prompt")

    assert kwargs["frame_rate"] == 24
    assert kwargs["input_images"] == ["image:image.png"]
    assert kwargs["input_images_indexes"] == [0, 10]
    assert kwargs["input_images_strength"] == 0.75
    assert kwargs["retake_video"] == ["video.mp4:512x768"]
    assert kwargs["in_context_videos"] == [
        ["context_a.mp4:512x768"],
        ["context_b.mp4:512x768"],
    ]
    assert kwargs["in_context_downsample_factor"] == 4
    assert kwargs["retake_video_regions"] == [(0.1, 0.4), (0.6, 0.9)]
    assert kwargs["retake_audio_regions"] == [(0.2, 0.8)]
    assert audio_calls == {
        "path": audio,
        "start_time": 0.0,
        "duration": None,
        "as_numpy": False,
    }
    assert kwargs["retake_audio"] == [0.0]
    assert kwargs["audio_sample_rate"] == 48000
    assert kwargs["tile_size_in_pixels"] == 256
    assert kwargs["tile_overlap_in_pixels"] == 64
    assert kwargs["tile_size_in_frames"] == 32
    assert kwargs["tile_overlap_in_frames"] == 8
    assert kwargs["use_two_stage_pipeline"] is True
    assert kwargs["stage2_spatial_upsample_factor"] == 3
    assert kwargs["clear_lora_before_state_two"] is True
    assert kwargs["use_distilled_pipeline"] is True
    assert "input_image" not in kwargs
    assert "input_audio" not in kwargs
    assert "switch_DiT_boundary" not in kwargs
    assert "sigma_shift" not in kwargs
    assert "cfg_merge" not in kwargs
    assert "framewise_decoding" not in kwargs
    assert set(kwargs) <= _call_signature_params(LTX2AudioVideoPipeline)
    assert "output_type" not in kwargs


def test_infer_diffsynth_builds_mova_switch_boundary():
    infer = _load_infer_module()
    args = infer.build_parser().parse_args(
        [
            "--model",
            "mova-720p",
            "--switch-dit-boundary",
            "0.75",
        ]
    )

    kwargs = infer._build_call_kwargs(args, "prompt")

    assert kwargs["frame_rate"] == 24
    assert kwargs["switch_DiT_boundary"] == 0.75


def test_infer_diffsynth_dry_run_uses_separate_script_and_flat_layout(tmp_path):
    _make_wan21_common(tmp_path)
    _touch(tmp_path / "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model-root",
            str(tmp_path),
            "--model",
            "wan1.3b",
            "--dry-run",
            "--print-json",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["backend"] == "diffsynth"
    assert payload["status"] == "dry_run"
    assert payload["model"] == "wan21-t2v-1.3b"
    assert payload["resolved_model"]["complete"] is True


def test_infer_diffsynth_deferred_model_fails_as_json_without_traceback():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "wantodance",
            "--dry-run",
            "--print-json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert result.stderr == ""
    assert payload["backend"] == "diffsynth"
    assert payload["model"] == "wantodance"
    assert payload["status"] == "failed"
    assert payload["error_type"] == "ValueError"
    assert "deferred/local-only as 'wan22-dancer-14b'" in payload["error"]
