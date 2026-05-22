from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


DEFAULT_HEIGHT = 720
DEFAULT_WIDTH = 1280
DEFAULT_SEED = 0

STA_NATIVE_SEQ_SHAPES = {"18x48x80", "30x48x80", "36x48x48"}
STA_STRATEGY_SHAPES = {
    "wan21-t2v-1.3b": (50, 30, 12),
    "wan21-vace-1.3b": (50, 30, 12),
    "wan21-t2v-14b": (50, 40, 40),
    "wan21-i2v-14b": (50, 40, 40),
    "wan21-vace-14b": (50, 40, 40),
    "skyreels-v2-t2v-14b": (50, 40, 40),
    "skyreels-v2-i2v-14b": (50, 40, 40),
    "wan22-t2v-a14b": (40, 40, 40),
    "wan22-i2v-a14b": (40, 40, 40),
    "wan22-animate-14b": (20, 40, 40),
    "hunyuan-t2v": (50, 60, 24),
    "hunyuan-i2v": (50, 60, 24),
    "cogvideox-t2v": (50, 42, 48),
    "cogvideox-i2v": (50, 42, 48),
    "ltx-video": (50, 28, 32),
    "ltx-video-i2v": (50, 28, 32),
    "allegro": (100, 32, 24),
    "mochi-1": (64, 48, 24),
    "easyanimate-v5-t2v-12b": (50, 48, 48),
}
STA_UNSUPPORTED_STRATEGY_MODELS: dict = {}

FLASHOMNI_SPARSE_INFO_KEYS = (
    "sparse_info",
    "sparse_kv_info",
    "sparse_info_indptr",
    "sparse_kv_info_indptr",
)

WAN_SAMPLE_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

HUNYUAN_VIDEO_NEGATIVE_PROMPT = (
    "Aerial view, aerial view, overexposed, low quality, deformation, "
    "a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
)

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    pipeline_class: str
    hf_id: str
    local_dir: Optional[str]
    fps: int
    default_frames: int
    default_steps: int
    guidance_scale: float
    output_type: str
    sparse_supported: bool = True
    sparse_methods: Optional[Tuple[str, ...]] = None
    compatibility_label: str = "likely-compatible"
    unsupported_reason: Optional[str] = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "wan21-t2v-1.3b": ModelSpec(
        key="wan21-t2v-1.3b", family="wan", pipeline_class="WanPipeline",
        hf_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", local_dir="Wan2.1-T2V-1.3B-Diffusers",
        fps=16, default_frames=81, default_steps=50, guidance_scale=5.0, output_type="np",
    ),
    "wan21-t2v-14b": ModelSpec(
        key="wan21-t2v-14b", family="wan", pipeline_class="WanPipeline",
        hf_id="Wan-AI/Wan2.1-T2V-14B-Diffusers", local_dir="Wan2.1-T2V-14B-Diffusers",
        fps=16, default_frames=81, default_steps=50, guidance_scale=5.0, output_type="np",
    ),
    "wan22-t2v-a14b": ModelSpec(
        key="wan22-t2v-a14b", family="wan", pipeline_class="WanPipeline",
        hf_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers", local_dir="Wan2.2-T2V-A14B-Diffusers",
        fps=16, default_frames=81, default_steps=40, guidance_scale=5.0, output_type="np",
    ),
    "hunyuan-t2v": ModelSpec(
        key="hunyuan-t2v", family="hunyuan_video", pipeline_class="HunyuanVideoPipeline",
        hf_id="tencent/HunyuanVideo", local_dir="HunyuanVideo",
        fps=24, default_frames=129, default_steps=50, guidance_scale=6.0, output_type="pil",
    ),
    "wan21-i2v-14b": ModelSpec(
        key="wan21-i2v-14b", family="wan", pipeline_class="WanImageToVideoPipeline",
        hf_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", local_dir="Wan2.1-I2V-14B-720P-Diffusers",
        fps=16, default_frames=81, default_steps=50, guidance_scale=5.0, output_type="np",
    ),
    "wan22-i2v-a14b": ModelSpec(
        key="wan22-i2v-a14b", family="wan", pipeline_class="WanImageToVideoPipeline",
        hf_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers", local_dir="Wan2.2-I2V-A14B-Diffusers",
        fps=16, default_frames=81, default_steps=40, guidance_scale=5.0, output_type="np",
    ),
    "hunyuan-i2v": ModelSpec(
        key="hunyuan-i2v", family="hunyuan_video", pipeline_class="HunyuanVideoImageToVideoPipeline",
        hf_id="hunyuanvideo-community/HunyuanVideo-I2V", local_dir="HunyuanVideo-I2V",
        fps=24, default_frames=129, default_steps=50, guidance_scale=6.0, output_type="pil",
    ),
    "skyreels-v2-t2v-14b": ModelSpec(
        key="skyreels-v2-t2v-14b", family="wan", pipeline_class="SkyReelsV2Pipeline",
        hf_id="Skywork/SkyReels-V2-T2V-14B-720P-Diffusers", local_dir="skyreels-v2-t2v-14b",
        fps=24, default_frames=97, default_steps=50, guidance_scale=6.0, output_type="np",
    ),
    "skyreels-v2-i2v-14b": ModelSpec(
        key="skyreels-v2-i2v-14b", family="wan", pipeline_class="SkyReelsV2ImageToVideoPipeline",
        hf_id="Skywork/SkyReels-V2-I2V-14B-720P-Diffusers", local_dir="skyreels-v2-i2v-14b",
        fps=24, default_frames=97, default_steps=50, guidance_scale=5.0, output_type="np",
    ),
    "wan22-animate-14b": ModelSpec(
        key="wan22-animate-14b", family="wan", pipeline_class="WanAnimatePipeline",
        hf_id="Wan-AI/Wan2.2-Animate-14B-Diffusers", local_dir="Wan2.2-Animate-14B-Diffusers",
        fps=16, default_frames=77, default_steps=20, guidance_scale=1.0, output_type="np",
    ),
    "wan21-vace-1.3b": ModelSpec(
        key="wan21-vace-1.3b", family="wan", pipeline_class="WanVACEPipeline",
        hf_id="Wan-AI/Wan2.1-VACE-1.3B-diffusers", local_dir="Wan2.1-VACE-1.3B-diffusers",
        fps=16, default_frames=81, default_steps=50, guidance_scale=5.0, output_type="np",
    ),
    "wan21-vace-14b": ModelSpec(
        key="wan21-vace-14b", family="wan", pipeline_class="WanVACEPipeline",
        hf_id="Wan-AI/Wan2.1-VACE-14B-diffusers", local_dir="Wan2.1-VACE-14B-diffusers",
        fps=16, default_frames=81, default_steps=50, guidance_scale=5.0, output_type="np",
    ),
    "cogvideox-t2v": ModelSpec(
        key="cogvideox-t2v", family="cogvideox", pipeline_class="CogVideoXPipeline",
        hf_id="THUDM/CogVideoX-5b", local_dir="CogVideoX-5b",
        fps=8, default_frames=49, default_steps=50, guidance_scale=6.0, output_type="pil",
        sparse_supported=True,
    ),
    "cogvideox-i2v": ModelSpec(
        key="cogvideox-i2v", family="cogvideox", pipeline_class="CogVideoXImageToVideoPipeline",
        hf_id="THUDM/CogVideoX-5b-I2V", local_dir="CogVideoX-5b-I2V",
        fps=8, default_frames=49, default_steps=50, guidance_scale=6.0, output_type="pil",
        sparse_supported=True,
    ),
    "ltx-video": ModelSpec(
        key="ltx-video", family="ltx_video", pipeline_class="LTXPipeline",
        hf_id="Lightricks/LTX-Video", local_dir="ltx-video",
        fps=25, default_frames=161, default_steps=50, guidance_scale=3.0, output_type="pil",
        sparse_supported=True,
    ),
    "ltx-video-i2v": ModelSpec(
        key="ltx-video-i2v", family="ltx_video", pipeline_class="LTXImageToVideoPipeline",
        hf_id="Lightricks/LTX-Video", local_dir="ltx-video",
        fps=25, default_frames=161, default_steps=50, guidance_scale=3.0, output_type="pil",
        sparse_supported=True,
    ),
    "allegro": ModelSpec(
        key="allegro", family="allegro", pipeline_class="AllegroPipeline",
        hf_id="rhymes-ai/Allegro", local_dir="allegro",
        fps=15, default_frames=88, default_steps=100, guidance_scale=7.5, output_type="pil",
        sparse_supported=True,
    ),
    "mochi-1": ModelSpec(
        key="mochi-1", family="mochi", pipeline_class="MochiPipeline",
        hf_id="genmo/mochi-1-preview", local_dir="mochi-1",
        fps=8, default_frames=19, default_steps=64, guidance_scale=4.5, output_type="pil",
        sparse_supported=True,
    ),
    "easyanimate-v5-t2v-12b": ModelSpec(
        key="easyanimate-v5-t2v-12b", family="easyanimate", pipeline_class="EasyAnimatePipeline",
        hf_id="alibaba-pai/EasyAnimateV5.1-12b-zh-diffusers", local_dir="easyanimate-v5-t2v-12b",
        fps=8, default_frames=49, default_steps=50, guidance_scale=5.0, output_type="pil",
        sparse_supported=True,
    ),
    "sana-video": ModelSpec(
        key="sana-video", family="sana_video", pipeline_class="SanaVideoPipeline",
        hf_id="Efficient-Large-Model/SANA-Video_2B_480p_diffusers", local_dir="sana-video",
        fps=24, default_frames=17, default_steps=20, guidance_scale=5.0, output_type="pil",
        sparse_supported=False, sparse_methods=(), compatibility_label="incompatible",
        unsupported_reason=(
            "SanaVideo uses Diffusers' SanaLinearAttnProcessor3_0 linear attention, "
            "not softmax QK^T V attention; current SparseVideo sparse-softmax methods "
            "are incompatible."
        ),
    ),
    "motif-video": ModelSpec(
        key="motif-video", family="motif_video", pipeline_class="UnavailablePipeline",
        hf_id="", local_dir=None,
        fps=24, default_frames=1, default_steps=1, guidance_scale=1.0, output_type="pil",
        sparse_supported=False, sparse_methods=(), compatibility_label="unknown",
        unsupported_reason=(
            "MotifVideo is not available in the current Diffusers installation "
            "and no confirmed local/Hugging Face checkpoint is configured, so "
            "SparseVideo cannot verify a processor-swap path."
        ),
    ),
    "ltx-video-2": ModelSpec(
        key="ltx-video-2", family="ltx_video_2", pipeline_class="UnavailablePipeline",
        hf_id="", local_dir=None,
        fps=24, default_frames=1, default_steps=1, guidance_scale=1.0, output_type="pil",
        sparse_supported=False, sparse_methods=(), compatibility_label="unknown",
        unsupported_reason=(
            "LTX Video 2 is not available in the current Diffusers installation; "
            "SparseVideo cannot verify the requested video attn1 plus audio_attn1 "
            "structure or safely reuse the plain LTX Video processor."
        ),
    ),
    "kandinsky5-t2v": ModelSpec(
        key="kandinsky5-t2v", family="kandinsky5", pipeline_class="Kandinsky5T2VPipeline",
        hf_id="ai-forever/Kandinsky-5.0-T2V", local_dir="kandinsky5-t2v",
        fps=12, default_frames=49, default_steps=50, guidance_scale=5.0, output_type="pil",
        sparse_supported=False, sparse_methods=(), compatibility_label="native-N/A",
        unsupported_reason=(
            "Kandinsky5 exposes native sparse attention controls through transformer "
            "sparse_params/window parameters, so it is not a SparseVideo processor-swap target."
        ),
    ),
}

MODEL_ALIASES: dict[str, str] = {
    "wan1.3b": "wan21-t2v-1.3b",
    "wan21-1.3b": "wan21-t2v-1.3b",
    "wan21-t2v-1.3b": "wan21-t2v-1.3b",
    "wan14b": "wan21-t2v-14b",
    "wan21-14b": "wan21-t2v-14b",
    "wan21-t2v-14b": "wan21-t2v-14b",
    "wan22": "wan22-t2v-a14b",
    "wan22-a14b": "wan22-t2v-a14b",
    "wan22-t2v-a14b": "wan22-t2v-a14b",
    "hunyuan": "hunyuan-t2v",
    "hunyuan-t2v": "hunyuan-t2v",
    "hunyuan-i2v": "hunyuan-i2v",
    "skyreels": "skyreels-v2-t2v-14b",
    "skyreels-v2": "skyreels-v2-t2v-14b",
    "skyreels-v2-t2v": "skyreels-v2-t2v-14b",
    "skyreels-v2-t2v-14b": "skyreels-v2-t2v-14b",
    "skyreels-i2v": "skyreels-v2-i2v-14b",
    "skyreels-v2-i2v": "skyreels-v2-i2v-14b",
    "skyreels-v2-i2v-14b": "skyreels-v2-i2v-14b",
    "wananimate": "wan22-animate-14b",
    "wan-animate": "wan22-animate-14b",
    "wan22-animate": "wan22-animate-14b",
    "wan22-animate-14b": "wan22-animate-14b",
    "vace": "wan21-vace-1.3b",
    "wan-vace": "wan21-vace-1.3b",
    "wan21-vace": "wan21-vace-1.3b",
    "wan21-vace-1.3b": "wan21-vace-1.3b",
    "wan21-vace-14b": "wan21-vace-14b",
    "wan-i2v": "wan21-i2v-14b",
    "wan14b-i2v": "wan21-i2v-14b",
    "wan21-i2v": "wan21-i2v-14b",
    "wan21-i2v-14b": "wan21-i2v-14b",
    "wan22-i2v": "wan22-i2v-a14b",
    "wan22-i2v-a14b": "wan22-i2v-a14b",
    "cog": "cogvideox-t2v",
    "cogvideox": "cogvideox-t2v",
    "cogvideox-t2v": "cogvideox-t2v",
    "cog-i2v": "cogvideox-i2v",
    "cogvideox-i2v": "cogvideox-i2v",
    "cogvideox-5b-i2v": "cogvideox-i2v",
    "ltx": "ltx-video",
    "ltx-video": "ltx-video",
    "ltx-i2v": "ltx-video-i2v",
    "ltx-video-i2v": "ltx-video-i2v",
    "allegro": "allegro",
    "mochi": "mochi-1",
    "mochi-1": "mochi-1",
    "easyanimate": "easyanimate-v5-t2v-12b",
    "easyanimate-v5": "easyanimate-v5-t2v-12b",
    "easyanimate-v5-t2v-12b": "easyanimate-v5-t2v-12b",
    "motif": "motif-video",
    "motif-video": "motif-video",
    "ltx2": "ltx-video-2",
    "ltx-video-2": "ltx-video-2",
    "sana-video": "sana-video",
    "sanavideo": "sana-video",
    "kandinsky5": "kandinsky5-t2v",
    "kandinsky5-t2v": "kandinsky5-t2v",
}
