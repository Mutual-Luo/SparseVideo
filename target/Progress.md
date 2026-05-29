# Backbone Coverage Progress

## Diffusers Backend

19 backbones supported (all 10 methods × all listed models in `inference_diffusers.sh`).

| Backbone | Pipeline | Status |
|---|---|---|
| Wan 2.1 T2V 1.3B | WanPipeline | ✅ |
| Wan 2.1 T2V 14B | WanPipeline | ✅ |
| Wan 2.2 T2V A14B | WanPipeline | ✅ |
| Wan 2.1 I2V 14B | WanImageToVideoPipeline | ✅ |
| Wan 2.2 I2V A14B | WanImageToVideoPipeline | ✅ |
| Wan 2.2 Animate 14B | WanAnimatePipeline | ✅ |
| Wan 2.1 VACE 1.3B | WanVACEPipeline | ✅ |
| Wan 2.1 VACE 14B | WanVACEPipeline | ✅ |
| HunyuanVideo T2V | HunyuanVideoPipeline | ✅ |
| HunyuanVideo I2V | HunyuanVideoImageToVideoPipeline | ✅ |
| SkyReels V2 T2V 14B | SkyReelsV2Pipeline | ✅ |
| SkyReels V2 I2V 14B | SkyReelsV2ImageToVideoPipeline | ✅ |
| CogVideoX T2V | CogVideoXPipeline | ✅ |
| CogVideoX I2V | CogVideoXImageToVideoPipeline | ✅ |
| LTX Video T2V | LTXPipeline | ✅ |
| LTX Video I2V | LTXImageToVideoPipeline | ✅ |
| Allegro | AllegroPipeline | ✅ |
| Mochi | MochiPipeline | ✅ |
| EasyAnimate V5 12B | EasyAnimatePipeline | ✅ |
| Sana Video | SanaVideoPipeline | ❌ linear attention, incompatible |
| Motif Video | — | ❌ not available in Diffusers |
| LTX Video 2 | — | ❌ not available in Diffusers |
| Kandinsky 5 T2V | Kandinsky5T2VPipeline | ❌ native sparse params, not a swap target |

---

## DiffSynth-Studio Backend

DiffSynth has 3 video pipeline types. All are code-supported by `_diffsynth.py`.
"✅" = in `inference_diffsynth.sh` with all 10 methods.
"⬜" = code path works, not yet in sh.

### WanVideoPipeline

| Backbone | Status |
|---|---|
| Wan 2.1 T2V 1.3B | ✅ |
| Wan 2.1 T2V 14B | ✅ |
| Wan 2.2 T2V A14B | ✅ |
| Wan 2.1 I2V 14B 720P | ✅ |
| Wan 2.2 I2V A14B | ✅ |
| Wan 2.2 Animate 14B | ✅ |
| Wan 2.1 VACE 1.3B | ✅ |
| Wan 2.1 VACE 14B | ✅ |
| Wan 2.1 SpeedControl 1.3B | ✅ |
| Wan 2.1 I2V 14B 480P | ✅ |
| Wan 2.1 FLF2V 14B 720P | ✅ |
| Wan 2.1 Fun 1.3B Control | ✅ |
| Wan 2.1 Fun 1.3B InP | ✅ |
| Wan 2.1 Fun 14B Control | ✅ |
| Wan 2.1 Fun 14B InP | ✅ |
| Wan 2.1 Fun V1.1 1.3B Control | ✅ |
| Wan 2.1 Fun V1.1 1.3B Control-Camera | ✅ |
| Wan 2.1 Fun V1.1 14B Control | ✅ |
| Wan 2.1 Fun V1.1 14B Control-Camera | ✅ |
| Wan 2.2 TI2V 5B | ✅ |
| Wan 2.2 S2V 14B | ✅ |
| Wan 2.2 Fun A14B Control | ✅ |
| Wan 2.2 Fun A14B Control-Camera | ✅ |
| LongCat-Video | ✅ |
| Video-as-Prompt Wan 2.1 14B | ✅ |
| Krea Realtime Video | ✅ |
| Wan 2.2 Dancer 14B | ❌ deferred — no confirmed checkpoint source |

### MovaAudioVideoPipeline

| Backbone | Status |
|---|---|
| MOVA 720P | ⬜ |

### LTX2AudioVideoPipeline

| Backbone | Status |
|---|---|
| LTX-2 | ⬜ |
| LTX-2.3 | ⬜ |
