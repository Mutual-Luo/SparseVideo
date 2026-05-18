# SparseVideo All-Backbone Goal

Goal: make SparseVideo a plug-and-play sparse-attention layer for Diffusers video DiT pipelines. Use one API, `sparsevideo.apply(pipe, method=...)`; keep `dense` and restore-to-dense. Current: Wan/HunyuanVideo done; CogVideoX detection/step tracking only. Target softmax video-attention DiTs; skip incompatible math or native sparse designs.

Methods: `svg1`, `svg2`, `spargeattn`, `radial`, `sta`, `draft`, `adacluster`, `flashomni`, `svoo`, plus `dense`.

## Coverage Plan

Tier 1, wiring only: WanAnimate, WanVACE, SkyReels V2. They match Wan `blocks[i].attn1` video self-attn. Add `discover_model()` class aliases and inference aliases; reuse Wan enumeration/processor. No new processor.

Tier 2, separate video self-attn with new processor: LTX Video, Allegro, CogVideoX. Add enumeration plus processor.
- LTX Video: `transformer_blocks[i].attn1 = LTXAttention`; processor dispatch; self-attn uses `encoder_hidden_states=None`.
- Allegro: standard Diffusers `Attention`; sparse `attn1`, keep `attn2` dense.
- CogVideoX: concatenated text+video sequence; split, sparse video slice only, recombine.

Tier 3, joint/dual-stream/multimodal: Mochi, EasyAnimate, MotifVideo, LTX Video 2. Processors must identify text/video tokens and sparse only video-video; keep text, text-video, and audio paths dense.
- Mochi: joint dual-stream `MochiAttention`, returns video/text; similar to Hunyuan double; SpargeAttn/radial have upstream evidence.
- EasyAnimate: concatenates `[text, video]`, attends jointly, then splits.
- MotifVideo: mixed double/single blocks, field `attn`; needs two enumerate paths like Hunyuan.
- LTX Video 2: video `attn1` plus `audio_attn1`; leave audio dense.

Skip/defer: SanaVideo uses linear attention, not softmax `Q K^T V`; incompatible unless methods are redesigned. Kandinsky5 already has native sparse params (`sparse_params`, `sta_mask`, `P`, window dims`) and is not a processor-swap target.

## Architecture Rules

- Separate self-attn + cross-attn: Wan family, SkyReels, LTX, Allegro. Hook video `attn1`; keep cross-attn dense.
- Joint dual/single stream: Hunyuan, Mochi, Motif. Sparse only video-video.
- Concatenated text/video sequence: CogVideoX, EasyAnimate. Split text/video, sparse video slice, recombine.
- Linear attention: SanaVideo. Do not adapt with current sparse-softmax methods.

## Integration Points

Update only required code: `_model_info.discover_model()` for detection/enumeration; `processors/<model>.py` for Q/K/V interception and sparse dispatch; `_step_tracker.py` only if existing gating cannot track the model; inference aliases for real pipeline smoke tests; method configs for model-aware defaults without renaming public methods.

## Priority

Phase 1: WanAnimate, WanVACE, SkyReels V2, CogVideoX processor.
Phase 2: Mochi, LTX Video.
Phase 3: Allegro, EasyAnimate, MotifVideo, LTX Video 2.
Phase 4: SanaVideo and Kandinsky5 deferred/skipped.

## Compatibility Labels

Wan family/SkyReels: all methods expected after wiring, subject to preflight. HunyuanVideo: supported; preserve restrictions. CogVideoX/Mochi/LTX/Allegro/EasyAnimate/Motif/LTX2: validate per method; SpargeAttn is most architecture-agnostic; radial has upstream evidence for Mochi/LTX-like targets. SanaVideo: incompatible. Kandinsky5: native-N/A.

Use labels: upstream-supported, likely-compatible, unknown, incompatible, native-N/A. No benchmark claims from shape-only tests.

## Success Criteria

- Tier 1 dry-run passes for all methods through the Wan path.
- Tier 2 real processors pass smoke inference with `dense` plus at least 2 sparse methods.
- Tier 3 text/video-aware processors pass smoke inference with `dense` plus at least 1 sparse method.
- SpargeAttn works on every compatible backbone, or preflight gives a precise reason.
- No Wan/Hunyuan regression in quality, speed gates, or restore-to-dense.
- Same small API and unified registration/config flow for every model.
