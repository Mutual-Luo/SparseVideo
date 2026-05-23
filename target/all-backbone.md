# SparseVideo All-Backbone All-Method Goal

Goal: make SparseVideo a plug-and-play sparse-attention layer for both major community video generation stacks:
Diffusers video DiT pipelines and DiffSynth-Studio video pipelines. Use one API,
`sparsevideo.apply(pipe, method=...)`; keep `dense` as the baseline and guarantee restore-to-dense.

Current Diffusers status: Wan, HunyuanVideo, SkyReels, CogVideoX, LTX Video, Allegro, Mochi, and EasyAnimate have processor wiring. The reduced-surface method guards for the new backbones have been removed, but the default all-backbone audit still requires missing real smoke evidence before this can be called complete.

Current DiffSynth-Studio status: adapter work is in progress. SparseVideo now keeps Diffusers and DiffSynth discovery in
separate backend files, keeps DiffSynth-native local model loading in a separate DiffSynth inference entrypoint, and has
a DiffSynth Wan-family adapter for the newest installed `diffsynth` package version, verified at audit time with
`importlib.metadata.version("diffsynth")`. As of the 2026-05-21 target update, the local environment was upgraded to
`diffsynth==2.0.12`; a 2026-05-22 `pip index versions diffsynth` check still reports 2.0.12 as the latest published
version, and direct package inspection shows the current video pipeline classes are exactly `WanVideoPipeline`,
`MovaAudioVideoPipeline`, and `LTX2AudioVideoPipeline`. Wan2.1 T2V 1.3B has real DiffSynth load/apply/restore evidence
for `dense` and all public sparse
methods, including a standalone 10-method `--apply-only` sweep where `dense` restores without patching and every sparse
method patches 30 Wan attention modules and restores cleanly. It also has current strict-format 128x128/5-frame
generation smoke for `dense` and `svg2` with `resolved_model.complete=true`; `svg2` records 60 sparse `flashinfer`
dispatches. Older generation smoke also covered `svg1`,
`spargeattn`, `radial`, `sta`, `draft`, `adacluster`, `flashomni`, and `svoo`. The all-method generation sweep
patches 30 Wan attention modules and records sparse dispatch for every public sparse method: `svg1` uses
`flex_attention`, `svg2` uses `flashinfer`, `spargeattn` uses `spas_sage_topk`, `radial` uses `flashinfer` with its
expected dense edge cases, `sta` uses
`fastvideo_sta_a100_triton`, `draft` uses `mit_block_sparse` after its upstream high-timestep dense gate, `adacluster`
uses `triton_cluster_sparse_attn_topk`, `flashomni` uses the diagnostic `global_random` sparse pattern unless explicit
upstream sparse-info tensors are supplied, and `svoo` uses `svoo_flashinfer`. Wan2.1 T2V 14B has real DiffSynth
`dense` baseline apply/restore evidence plus all-public-sparse-method `--apply-only` evidence after its six native shards
completed locally; `svg1`, `svg2`, `spargeattn`, `radial`, `sta`, `draft`, `adacluster`, `flashomni`, and `svoo` each
patch 40 DiffSynth attention modules and restore cleanly. Wan2.1 I2V 14B 480P has real DiffSynth `dense` baseline and
all-public-sparse-method `--apply-only` evidence with the image encoder loaded; every sparse method patches 40 DiffSynth
attention modules and restores cleanly. Wan2.1 I2V 14B 720P now has the same all-public-method apply/restore evidence
after all seven native I2V DiT shards completed locally; `dense` patches zero modules, while each sparse method patches
40 DiffSynth attention modules and restores cleanly. Wan2.1 FLF2V 14B 720P now has real `dense` baseline and
all-public-sparse-method `--apply-only` evidence after all seven native FLF DiT shards completed locally; `dense`
patches zero modules, while each sparse method patches 40 DiffSynth attention modules and restores cleanly. Wan2.1 VACE
1.3B Preview now has real `dense` baseline and all-public-sparse-method `--apply-only` evidence after its native
checkpoint completed locally; `dense` patches zero modules, while each sparse method patches 45 DiffSynth attention
modules across `dit.blocks.*` and `vace.vace_blocks.*` and restores cleanly. Wan2.1 VACE 14B now has real `dense`
baseline and all-public-sparse-method `--apply-only` evidence after all seven native VACE DiT shards completed locally;
`dense` patches zero modules, while each sparse method patches 48 DiffSynth attention modules across `dit.blocks.*` and
`vace.vace_blocks.*` and restores cleanly. Wan2.2 TI2V 5B has real `dense` baseline and all-public-sparse-method
apply/restore evidence, patching 30 DiffSynth attention modules for each sparse method; it also
has tiny DiffSynth generation smoke for `dense` and `svg2`, where `svg2` records 60 sparse `flashinfer` dispatches.
Wan2.2 T2V A14B has real `dense` baseline and all-public-sparse-method load/apply/restore evidence for the dual-DiT
high/low-noise pipeline, patching 80 DiffSynth attention modules across `dit` and `dit2` for each sparse method. Wan2.2
I2V A14B has the same strict-format `dense` baseline and all-public-sparse-method `--apply-only` evidence after its
low-noise shards completed locally, also patching 80 DiffSynth attention modules across `dit` and `dit2` for each sparse
method. Wan2.2
Animate 14B now has real `dense` baseline and all-public-sparse-method `--apply-only` evidence after all four native
Animate shards completed locally; `dense` patches zero modules, while each sparse method patches 40 main Wan attention
modules and reports `animate_adapter.scaled_dot_product_attention` as an unpatched auxiliary path. Wan2.2 S2V 14B now
has real `dense` baseline and all-public-sparse-method `--apply-only` evidence after the native DIT shards, wav2vec
audio encoder checkpoint, and wav2vec processor files completed locally; `dense` patches zero modules, while each sparse
method patches 40 main Wan attention modules and reports `dit.audio_injector.injector.*.attn` as an unpatched auxiliary
path. Krea realtime video 14B now has real `dense` baseline and all-public-sparse-method `--apply-only` evidence after
the native Krea DiT checkpoint completed locally; `dense` patches zero modules, while each sparse method patches 40 main
Wan attention modules and restores cleanly. The existing local
LongCat bundle under `Longcat/weights/LongCat-Video` is detected without redownload; LongCat has real
`dense` baseline and all-public-sparse-method load/apply/restore evidence, patching 48 DiffSynth
`_process_attn(q, k, v, shape)` paths for each sparse method and reporting LongCat cross-attention as an unpatched path.
Wan2.1 SpeedControl 1.3B, Wan2.1-Fun 1.3B Control/InP, Wan2.1-Fun 14B Control/InP, Wan2.1-Fun V1.1 1.3B
Control/Control-Camera, and Wan2.1-Fun V1.1 14B Control/Control-Camera now have real `dense` baseline and all-public-sparse-method
`--apply-only` evidence after their local bundles completed; the SVOO defaults use online clustering for these
unprofiled Wan-family variants instead of borrowing an unrelated dynamic sparsity CSV. Wan2.2-Fun A14B
Control/Control-Camera now have real `dense` baseline and all-public-sparse-method `--apply-only` evidence after their
high-noise DiTs completed locally. Video-as-Prompt Wan2.1 14B now has real `dense` baseline and
all-public-sparse-method `--apply-only` evidence after all five native shards completed locally; VAP
`MotWanAttentionBlock.flash_attention` is reported as an unpatched auxiliary path. MOVA 720P now has real `dense`
baseline and all-public-sparse-method `--apply-only` evidence after its tokenizer, audio DiT, audio VAE, and
dual-tower bridge components completed locally; MOVA `audio_dit` and `dual_tower_bridge` attention paths are reported
as unpatched auxiliary paths.
The reproducible sweep entrypoint is
`scripts/smoke_diffsynth_methods.py`, whose default `--methods all` includes `dense` plus every public sparse method;
use `--methods sparse` only for sparse-method-only debugging, and use `--list-models` or `--models all` to inspect or
sweep the full DiffSynth catalog. The test suite now checks the active downloader/catalog against the installed
DiffSynth video model-config examples for Wan/MOVA/LTX2, allowing only documented converted/shared-file alternatives
and explicit deferred/local-only entries. The current strict audit requires each DiffSynth smoke/inference record to include
`resolved_model.complete=true` for the same local bundle it loaded. The LTX2 Gemma text model is now complete locally.
LTX-2 now has real `dense` baseline and all-public-sparse-method `--apply-only` evidence after its repackaged DiT,
video VAE, audio VAE/vocoder, and text post-module components completed locally; `dense` patches zero modules, while
each sparse method patches 48 LTX2 video self-attention modules and reports text/audio/cross-attention as unpatched
auxiliary paths. LTX-2.3 now has the same real `dense` baseline and all-public-sparse-method `--apply-only` evidence
after its source checkpoint and latent upsampler completed locally; `dense` patches zero modules, while each sparse
method patches 48 LTX2 video self-attention modules and reports text/audio/cross-attention as unpatched auxiliary paths.
The audit reports `diffsynth_checkpoint_availability` passing with 29 complete active local bundles, 0 incomplete
bundles, 0 missing component checks, and 1 explicit deferred/local-only WanToDance bundle. The
`diffsynth_apply_restore_evidence` gate also passes with no missing method records.
The 29 complete local bundles have regenerated `dense` plus all-public-sparse-method apply/restore
records with resolved local bundle metadata: Wan2.1 T2V 1.3B/14B, Wan2.1 I2V 14B 480p/720p, Wan2.1 FLF2V 14B 720p,
Wan2.1 SpeedControl 1.3B, Wan2.1-Fun 1.3B Control/InP, Wan2.1-Fun 14B Control/InP, Wan2.1-Fun V1.1 1.3B
Control/Control-Camera, Wan2.1-Fun V1.1 14B Control/Control-Camera, Wan2.1 VACE 1.3B/14B, Wan2.2 Animate 14B, Wan2.2 T2V A14B,
Wan2.2 I2V A14B, Wan2.2 TI2V 5B, Wan2.2 S2V 14B, Wan2.2-Fun A14B Control/Control-Camera, Video-as-Prompt Wan2.1 14B,
Krea realtime video 14B, MOVA 720P, LongCat-Video, LTX-2, and LTX-2.3. Older local
apply/restore records without resolved bundle metadata no longer count. DiffSynth checkpoint/load/apply/restore status is
complete. DiffSynth Wan2.1 T2V 1.3B now has a first matched 50-step quality pair under
`outputs/diffsynth_quality/wan21-t2v-1.3b/20260522_194929/`: `dense` and `svg2` use the same prompt, seed 0,
480x832, 81 frames, 15 fps, and local native DiffSynth bundle; `svg2` records 2610 sparse FlashInfer dispatches,
390 dense warmup dispatches, restore success, and valid 81-frame mp4 output. The DiffSynth inference entrypoint now
writes audit-compatible `sparse_attention_handle` summaries for strict quality records. Wan2.1 T2V 1.3B also has
50-step strict DiffSynth quality records under `outputs/diffsynth_quality/wan21-t2v-1.3b/20260522_200551/` for
`adacluster`, `draft`, and `svoo`, with valid 81-frame mp4 outputs, restore success, and observed sparse backend
dispatch: `triton_cluster_sparse_attn_topk`, `mit_block_sparse`, and `svoo_flashinfer`, respectively. A second
Wan2.1 T2V 1.3B 50-step strict DiffSynth quality sweep under
`outputs/diffsynth_quality/wan21-t2v-1.3b/20260522_201452/` adds `svg1`, `spargeattn`, `radial`, and `sta`, with valid
81-frame mp4 outputs, restore success, and observed sparse backends
`flex_attention`, `spas_sage_topk`, `flashinfer`, and `fastvideo_sta_a100_triton`. A matching STA legacy-triton record
under `outputs/diffsynth_quality/wan21-t2v-1.3b/20260522_202816/` uses the same prompt, seed, shape, frames, fps, and
50-step settings with `fastvideo_sta_triton`, produces a valid 81-frame mp4, and records 269.46s generation time versus
211.00s for the A100-optimized record. This satisfies the current A100 STA before/after speed evidence. The method-level
strict quality gate now remains blocked only by FlashOmni explicit/paper-policy Hunyuan evidence. Full all-backbone
status remains incomplete while the broader non-DiffSynth smoke gate still has missing records, and FlashOmni still
needs its explicit/paper-policy evidence.

Current non-DiffSynth audit status: `all_backbone_checkpoint_availability` now passes after the Hunyuan Diffusers model
specs were aligned with the existing local directories `HunyuanVideo-Diffusers` and `HunyuanVideo-I2V-Diffusers`.
Hunyuan T2V and I2V now have real 1-step `--skip-decode` smoke records for `dense`, `svg2`, and `svoo` under
`result/inference/hunyuan-t2v-smoke/metrics.jsonl` and `result/inference/hunyuan-i2v-smoke/metrics.jsonl`; both sparse
methods record sparse dispatch. `scripts/smoke_all_backbones.sh` is the bandwidth-safe Diffusers smoke entrypoint for
this gate: it defaults to `--missing`, `--local-files-only`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`--skip-decode`, and automatically adds the local portrait image for I2V backbones. The Wan I2V/T2V high-priority
gap records in `result/inference/all_backbone_smoke/metrics.jsonl` now include `wan21-i2v-14b`, `wan22-t2v-a14b`, and
`wan22-i2v-a14b` for `dense`, `svg2`, and `svoo`; `svg2` records `flashinfer` sparse dispatch and `svoo` records
`svoo_flashinfer` sparse dispatch. CogVideoX T2V and I2V now also have 1-step `--skip-decode` records for `dense`,
`svg2`, and `svoo`; both sparse methods record the expected `flashinfer`/`svoo_flashinfer` sparse dispatch. The
`all_backbone_smoke_evidence` gate is Diffusers-only and must not count DiffSynth quality/apply records as Diffusers
smoke. It is still incomplete with 67 missing smoke records across the remaining Diffusers backbones/methods.

Public methods: `svg1`, `svg2`, `spargeattn`, `radial`, `sta`, `draft`, `adacluster`, `flashomni`, `svoo`, plus `dense`.

## Target Backbones

Already broad-supported, preserve behavior and avoid regressions:
- `wan21-t2v-1.3b`, `wan21-t2v-14b`, `wan22-t2v-a14b`
- `wan21-i2v-14b`, `wan22-i2v-a14b`
- `wan22-animate-14b`
- `wan21-vace-1.3b`, `wan21-vace-14b`
- `skyreels-v2-t2v-14b`, `skyreels-v2-i2v-14b`
- `hunyuan-t2v`, `hunyuan-i2v`

New backbones that must be upgraded from `dense/svg2/svoo` to all public sparse methods:
- `cogvideox-t2v`
- `cogvideox-i2v`
- `ltx-video`
- `ltx-video-i2v`
- `allegro`
- `mochi-1`
- `easyanimate-v5-t2v-12b`

Non-targets unless the architecture changes:
- `sana-video`: incompatible with the current softmax sparse-attention methods because it uses linear attention.
- `kandinsky5-t2v`: native-N/A because it already exposes native sparse parameters instead of a processor-swap surface.
- `motif-video` and `ltx-video-2`: unknown until their current Diffusers attention ownership and token layout are verified.

## DiffSynth-Studio Target

DiffSynth-Studio is a first-class target, not an optional demo path. A SparseVideo release cannot claim all-community
video-pipeline support while it only handles Diffusers processor-swap pipelines.

Current DiffSynth-Studio video pipelines that must be covered or explicitly deferred:
- `diffsynth.pipelines.wan_video.WanVideoPipeline`: first priority. It owns `WanModel`/`LongCatVideoTransformer3DModel`
  style video DiT modules through `pipe.dit`, optional `pipe.dit2`, VACE modules through `pipe.vace`/`pipe.vace2`,
  and extra Wan-family adapters such as VAP, Animate, S2V, TI2V, VACE-Fun, WanToDance, Krea realtime video,
  Video-as-Prompt Wan, LongCat-Video, and Wan2.1/Wan2.2 T2V/I2V/FLF/VACE variants.
  Current Wan `AttentionModule.forward(q, k, v)` patching covers WanModel-style `blocks.*.self_attn.attn` and VACE
  `vace_blocks.*.self_attn.attn`, plus LongCat `Attention._process_attn(q, k, v, shape)` self-attention.
  Wan2.2 A14B local loading must pass high-noise then low-noise DiT configs before shared text/VAE components so
  DiffSynth constructs the expected dual-DiT pipeline.
  VAP/MotWan custom `flash_attention`, Animate adapter SDPA, and S2V audio-injector
  cross-attention are reported through `handle.summary()["unpatched_attention_paths"]` and remain partial outside the
  patched main Wan/LongCat self-attention path. LongCat cross-attention must also be reported as an unpatched path and
  remains dense unless a method explicitly owns that interaction. The S2V local loader must include both the wav2vec
  audio encoder checkpoint and the wav2vec audio processor directory. Because the current Wan2.2 S2V processor config
  points at `Wav2Vec2ProcessorWithLM`, the
  SparseVideo script loader should construct a standard `Wav2Vec2Processor` from the local `Wav2Vec2CTCTokenizer` and
  `Wav2Vec2FeatureExtractor` files instead of letting DiffSynth's default `audio_processor_config` path require extra
  LM-tokenizer backends.
  The Wan2.1 1.3B speed-control model must load its DiffSynth motion-controller checkpoint alongside the base
  Wan2.1 T2V 1.3B DiT/text/VAE files, and inference must pass `motion_bucket_id` through to the native Wan pipeline.
  Wan2.1-Fun Control/InP/Control-Camera and Wan2.2-Fun A14B Control/Control-Camera checkpoints exposed by DiffSynth
  2.x must resolve as WanVideoPipeline DiT bundles with their documented media inputs, while keeping unsupported extra
  adapter attention paths visible instead of silently treating them as fully patched.
  Wan2.2-Dancer/WanToDance is explicit deferred/local-only until a stable downloadable HF or ModelScope source is
  confirmed; if a local pipeline is loaded, `handle.summary()["unpatched_attention_paths"]` must report its
  `music_injector` and `music_encoder` auxiliary attention instead of claiming full sparse coverage. DiffSynth
  `--list-models` output should show deferred/local-only entries separately from active downloader targets.
- `diffsynth.pipelines.mova_audio_video.MovaAudioVideoPipeline`: second priority. SparseVideo targets the video DiT
  path through `pipe.video_dit` and optional `pipe.video_dit2`; audio-DiT sparse attention is a separate target unless a
  method explicitly defines audio attention semantics. Loaded MOVA `audio_dit` and `dual_tower_bridge` attention paths
  are reported through `handle.summary()["unpatched_attention_paths"]`. DiffSynth MOVA inference must preserve
  `frame_rate`, `sigma_shift`, and `switch_DiT_boundary` semantics from `MovaAudioVideoPipeline.__call__`. The local
  loader must construct MOVA from the Wan2.1 T2V video backbone plus MOVA tokenizer, `audio_dit`, `audio_vae`, and
  `dual_tower_bridge` components.
- `diffsynth.pipelines.ltx2_audio_video.LTX2AudioVideoPipeline`: third priority. LTX-2/LTX-2.3 support requires a
  separate layout review of the audio-video DiT path. The download script and Python model resolver should cover the
  LTX-2/LTX-2.3 component bundles for `--dry-run`. SparseVideo can patch LTX2 video self-attention at
  `dit.transformer_blocks.*.attn1`, reports text, audio, and audio-video cross-attention as unpatched, and can construct
  the LTX-2 base pipeline from the repackaged local components. LTX-2.3 loading is wired through the DiffSynth 2.0.12
  source checkpoint plus latent upsampler path to avoid duplicate large-model loading from mixed repackage/source
  components. The LTX2 video self-attention wrapper must preserve DiffSynth's `self_attention_mask` path by passing it
  to the selected SparseVideo method instead of rejecting or dropping it. LTX-2.3 has real local-file load/apply/restore
  evidence after the source checkpoint and latent upsampler are present locally.
- Legacy DiffSynth checkpoints such as `HunyuanVideo` or old CogVideoX directories are not current DiffSynth 2.x video
  pipeline targets unless the installed `diffsynth` package exposes a matching pipeline class again. Keep their current
  SparseVideo support under the Diffusers backend until a real DiffSynth runtime path exists.

DiffSynth-Studio support rules:
- Do not try to pass DiffSynth-Studio pipelines through the Diffusers `.transformer`/`.processor` discovery path.
  Keep Diffusers discovery in `src/sparsevideo/_diffusers.py`, DiffSynth discovery/patching in
  `src/sparsevideo/_diffsynth.py`, and only shared model metadata/dispatch in `src/sparsevideo/_model_info.py`.
  Keep `src/sparsevideo/_api.py` as the small public dispatcher; backend-specific apply/patch logic belongs in the
  backend files so Diffusers and DiffSynth can be repaired independently.
- The DiffSynth apply path must still call method-level `install_model_patches()` hooks so future backend-specific
  model patches are not bypassed. Diffusers-only base Wan/Hunyuan fast-block patches must return no callbacks on
  DiffSynth pipelines instead of globally patching unrelated Diffusers classes.
- Keep inference/model-download conveniences out of the importable `sparsevideo` package. The package may own DiffSynth
  discovery and attention patching, but script-specific model catalogs, local path resolution, `ModelConfig(path=...)`
  loading, video/audio export helpers, and benchmark/smoke orchestration belong under `scripts/`.
  Do not add package modules such as `src/sparsevideo/_diffsynth_infer.py`; DiffSynth inference/test orchestration is a
  script/test concern, not reusable SparseVideo runtime API.
- Keep Diffusers inference in `scripts/infer.py`; keep DiffSynth-native local model helpers in
  `scripts/_diffsynth_models.py`, single-method DiffSynth runs in `scripts/infer_diffsynth.py`, and all-method
  DiffSynth dispatch smoke in `scripts/smoke_diffsynth_methods.py`. The DiffSynth inference entrypoint should expose
  model-family media inputs such as I2V/FLF images, video-to-video inputs, VACE/Animate/VAP/LongCat videos, and S2V
  audio without moving that orchestration into the importable package. The DiffSynth smoke entrypoint should support
  multi-model `--models ... --apply-only` checks so newly downloaded model bundles can be validated for apply/restore
  before full media-dependent generation evidence is available.
- For LTX-2.3, avoid downloading or loading duplicate split repackage components when the DiffSynth 2.0.12 loader path
  uses the `Lightricks/LTX-2.3` source checkpoint plus latent upsampler. The download script may keep the repackage
  entries visible in the catalog, but should skip them for `ltx23` instead of spending bandwidth on unused duplicates.
- Keep the DiffSynth Python model resolver aligned with `scripts/download_diffsynth_models.sh` for all Wan/MOVA/LTX2 bundles:
  a bundle that the download script lists should resolve in `scripts/infer_diffsynth.py --dry-run`, even when full
  generation still needs model-specific media inputs or broader evidence. After downloading each selected bundle, the
  shell downloader should verify the result with the Python resolver so directory-style components such as S2V wav2vec
  cannot be falsely treated as complete merely because a directory contains unrelated files. Wan UMT5, MOVA, and LTX2
  Gemma tokenizer directories must also require their tokenizer/config/processor files before the bundle can verify
  complete.
- For bandwidth-safe downloads, complete local bundles must be skipped, incomplete shard sets must be resumed, and
  `--source modelscope-first` must try ModelScope first and then fall back to Hugging Face/HF mirror for repos that are
  not available there. `--source huggingface-first`/`--source hf-first` must try Hugging Face/HF mirror first and then
  fall back to ModelScope when useful. HF-only repos must stay on Hugging Face directly, and known ModelScope-only repos
  should not waste time on a guaranteed failed Hugging Face request. A matching checkpoint file must be non-empty before
  the resolver or downloader can treat it as complete. The default network policy is HF mirror plus no proxy; proxy use
  must be explicit because proxy bandwidth is constrained.
- DiffSynth Wan-family loaders must de-duplicate shared checkpoint paths before constructing `ModelConfig` objects.
  Some DiffSynth checkpoints are intentionally multi-role by hash, such as VACE/Animate/VAP files that load both a
  video DiT and an adapter from the same file. Passing the same file twice causes duplicate large-model loading.
- For Wan-family DiffSynth modules, patch the self-attention call path around DiffSynth `AttentionModule.forward(q, k, v)`
  or an equivalent owned wrapper. Preserve cross-attention, image/audio/control branches, VACE branches, CFG behavior,
  and pipeline call signatures.
- If DiffSynth Unified Sequence Parallel has already patched `self_attn.forward`, do not silently claim support through
  the Wan `AttentionModule.forward(q, k, v)` path; fail clearly until a USP-aware SparseVideo path exists.
- Record `pipeline_backend="diffsynth"`, the exact `diffsynth` package version, patched module paths, backend counts,
  sparse dispatch counts, restore status, and resolved local DiffSynth bundle metadata in smoke/inference/quality
  metrics.
- `scripts/audit_parity.py` must read `outputs/diffsynth_method_smoke/**/*.jsonl` by default, report local native-bundle
  completeness through `diffsynth_checkpoint_availability`, and keep the `diffsynth_apply_restore_evidence` gate failing
  until every DiffSynth catalog model has `dense` plus all public sparse method apply/restore records.
- The DiffSynth apply/restore audit gate may count only `mode="apply_only"` and `mode="generate"` records. Dry-run,
  download, or other administrative records are not runtime evidence even if they include copied summary fields.
- DiffSynth smoke metrics should be replaced at run start by default so repeated validation of one model does not leave
  stale successful records behind a newer failed run; use `--append-metrics` only when intentionally collecting one
  metrics stream across multiple commands.
- DiffSynth generation smoke must fail when a public sparse method patches no attention path, records no sparse dispatch,
  or cannot restore the original attention path. A tiny `.mp4` alone is not evidence that SparseVideo executed.
  Sparse methods must patch every discovered DiffSynth self-attention path, not merely a nonzero subset; `dense` must
  patch zero modules.
- DiffSynth smoke generation is only for tiny no-media dispatch checks. Models that require images, video, or audio must
  fail before loading and use `--apply-only` for bundle/patch validation, or `scripts/infer_diffsynth.py` for a
  media-specific generation run.
- DiffSynth single-model inference must also validate required media inputs before loading local checkpoints, so a missing
  `--input-image`, `--input-audio`, `--vap-video`, or `--longcat-video` cannot waste GPU memory or I/O before failing.
  For non-`dense` methods, it must fail if apply patches no DiffSynth attention path, patches only part of the
  discovered DiffSynth self-attention paths, or generation records no sparse dispatch; a saved video is not enough
  evidence. `dense` must not patch DiffSynth attention modules.
- DiffSynth inference and smoke scripts must fail before model loading when requested shapes would be silently rounded by
  DiffSynth. Wan/MOVA runs require height and width multiples of 16 and `num_frames % 4 == 1`; LTX2 runs require height
  and width multiples of 32 and `num_frames % 8 == 1`.
- Keep DiffSynth generation kwargs aligned with the installed DiffSynth-Studio call signatures. For Wan 2.0.12 this means
  VACE masks are single image masks, not video-frame lists, and the inference entrypoint must expose native controls such
  as `cfg_merge`, `motion_bucket_id`, camera controls, tile size/stride, TeaCache controls, WanToDance inputs,
  sliding-window options, `framewise_decoding`, and `output_type`. For LTX-2 this includes input-image indexes/strength,
  in-context videos, retake video/audio regions, tile controls, and two-stage/distilled pipeline flags. Preserve
  pipeline-specific audio types: Wan S2V receives numpy/array audio, while LTX2 `retake_audio` receives the tensor
  returned by DiffSynth's audio loader.
- Do not assume Diffusers-format weights can be reused for all DiffSynth-Studio models. Some DiffSynth entries use native
  Wan checkpoints, component-level converted common files, component-level LTX-2/LTX-2.3 repackages, ModelScope repos,
  or state-dict converters. Use DiffSynth-native `ModelConfig` downloads for DiffSynth evidence unless a specific model
  entry is proven to accept the existing Diffusers checkpoint layout.
- `dense` remains the baseline for DiffSynth too. Sparse support is complete only when `dense` and each public sparse
  method can apply, run, record sparse dispatch, and restore on the target DiffSynth pipeline.

## Required Method Expansion

For CogVideoX, LTX Video, Allegro, Mochi, and EasyAnimate, support all of:
- `svg1`
- `svg2`
- `spargeattn`
- `radial`
- `sta`
- `draft`
- `adacluster`
- `flashomni`
- `svoo`

Do not count `dense` as sparse support. Do not claim a method is supported on a backbone until `sparsevideo.apply()` installs the correct processor or model patch, sparse dispatch is observed, and `restore()` returns the original dense path.

Current new-backbone evidence files:
- `svg1`: `.tmp_smoke/metrics_svg1_new_backbones_recheck.jsonl`
- `spargeattn`: `.tmp_smoke/metrics_spargeattn_new_backbones_recheck2.jsonl`
- `adacluster`: `.tmp_smoke/metrics_adacluster_new_backbones_recheck.jsonl`
- `radial`: `.tmp_smoke/metrics_radial_new_backbones_recheck.jsonl`, `.tmp_smoke/metrics_radial_new_backbones_recheck2.jsonl`, `.tmp_smoke/metrics_radial_cog_i2v_125f.jsonl`
- `sta`: `.tmp_smoke/metrics_sta_new_backbones_ok.jsonl`
- `draft`: `.tmp_smoke/metrics_draft_new_backbones.jsonl`
- `flashomni`: `.tmp_smoke/metrics_flashomni_new_backbones.jsonl`

## Architecture Rules

- Separate self-attn plus cross-attn: Wan family, SkyReels, LTX Video, Allegro. Apply sparse methods to video self-attn only; keep cross-attn dense.
- Concatenated text/video sequence: CogVideoX and EasyAnimate. Split text and video tokens, preserve text and text-video interactions as dense where the method cannot own them safely, apply sparse dispatch only to the video-video block, then recombine without changing output contracts.
- Joint dual-stream attention: HunyuanVideo and Mochi. Preserve the text stream and sparse only the video-video path unless the upstream method explicitly defines a safe joint-stream policy.
- Model-specific token counts, frame/height/width layout, masks, and timestep gates must come from the pipeline state or validated method config. No hard-coded shape hacks for a single smoke command.

## Method-Specific Requirements

- `svg1` and `svg2`: match the Sparse-VideoGen reference behavior for timestep warmup, token policy, clustering, permutation, and restoration. Config names must stay compatible with the current public registry.
- `spargeattn`: support every target backbone or fail preflight with a precise architectural reason. It must own Q/K/V and sparse dispatch, not only wrap a dense attention call.
- `radial`: implement the actual radial token layout/mask behavior for each target token layout. Dense masked approximation is not enough.
- `sta`: provide model-aware spatial/temporal layout, mask strategy, and available native backend dispatch. A100/current-hardware support is required; H100/TK artifacts are required only when Hopper hardware is available.
- `draft`: implement the upstream reorder, draft selection, and restore path end to end for the target sequence layout.
- `adacluster`: implement adaptive clustering against the model's video-token layout and preserve output ordering.
- `flashomni`: become turnkey for supported backbones by generating or sourcing the required sparse-info tensors in the method path. A caller-only missing-tensor path is not support.
- `svoo`: keep the existing smoke-passing behavior and extend any model-specific warmup/config gaps discovered while enabling the other methods.

## Implementation Constraints

- `training_free/*` is the reference for method behavior. Compare execution paths, not only method names or formulas.
- Keep `LIMITED_METHODS_BY_MODEL_TYPE` or equivalent guards until each model-method pair has evidence. Removing guards is the last step, not the first.
- Keep one public API and the existing registry/config flow. Do not add per-model public method names.
- Optional native kernels must not be required for basic import. CPU or dense fallbacks may exist for tests, but benchmark/support claims require real sparse backend dispatch.
- Touch only the files needed for detection, processors, method behavior, preflight, inference aliases, tests, and audit gates.

## Verification Gate

The goal is complete only when all required target model-method pairs satisfy these checks:
- `sparsevideo.apply(pipe, method=...)` accepts the method for the model without unsupported-method guards.
- The method installs the expected processor or model patch and `restore()` returns the dense processor/path.
- A real smoke run records `status="ok"`, non-empty `backend_counts`, and `dispatch_counts.sparse > 0` for sparse methods.
- The audit gate requires every target backbone to have `dense` plus all public sparse methods in smoke evidence, not only `svg2/svoo`.
- Existing Wan, HunyuanVideo, SkyReels, CogVideoX, LTX Video, Allegro, Mochi, and EasyAnimate smoke evidence must not regress.
- Method-level quality/strict-dispatch gates remain intact before making performance or parity claims.

## Priority

1. Build a model-method gap matrix from the current guards and audit evidence.
2. Port one method at a time across the seven reduced-surface backbones, starting with the least model-specific path.
3. Add focused unit tests for token splitting, layout restoration, processor restore, preflight, and config acceptance.
4. Run multi-GPU smoke experiments in parallel when GPUs are idle, with one large model run per GPU when resource-safe.
5. Update `scripts/audit_parity.py` so incomplete model-method pairs fail the all-backbone gate until evidence exists.
