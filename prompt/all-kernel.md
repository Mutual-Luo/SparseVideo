# SparseVideo Kernel Acceleration Target

目标：SparseVideo 的所有 sparse attention methods 都应该尽可能走最快的
SparseVideo-owned Triton/C++/CUDA kernel 路径，而不是只保证功能正确或只在
某几个方法上启用优化。

## 核心需求

- 每个 method 都要明确记录实际使用了哪些快速路径：
  - fused QK RMSNorm / LayerNorm
  - fused RoPE
  - fast block / forward patch
  - sparse attention backend
  - method-specific Triton/C++/CUDA kernels
- 默认运行路径应该优先使用最快的 owned Triton/C++/CUDA implementation。
- 如果某个 method 不能启用某个 kernel，必须写清楚原因：
  upstream parity、质量风险、shape 限制、硬件限制、kernel 不存在、或当前未验证。
- 不能为了“跑通”默认依赖会严重损害速度的 CPU offload。
  CPU offload 只能作为显存兜底策略，并且需要单独标记 speed caveat。
- 不能从 `training_free/` 运行别人的代码作为 runtime。
  需要使用 `src/sparsevideo/` 下面 owned/copy/rewrite 的 kernel 和 runtime。
- 快速路径必须有 correctness/equivalence tests，不能只靠视频能生成或 import 成功。

## Dense / Full 阶段开销也必须审计

所有 sparse methods 都不能只看整体平均速度。很多方法都有 dense/full warmup
阶段、周期性 full refresh、或用于生成 sparse metadata/cache 的 full attention
阶段。审计时必须把这些阶段单独拆开看：

- 区分算法上必要的 full 阶段和实现引入的额外 overhead。
- full 阶段如果只是为了保持算法策略，可以优先考虑复用原生 dense attention
  快路径，而不是强制走 sparse backend 的 full-kernel wrapper。
- 如果 full 阶段必须走 method-specific kernel，需要证明它比原生 dense
  路径更快或至少没有明显变慢。
- sparse metadata/cache 生成不能在显存峰值处制造大临时张量；必要时做分块、
  kernel fusion、workspace/cache 复用，而不是默认 CPU offload。
- 速度报告必须拆分 dense/full dispatch、sparse dispatch、metadata/cache
  构建、decode/export，不能只报总 `generate_sec`。
- 这个要求适用于所有方法：`svg1`, `svg2`, `spargeattn`, `radial`, `sta`,
  `draft`, `adacluster`, `flashomni`, `svoo`，以及 dense baseline 的公平对照。

## Current Known Gaps To Resolve

1. `spargeattn`
   - 当前显式禁用了 fused QK norm/RoPE，以贴近 upstream 只替换 attention kernel。
   - 需要增加 speed-first 可控路径，并验证 fused norm/RoPE 是否等价且不影响质量。

2. `svoo` Wan
   - 当前启用了 fused QK norm，但禁用了 fused RoPE。
   - 需要验证打开 Triton/native RoPE 是否等价、是否更快；若不能默认开启，写清原因。

3. Wan fast-block patch
   - 当前主要由 SVOO Wan 使用 `wan_fast_block_patch`。
   - 需要评估并尽量推广到其他 Wan methods：
     `svg1`, `svg2`, `radial`, `draft`, `adacluster`, `sta`, `flashomni`。

4. Hunyuan fast-block path
   - 当前 Hunyuan attention processor 有 fused RMSNorm/RoPE，但没有通用 Hunyuan
     block-level fast patch 覆盖所有 methods。
   - 需要评估是否能做通用 fast block/forward acceleration，并保证 parity。

5. `flashomni` Hunyuan 720p benchmark
   - 目标规格：Hunyuan + FlashOmni + 720x1280 + 129 frames + 50 steps + single
     A100 80GB。
   - 需要尽量在不严重牺牲速度的情况下跑通。
   - `use_sparse_gemm=false` 或 CPU offload 可以作为诊断/兜底，但不能直接当作最快完成状态。
   - 需要记录速度、显存峰值、dispatch counts、是否使用 FlashOmni C++/CUDA kernels。

6. `sta_h100`
   - H100/TK path 因当前机器只有 A100，属于 hardware-deferred。
   - A100 上应使用并验证 FastVideo Triton fallback；H100 需要未来有硬件后再验。

## Success Criteria

- 产出一个 per-method kernel audit，列出每个 method 在 Wan/Hunyuan 上是否启用：
  `fused_qk_norm`, `fused_rope`, `fast_block_patch`, `attention_backend`,
  `native_kernel_backend`, `known_fallbacks`, `reason_if_not_fastest`。
- per-method audit 必须单独记录 dense/full warmup 或 refresh 阶段：
  `full_dispatch_backend`, `full_stage_overhead`, `metadata_cache_overhead`,
  `sparse_dispatch_backend`, `full_vs_native_dense_decision`。
- 每个 method 的最快路径都有最小 correctness/equivalence test。
- 对可能影响质量的 fast path，至少有 50-step real video inference 证据。
- Hunyuan FlashOmni 720p/129f/50-step 单 A100 目标需要单独验证：
  - 优先 no CPU offload 或轻量显存优化；
  - 如果必须 CPU offload，必须报告速度损失，不能把它当作 speed benchmark；
  - 输出 metrics 必须包含 runtime dispatch 和 CUDA peak memory。
- 最终状态不能只说 audit complete；必须说明哪些方法真正用上最快 kernel，
  哪些仍是未验证、hardware-deferred、或 speed/quality tradeoff。

## Engineering Constraints

- 不破坏 upstream method parity。
- 不用 `training_free/` 作为 runtime。
- 不为了速度 silently fallback 到近似实现。
- 默认路径应该快，但必须可恢复 dense baseline。
- 每次改动后跑最窄有用测试，并记录命令和结果。
