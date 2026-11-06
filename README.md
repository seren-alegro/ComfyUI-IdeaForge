# ComfyUI-IdeaForge

Standalone ComfyUI sampler nodes built from published mathematical
concepts. The ideas are referenced; every implementation here is
an original clean-room implementation written for this pack.

All samplers follow the ComfyUI 0.44 sampler contract and use a
built-in Anima-style flow linear-shift sigma grid (an optional
`sigmas` input overrides it). Warp Sigma Scheduler is
model-agnostic.

## Nodes

| Node | Concept source | Cost | Notes |
| --- | --- | --- | --- |
| Overshoot Sampler | AMO Sampler (arXiv:2411.19415), base overshooting mechanism | 1 NFE/step | Oversimulates each step by (1+overshoot)×, then re-noises back to the target sigma's marginal. Attention modulation from the paper is not included. |
| ERK Guidance Sampler | "Error as Signal" (ERK-Guid) | 2 NFE/step | Heun step with an embedded Euler estimate; the pair difference is the solver's own local error estimate and is added back as a guidance direction scaled by erk_strength (0 = pure Heun). |
| Flow Taylor Sampler | RF-Solver (arXiv:2411.04746), Taylor-expansion idea | 1 NFE/step | Second-order step: velocity derivative estimated by finite difference from the previous step. |
| Warp Sigma Scheduler | Align Your Steps (arXiv:2404.14507), schedule-shape spirit | – | SIGMAS node: takes any built-in scheduler grid for the connected model and warps point placement (detail = denser low-sigma, smooth = denser ends). No per-model optimisation is performed. |
| Density Sampler | Diffusion-density / mode-tracking writeup (Karczewski) | 2 NFE/step | Experimental approximation: a probe evaluation measures how the prediction field bends away from the current data estimate, and the step is corrected against that bend. |

## KSampler registration

On load, the pack also registers four of the samplers into the
stock KSampler sampler list — `overshoot`, `erk_guidance`,
`flow_taylor`, `density_probe` — with fixed default parameters
(overshoot 0.5, erk_strength 0.5, density 0.3 / probe 0.05), so
they can be picked from any ordinary KSampler (scheduler `simple`
recommended). The standalone nodes remain the place for adjustable
knobs. Registration is wrapped so a failure can never break node
loading.

## Verified on real hardware — and one bug worth writing down

Tested on an RTX PRO 6000 (MIG 24 GB) pod, mac studio m3 ultra 512gb, ComfyUI 0.44,
Chiarezza 2.9B hybrid: all four samplers generate cleanly, via
both the standalone nodes and the KSampler registration.

Overshoot's first version diverged into neon blow-out even at
overshoot 0.2. The cause: its re-noise step restored the marginal
with the VP-diffusion formula, which over-weights the signal
(x0) component under flow interpolation — a small, systematic
per-step bias that compounds. The flow-correct step subtracts
`(sigma_next − s_over) · denoised` after re-noising; a numpy
check of the one-step statistics (signal coefficient 0.348
before, 0.298 after, target 0.300) confirmed the fix before the
pod re-test passed at overshoot 0.5.(same mac studio)

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/seren-alegro/ComfyUI-IdeaForge
# restart ComfyUI
```

Or copy this folder as `ComfyUI-IdeaForge` into `ComfyUI/custom_nodes`
manually, then restart ComfyUI.

## Tests

`tests/` — math tests for the stepping rules (run where torch
exists).

## License

MIT

---

## 한국어 요약

논문·글에서 본 수학 구상 다섯 개를 깨끗하게 다시 구현한 실험
샘플러 팩이에요. Overshoot, ERK Guidance, Flow Taylor, Density
Probe 샘플러와 Warp Sigma 스케줄러가 들어 있고, 앞의 네 샘플러는
일반 KSampler 목록에도 등록돼요. 전 샘플러 포드와 mac studio m3 ultra 512gb으로 실측을 마쳤고,
Overshoot의 flow용 재노이즈 보정 과정도 위에 기록해뒀어요.
