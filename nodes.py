"""Sampler Ideas - standalone samplers built from published concepts.

Ideas are referenced, implementations are our own (clean-room):
  1. OvershootSampler  - oversimulate the ODE past the target sigma,
     then re-noise back to the target marginal (AMO Sampler paper's
     base mechanism, arXiv:2411.19415; attention modulation omitted).
  2. ERKGuidSampler    - Heun with an embedded Euler estimate; the
     pair difference (the solver's own local error estimate) is used
     as an extra guidance direction ("Error as Signal", ERK-Guid).
  3. FlowTaylorSampler - second-order Taylor step for rectified flow:
     the velocity's sigma-derivative is estimated by finite difference
     from the previous step (RF-Solver's Taylor-expansion idea,
     arXiv:2411.04746).
  4. WarpSigmaScheduler - a SIGMAS node that takes any built-in
     scheduler grid and warps its point placement (schedule-shape
     optimisation in the spirit of Align Your Steps, arXiv:2404.14507;
     no model-specific optimisation is performed).
  5. DensitySampler    - experimental approximation of the density /
     mode-tracking idea (Karczewski's diffusion-density writeup): a
     second model evaluation probes how the prediction field bends
     away from the data point, and the step is corrected against it.

All samplers are standalone nodes on the Anima flow sigma grid
(except WarpSigmaScheduler, which is model-agnostic), following the
ComfyUI 0.44 sampler contract: prepare_noise takes the samples
tensor, and the step callback receives a single dict.
"""

import torch

import comfy.sample
import comfy.samplers


def flow_linear_shift_sigmas(steps, shift, denoise):
    t = torch.linspace(float(denoise), 0.0, steps + 1)
    sigmas = shift * t / (1.0 + (shift - 1.0) * t)
    sigmas[-1] = 0.0
    return sigmas



def _model_wants_temporal(model):
    dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    mod = type(dm).__module__ if dm is not None else ""
    return "cosmos" in mod or "anima" in mod


def _run(model, positive, negative, latent_image, seed, cfg,
         sampler_function, sigmas, label):
    latent_samples = comfy.sample.fix_empty_latent_channels(
        model, latent_image["samples"])
    _expand5d = (_model_wants_temporal(model)
                 and latent_samples.ndim == 4)
    if _expand5d:
        latent_samples = latent_samples.unsqueeze(2)
    noise = comfy.sample.prepare_noise(latent_samples, seed)
    if _expand5d:
        noise = noise.unsqueeze(2)
    sampler = comfy.samplers.KSAMPLER(sampler_function)
    print(f"[IdeaForge] {label}")
    samples = comfy.sample.sample_custom(
        model, noise, cfg, sampler, sigmas, positive, negative,
        latent_samples, noise_mask=latent_image.get("noise_mask"),
        callback=None, disable_pbar=False, seed=seed)
    out = dict(latent_image)
    if _expand5d:
        samples = samples.squeeze(2)
    out["samples"] = samples
    return (out,)


def _base_inputs(extra):
    required = {
        "model": ("MODEL",),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "latent_image": ("LATENT",),
        "seed": ("INT", {"default": 0, "min": 0,
                         "max": 0xffffffffffffffff}),
        "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
        "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0,
                          "step": 0.1}),
    }
    required.update(extra)
    required["flow_shift"] = ("FLOAT", {"default": 3.0, "min": 0.5,
                                        "max": 10.0, "step": 0.1})
    required["denoise"] = ("FLOAT", {"default": 1.0, "min": 0.0,
                                     "max": 1.0, "step": 0.01})
    return {"required": required,
            "optional": {"sigmas": ("SIGMAS",)}}


# ---------------------------------------------------------------- 1
def overshoot_sigma(sigma, sigma_next, overshoot):
    """Sigma reached by oversimulating (1 + overshoot) x the step."""
    s = sigma + (1.0 + overshoot) * (sigma_next - sigma)
    return max(float(s), 0.0)


def _make_overshoot_function(overshoot, seed):
    def sampler_function(model, x, sigmas, extra_args=None,
                         callback=None, disable=None, **kwargs):
        extra_args = extra_args or {}
        sigmas = sigmas.to(x.device)
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            denoised = model(x, sigma * x.new_ones([x.shape[0]]),
                             **extra_args)
            s_over = overshoot_sigma(sigma, sigma_next, overshoot)
            if s_over == 0.0:
                x = denoised
            else:
                v = (x - denoised) / sigma
                x = x + v * (s_over - float(sigma))
                var = float(sigma_next) ** 2 - s_over ** 2
                if var > 0.0:
                    gen = torch.Generator(device=x.device)
                    gen.manual_seed((int(seed) * 100003 + i) % (1 << 63))
                    z = torch.randn(x.shape, generator=gen,
                                    device=x.device, dtype=x.dtype)
                    x = x + (var ** 0.5) * z \
                        - (float(sigma_next) - s_over) * denoised
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma,
                          "denoised": denoised})
        return x

    return sampler_function


class OvershootSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return _base_inputs({
            "overshoot": ("FLOAT", {"default": 0.5, "min": 0.0,
                                    "max": 1.0, "step": 0.05}),
        })

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/ideas"

    def sample(self, model, positive, negative, latent_image, seed,
               steps, cfg, overshoot, flow_shift, denoise, sigmas=None):
        if sigmas is None:
            sigmas = flow_linear_shift_sigmas(steps, flow_shift, denoise)
        return _run(model, positive, negative, latent_image, seed, cfg,
                    _make_overshoot_function(overshoot, seed), sigmas,
                    f"Overshoot o={overshoot}")


# ---------------------------------------------------------------- 2
def _make_erk_function(strength):
    def sampler_function(model, x, sigmas, extra_args=None,
                         callback=None, disable=None, **kwargs):
        extra_args = extra_args or {}
        sigmas = sigmas.to(x.device)
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            d = sigma_next - sigma
            denoised = model(x, sigma * x.new_ones([x.shape[0]]),
                             **extra_args)
            if float(sigma_next) == 0.0:
                x = denoised
            else:
                v0 = (x - denoised) / sigma
                x_euler = x + v0 * d
                denoised1 = model(x_euler,
                                  sigma_next * x_euler.new_ones([x_euler.shape[0]]),
                                  **extra_args)
                v1 = (x_euler - denoised1) / sigma_next
                x_heun = x + 0.5 * (v0 + v1) * d
                error_dir = x_heun - x_euler
                x = x_heun + strength * error_dir
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma,
                          "denoised": denoised})
        return x

    return sampler_function


class ERKGuidSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return _base_inputs({
            "erk_strength": ("FLOAT", {"default": 0.5, "min": -1.0,
                                       "max": 2.0, "step": 0.05}),
        })

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/ideas"

    def sample(self, model, positive, negative, latent_image, seed,
               steps, cfg, erk_strength, flow_shift, denoise, sigmas=None):
        if sigmas is None:
            sigmas = flow_linear_shift_sigmas(steps, flow_shift, denoise)
        return _run(model, positive, negative, latent_image, seed, cfg,
                    _make_erk_function(erk_strength), sigmas,
                    f"ERKGuid strength={erk_strength} (2 NFE/step)")


# ---------------------------------------------------------------- 3
def _make_taylor_function():
    def sampler_function(model, x, sigmas, extra_args=None,
                         callback=None, disable=None, **kwargs):
        extra_args = extra_args or {}
        sigmas = sigmas.to(x.device)
        v_prev = None
        sigma_prev = None
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            d = sigma_next - sigma
            denoised = model(x, sigma * x.new_ones([x.shape[0]]),
                             **extra_args)
            if float(sigma_next) == 0.0:
                x = denoised
            else:
                v = (x - denoised) / sigma
                if v_prev is not None and float(sigma - sigma_prev) != 0.0:
                    dv = (v - v_prev) / (sigma - sigma_prev)
                    x = x + v * d + 0.5 * dv * d * d
                else:
                    x = x + v * d
                v_prev = v
                sigma_prev = sigma
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma,
                          "denoised": denoised})
        return x

    return sampler_function


class FlowTaylorSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return _base_inputs({})

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/ideas"

    def sample(self, model, positive, negative, latent_image, seed,
               steps, cfg, flow_shift, denoise, sigmas=None):
        if sigmas is None:
            sigmas = flow_linear_shift_sigmas(steps, flow_shift, denoise)
        return _run(model, positive, negative, latent_image, seed, cfg,
                    _make_taylor_function(), sigmas, "FlowTaylor (2nd)")


# ---------------------------------------------------------------- 4
def warp_sigmas(base, curve, strength):
    """Resample a base sigma grid (ending in 0) at warped positions."""
    body = base[:-1]
    m = len(body)
    if m < 2 or strength == 0.0 or curve == "linear":
        return base
    logs = torch.log(body.float())
    t = torch.linspace(0.0, 1.0, m)
    if curve == "detail":
        warped = 1.0 - (1.0 - t) ** 2
    elif curve == "smooth":
        warped = 0.5 - 0.5 * torch.cos(torch.pi * t)
    else:
        warped = t
    pos = ((1.0 - strength) * t + strength * warped) * (m - 1)
    lo = pos.floor().long().clamp(max=m - 2)
    frac = (pos - lo.float()).to(logs.dtype)
    vals = logs[lo] * (1.0 - frac) + logs[lo + 1] * frac
    out = torch.exp(vals)
    return torch.cat([out, base.new_zeros(1)]).to(base.dtype)


class WarpSigmaScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "scheduler": (["simple", "sgm_uniform", "karras",
                           "exponential", "ddim_uniform", "beta",
                           "normal", "linear_quadratic", "kl_optimal"],),
            "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0,
                                  "max": 1.0, "step": 0.01}),
            "curve": (["linear", "detail", "smooth"],),
            "curve_strength": ("FLOAT", {"default": 0.5, "min": 0.0,
                                         "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "make"
    CATEGORY = "sampling/ideas"

    def make(self, model, scheduler, steps, denoise, curve,
             curve_strength):
        model_sampling = model.get_model_object("model_sampling")
        if denoise < 1.0:
            total = int(steps / denoise)
            base = comfy.samplers.calculate_sigmas(
                model_sampling, scheduler, total)
            base = base[-(steps + 1):]
        else:
            base = comfy.samplers.calculate_sigmas(
                model_sampling, scheduler, steps)
        sigmas = warp_sigmas(base, curve, curve_strength)
        print(f"[IdeaForge] WarpSigma base={scheduler}, "
              f"curve={curve}, strength={curve_strength}")
        return (sigmas,)


# ---------------------------------------------------------------- 5
def _make_density_function(strength, probe_eps):
    def sampler_function(model, x, sigmas, extra_args=None,
                         callback=None, disable=None, **kwargs):
        extra_args = extra_args or {}
        sigmas = sigmas.to(x.device)
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            d = sigma_next - sigma
            denoised = model(x, sigma * x.new_ones([x.shape[0]]),
                             **extra_args)
            if float(sigma_next) == 0.0:
                x = denoised
            else:
                away = x - denoised
                norm = away.flatten().norm()
                if float(norm) > 0.0:
                    probe = x + probe_eps * away / norm * float(sigma)
                    denoised_probe = model(probe,
                                           sigma * probe.new_ones([probe.shape[0]]),
                                           **extra_args)
                    bend = denoised_probe - denoised
                    denoised = denoised - strength * bend
                v = (x - denoised) / sigma
                x = x + v * d
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma,
                          "denoised": denoised})
        return x

    return sampler_function


class DensitySampler:
    @classmethod
    def INPUT_TYPES(cls):
        return _base_inputs({
            "density_strength": ("FLOAT", {"default": 0.3, "min": 0.0,
                                           "max": 2.0, "step": 0.05}),
            "probe_eps": ("FLOAT", {"default": 0.05, "min": 0.01,
                                    "max": 0.5, "step": 0.01}),
        })

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/ideas"

    def sample(self, model, positive, negative, latent_image, seed,
               steps, cfg, density_strength, probe_eps, flow_shift,
               denoise, sigmas=None):
        if sigmas is None:
            sigmas = flow_linear_shift_sigmas(steps, flow_shift, denoise)
        return _run(model, positive, negative, latent_image, seed, cfg,
                    _make_density_function(density_strength, probe_eps),
                    sigmas,
                    f"Density strength={density_strength} (2 NFE/step)")





NODE_CLASS_MAPPINGS = {
    "OvershootSampler": OvershootSampler,
    "ERKGuidSampler": ERKGuidSampler,
    "FlowTaylorSampler": FlowTaylorSampler,
    "WarpSigmaScheduler": WarpSigmaScheduler,
    "DensitySampler": DensitySampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OvershootSampler": "Overshoot Sampler",
    "ERKGuidSampler": "ERK Guidance Sampler (Heun+Error)",
    "FlowTaylorSampler": "Flow Taylor Sampler (2nd order)",
    "WarpSigmaScheduler": "Warp Sigma Scheduler",
    "DensitySampler": "Density Sampler (experimental)",
}


# ---------------------------------------------------------- KSampler
# dropdown registration with fixed default parameters (overshoot
# 0.5, erk_strength 0.5, density 0.3). Standalone nodes remain the
# way to tune the knobs.
def _overshoot_registered(model, x, sigmas, extra_args=None,
                          callback=None, disable=None, **kwargs):
    seed = (extra_args or {}).get("seed", 0)
    return _make_overshoot_function(0.5, seed)(
        model, x, sigmas, extra_args=extra_args, callback=callback,
        disable=disable, **kwargs)


def register_samplers():
    try:
        import comfy.samplers
        from comfy.k_diffusion import sampling as _k_sampling
        entries = {
            "overshoot": _overshoot_registered,
            "erk_guidance": _make_erk_function(0.5),
            "flow_taylor": _make_taylor_function(),
            "density_probe": _make_density_function(0.3, 0.05),
        }
        for name, fn in entries.items():
            if name not in comfy.samplers.KSampler.SAMPLERS:
                comfy.samplers.KSampler.SAMPLERS.append(name)
            setattr(_k_sampling, f"sample_{name}", fn)
        print(f"[IdeaForge] registered KSampler samplers: "
              f"{', '.join(entries)}")
    except Exception as e:  # never break node loading
        print(f"[IdeaForge] sampler registration skipped: {e}")
