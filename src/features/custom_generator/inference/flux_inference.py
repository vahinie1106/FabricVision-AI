"""Run FLUX.1-Kontext image-conditioned garment generation with VRAM tracking."""

from __future__ import annotations

import gc
import inspect
import logging
import os
import random
import time
from typing import Any, Callable, Dict, Optional

from PIL import Image, ImageDraw

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None


ProgressCallback = Optional[Callable[[str, int], None]]

# CLIP-safe prompts are ~70 tokens; T5 default of 512 was measured at ~360s encode
# under model_cpu_offload on RTX 3050. 128 preserves full prompt semantics with far less work.
DEFAULT_MAX_SEQUENCE_LENGTH = 128

# Safe completion-first defaults. Kaggle T4×2 Production target is 768×768.
# Local low-VRAM (RTX 3050) stays at 512 unless explicitly overridden.
ALLOWED_FLUX_GENERATION_RESOLUTIONS = (384, 512, 640, 704, 720, 768, 1024)
DEFAULT_FLUX_GENERATION_RESOLUTION = 512
DEFAULT_KAGGLE_PRODUCTION_RESOLUTION = 768
DEFAULT_KAGGLE_PRODUCTION_STEPS = 12
DEFAULT_KAGGLE_PRODUCTION_GUIDANCE = 3.0
MIN_KAGGLE_PRODUCTION_RESOLUTION = 700


def resolve_flux_generation_resolution(default: Optional[int] = None) -> int:
    """Square resolution for FLUX Kontext (global / Standard / Preview override)."""
    fallback = (
        DEFAULT_FLUX_GENERATION_RESOLUTION if default is None else int(default)
    )
    for key in ("FLUX_GENERATION_RESOLUTION",):
        raw = os.environ.get(key, "").strip()
        if not raw.isdigit():
            continue
        size = int(raw)
        if size in ALLOWED_FLUX_GENERATION_RESOLUTIONS:
            return size
    if fallback in ALLOWED_FLUX_GENERATION_RESOLUTIONS:
        return fallback
    return DEFAULT_FLUX_GENERATION_RESOLUTION


def resolve_flux_production_resolution(default: Optional[int] = None) -> int:
    """
    Production-only square resolution.

    Priority: FLUX_PRODUCTION_RESOLUTION → FLUX_PRODUCTION_SIZE →
    FLUX_GENERATION_RESOLUTION → default (Kaggle locked target: 768).
    """
    fallback = (
        DEFAULT_KAGGLE_PRODUCTION_RESOLUTION if default is None else int(default)
    )
    for key in (
        "FLUX_PRODUCTION_RESOLUTION",
        "FLUX_PRODUCTION_SIZE",
        "FLUX_GENERATION_RESOLUTION",
    ):
        raw = os.environ.get(key, "").strip()
        if not raw.isdigit():
            continue
        size = int(raw)
        if size in ALLOWED_FLUX_GENERATION_RESOLUTIONS:
            return size
    if fallback in ALLOWED_FLUX_GENERATION_RESOLUTIONS:
        return fallback
    # Snap upward to nearest allowed ≥700 when caller passes e.g. 700.
    if fallback >= MIN_KAGGLE_PRODUCTION_RESOLUTION:
        for size in ALLOWED_FLUX_GENERATION_RESOLUTIONS:
            if size >= fallback:
                return size
    return DEFAULT_KAGGLE_PRODUCTION_RESOLUTION


def resolve_flux_production_steps(default: Optional[int] = None) -> int:
    """Production steps from FLUX_PRODUCTION_STEPS (Kaggle locked target: 12)."""
    fallback = DEFAULT_KAGGLE_PRODUCTION_STEPS if default is None else int(default)
    raw = os.environ.get("FLUX_PRODUCTION_STEPS", "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return max(1, int(fallback))


def resolve_flux_production_guidance(default: Optional[float] = None) -> float:
    """Production guidance from FLUX_PRODUCTION_GUIDANCE (Kaggle locked target: 3.0)."""
    fallback = (
        DEFAULT_KAGGLE_PRODUCTION_GUIDANCE if default is None else float(default)
    )
    raw = os.environ.get("FLUX_PRODUCTION_GUIDANCE", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return float(fallback)


class FLUXInferenceEngine:
    """FLUX.1-Kontext inference: fabric conditioning image + garment edit prompt."""

    def __init__(self, model_loader: Any, allow_fallback: bool = True) -> None:
        self.model_loader = model_loader
        self.allow_fallback = allow_fallback
        self.logger = logging.getLogger("fabricvision.garment_generation.inference")
        self.last_execution_stats: Dict[str, Any] = {}
        # Small prompt-embed cache: identical prompt → skip expensive T5 encode on reuse.
        # Keyed by (prompt, max_sequence_length). Cleared on loader change.
        self._prompt_embed_cache: Dict[tuple[str, int], tuple[Any, Any, Any]] = {}

    def _profile_enabled(self) -> bool:
        flag = os.environ.get("FLUX_PROFILE", "true").strip().lower()
        return flag not in ("0", "false", "no", "off")

    def _cpu_ram_mb(self) -> float:
        if psutil is None:
            return 0.0
        try:
            return round(psutil.Process(os.getpid()).memory_info().rss / (1024**2), 1)
        except Exception:
            return 0.0

    def _max_sequence_length(self) -> int:
        raw = os.environ.get("FLUX_MAX_SEQUENCE_LENGTH", "").strip()
        if raw.isdigit():
            return max(32, min(512, int(raw)))
        return DEFAULT_MAX_SEQUENCE_LENGTH

    def _clear_cuda(self) -> None:
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _evict_text_encoders(self, pipeline: Any) -> None:
        """
        Move CLIP/T5 to CPU after prompt encoding.

        Do NOT call ``maybe_free_model_hooks()`` here: that tears down accelerate
        offload hooks on the NF4 transformer and can cause
        ``invalid argument to getCurrentStream`` on later diffusion steps.
        """
        for attr in ("text_encoder", "text_encoder_2"):
            mod = getattr(pipeline, attr, None)
            if mod is None:
                continue
            try:
                if hasattr(mod, "to"):
                    mod.to("cpu")
            except Exception as exc:
                self.logger.debug("Encoder %s CPU move skipped: %s", attr, exc)
        self._clear_cuda()

    def _vram_snapshot(self) -> dict[str, float]:
        """CUDA allocator counters — not the same as dedicated physical VRAM.

        Under model_cpu_offload / Windows shared GPU memory, peak allocated can
        exceed the card's dedicated MiB. Callers must label these fields as
        CUDA allocator stats, not "physical VRAM used".
        """
        if torch is None or not torch.cuda.is_available():
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "max_allocated_mb": 0.0,
                "max_reserved_mb": 0.0,
            }
        max_reserved = 0.0
        try:
            max_reserved = round(torch.cuda.max_memory_reserved() / (1024**2), 1)
        except Exception:
            max_reserved = round(torch.cuda.memory_reserved() / (1024**2), 1)
        return {
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 1),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
            "max_reserved_mb": max_reserved,
        }

    def _log_vae_dtype_debug(self, pipeline: Any, conditioning_image: Any = None) -> None:
        """Targeted dtype/device dump immediately before pipeline()."""
        lines = ["[FLUX DTYPE DEBUG]"]
        lines.append(f"conditioning input type: {type(conditioning_image).__name__}")
        if conditioning_image is not None and hasattr(conditioning_image, "dtype"):
            lines.append(f"conditioning tensor dtype: {conditioning_image.dtype}")
            lines.append(f"conditioning tensor device: {getattr(conditioning_image, 'device', None)}")
        elif conditioning_image is not None:
            lines.append(
                f"conditioning image: mode={getattr(conditioning_image, 'mode', None)} "
                f"size={getattr(conditioning_image, 'size', None)} "
                "(PIL/object — Diffusers will tensorize then cast to pipeline dtype)"
            )
        vae = getattr(pipeline, "vae", None)
        if vae is not None:
            try:
                p = next(vae.parameters())
                lines.append(f"VAE dtype: {p.dtype}")
                lines.append(f"VAE device: {p.device}")
                lines.append(
                    f"VAE config force_upcast: {getattr(getattr(vae, 'config', None), 'force_upcast', None)}"
                )
                lines.append(f"VAE parameters dtype: {p.dtype}")
                lines.append(f"VAE parameters device: {p.device}")
            except Exception as exc:
                lines.append(f"VAE inspect failed: {exc}")
            conv_in = getattr(getattr(vae, "encoder", None), "conv_in", None)
            if conv_in is not None:
                w = getattr(conv_in, "weight", None)
                b = getattr(conv_in, "bias", None)
                if w is not None:
                    lines.append(f"VAE encoder conv_in weight dtype: {w.dtype}")
                    lines.append(f"VAE encoder conv_in weight device: {w.device}")
                if b is not None:
                    lines.append(f"VAE encoder conv_in bias dtype: {b.dtype}")
                    lines.append(f"VAE encoder conv_in bias device: {b.device}")
            lines.append(
                f"VAE.encode wrapped: {bool(getattr(vae, '_fabricvision_fp32_encode_wrapped', False))}"
            )
            lines.append(
                f"VAE.decode wrapped: {bool(getattr(vae, '_fabricvision_fp32_decode_wrapped', False))}"
            )
        for line in lines:
            self.logger.info(line)
            print(line, flush=True)

    def _log_tensor_stats(self, name: str, tensor: Any) -> None:
        """Compact tensor diagnostics — never dump full tensors."""
        if tensor is None or torch is None:
            self.logger.info("[FLUX DEBUG] %s: <none>", name)
            print(f"[FLUX DEBUG] {name}: <none>", flush=True)
            return
        try:
            t = tensor.detach()
            if t.is_cuda:
                t = t.float()
            else:
                t = t.float()
            total = t.numel()
            finite = int(torch.isfinite(t).sum().item()) if total else 0
            nan_n = int(torch.isnan(t).sum().item()) if total else 0
            inf_n = int(torch.isinf(t).sum().item()) if total else 0
            zero_n = int((t == 0).sum().item()) if total else 0
            line = (
                f"[FLUX DEBUG] {name} shape={tuple(t.shape)} dtype={tensor.dtype} "
                f"device={tensor.device} min={float(t.min()):.6g} max={float(t.max()):.6g} "
                f"mean={float(t.mean()):.6g} std={float(t.std()):.6g} "
                f"finite%={(100.0 * finite / total) if total else 0:.2f} "
                f"nan%={(100.0 * nan_n / total) if total else 0:.2f} "
                f"inf%={(100.0 * inf_n / total) if total else 0:.2f} "
                f"zero%={(100.0 * zero_n / total) if total else 0:.2f}"
            )
            self.logger.info(line)
            print(line, flush=True)
        except Exception as exc:
            self.logger.info("[FLUX DEBUG] %s: inspect_failed=%s", name, exc)

    def _log_pil_stats(self, name: str, image: Any) -> dict[str, float]:
        import numpy as np

        stats = {
            "min": -1.0,
            "max": -1.0,
            "mean": -1.0,
            "std": -1.0,
            "zero_pct": -1.0,
        }
        try:
            arr = np.asarray(image)
            stats = {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "zero_pct": float(np.mean(arr == 0) * 100.0),
            }
            line = (
                f"[FLUX DEBUG] {name} mode={getattr(image, 'mode', None)} "
                f"size={getattr(image, 'size', None)} dtype={arr.dtype} "
                f"min={stats['min']} max={stats['max']} mean={stats['mean']:.4f} "
                f"std={stats['std']:.4f} zero%={stats['zero_pct']:.2f}"
            )
            self.logger.info(line)
            print(line, flush=True)
        except Exception as exc:
            self.logger.info("[FLUX DEBUG] %s: inspect_failed=%s", name, exc)
        return stats

    @staticmethod
    def _assert_non_black_pil(image: Any, *, stage: str) -> None:
        import numpy as np

        arr = np.asarray(image)
        if arr.size == 0:
            raise RuntimeError(f"FLUX produced an empty image at stage={stage}")
        if int(arr.max()) == 0 or (float(arr.std()) == 0.0 and float(arr.mean()) == 0.0):
            raise RuntimeError(
                f"FLUX produced a completely black image before save "
                f"(stage={stage}, shape={arr.shape}, max={arr.max()}, "
                f"mean={arr.mean()}, std={arr.std()}). "
                f"Likely VAE fp16 NaN decode on T4 — VAE must run in float32."
            )

    def _install_vae_decode_probe(self, pipeline: Any) -> None:
        """Log latent/decode tensor stats once around VAE.decode (temporary probe)."""
        vae = getattr(pipeline, "vae", None)
        if vae is None or getattr(vae, "_fabricvision_decode_probe", False):
            return
        # Prefer the underlying decode if we already wrapped for dtype matching.
        original = vae.decode

        def _probed_decode(latents, *args, **kwargs):  # noqa: ANN001
            self._log_tensor_stats("LATENTS_BEFORE_VAE", latents)
            try:
                p = next(vae.parameters())
                self.logger.info(
                    "[FLUX DEBUG] VAE_AT_DECODE device=%s dtype=%s",
                    p.device,
                    p.dtype,
                )
                print(
                    f"[FLUX DEBUG] VAE_AT_DECODE device={p.device} dtype={p.dtype}",
                    flush=True,
                )
            except Exception:
                pass
            out = original(latents, *args, **kwargs)
            sample = out[0] if isinstance(out, (tuple, list)) else getattr(out, "sample", out)
            self._log_tensor_stats("VAE_DECODE_OUTPUT", sample)
            return out

        vae.decode = _probed_decode  # type: ignore[method-assign]
        vae._fabricvision_decode_probe = True

    def _park_pipeline(self, pipeline: Any) -> None:
        loader = self.model_loader
        if loader is not None and hasattr(loader, "park_on_cpu"):
            loader.park_on_cpu()
            return
        # Fallback path when loader has no park helper — still avoid NF4 .to(cpu)
        self._evict_text_encoders(pipeline)
        vae = getattr(pipeline, "vae", None)
        if vae is not None and hasattr(vae, "to"):
            try:
                vae.to("cpu")
            except Exception:
                pass
        self._clear_cuda()

    def _log_generation_stage(self, stage: str, **extra: Any) -> None:
        """Structured stage markers for Kaggle CUDA diagnosis."""
        snap = self._vram_snapshot()
        gpu_name = None
        total_mb = None
        if torch is not None and torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                total_mb = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**2), 1
                )
            except Exception:
                pass
        bits = " ".join(f"{k}={v}" for k, v in extra.items() if v is not None)
        line = (
            f"[FLUX GENERATION] {stage} "
            f"alloc_mb={snap['allocated_mb']} reserved_mb={snap['reserved_mb']} "
            f"max_alloc_mb={snap['max_allocated_mb']} "
            f"gpu={gpu_name} total_mb={total_mb}"
        )
        if bits:
            line = f"{line} {bits}"
        self.logger.info(line)
        print(line, flush=True)

    def _log_component_dtypes(self, pipeline: Any) -> None:
        """Log device/dtype of FLUX components (no full-model conversion)."""
        if torch is None:
            return
        for name in ("transformer", "vae", "text_encoder", "text_encoder_2"):
            mod = getattr(pipeline, name, None)
            if mod is None:
                continue
            try:
                p = next(mod.parameters())
                self.logger.info(
                    "[FLUX GENERATION] component=%s device=%s dtype=%s",
                    name,
                    p.device,
                    p.dtype,
                )
            except Exception as exc:
                self.logger.info(
                    "[FLUX GENERATION] component=%s inspect_failed=%s", name, exc
                )

    def _maybe_cuda_sync(self, label: str) -> None:
        """Optional synchronous CUDA for diagnosis only (FLUX_CUDA_SYNC_DEBUG=1)."""
        if os.environ.get("FLUX_CUDA_SYNC_DEBUG", "").strip().lower() not in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return
        if torch is None or not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
            self.logger.info("[FLUX GENERATION] CUDA_SYNC ok after %s", label)
        except Exception as exc:
            self.logger.error(
                "[FLUX GENERATION] CUDA_SYNC FAILED after %s: %s", label, exc
            )
            raise

    def _ensure_generation_devices(self, pipeline: Any) -> None:
        loader = self.model_loader
        if loader is not None and hasattr(loader, "ensure_generation_devices"):
            info = loader.ensure_generation_devices()
            self._log_generation_stage(
                "DEVICES_READY",
                restored=info.get("restored"),
                vae_device=info.get("vae_device"),
                transformer_device=info.get("transformer_device"),
                offload=info.get("offload_strategy"),
            )
            return
        # Fallback: if VAE was parked to CPU without accelerate hooks, put it back.
        if torch is None or not torch.cuda.is_available():
            return
        vae = getattr(pipeline, "vae", None)
        if vae is not None and hasattr(vae, "to"):
            try:
                vae.to("cuda")
            except Exception as exc:
                self.logger.warning("VAE → CUDA restore skipped: %s", exc)

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        name = type(exc).__name__.lower()
        msg = str(exc).lower()
        if "outofmemory" in name:
            return True
        return "cuda" in msg and ("out of memory" in msg or "oom" in msg)

    @staticmethod
    def _is_attention_kernel_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "no available kernel" in msg or "attention_kernel_unavailable" in msg

    def _physical_vram_mib(self) -> float:
        if torch is None or not torch.cuda.is_available():
            return 0.0
        try:
            return round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1)
        except Exception:
            return 0.0

    def _encode_prompt_timed(
        self,
        pipeline: Any,
        prompt: str,
        max_sequence_length: int,
        device: Any,
    ) -> tuple[Any, Any, Any, float, bool]:
        """
        Encode once with explicit timing.

        WHY: Under model_cpu_offload, T5 (text_encoder_2) dominates wall time when
        max_sequence_length=512 (~6 min measured). Encoding at 128 + caching identical
        prompts removes that bottleneck without changing garment semantics.

        Cache key is the full prompt string + sequence length. Safe because the
        prompt builder embeds garment metadata/customization into that string;
        different user requests produce different keys. Kontext image conditioning
        is passed separately and is never cached here.
        """
        cache_key = (prompt, max_sequence_length)
        if cache_key in self._prompt_embed_cache:
            embeds = self._prompt_embed_cache[cache_key]
            self.logger.info("[FLUX] Prompt embed cache HIT (skipping T5 encode)")
            return embeds[0], embeds[1], embeds[2], 0.0, True

        t0 = time.perf_counter()
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        encode_s = round(time.perf_counter() - t0, 3)

        # Always return/cache CPU tensors — do not keep T5 outputs on GPU into diffusion.
        try:
            prompt_embeds = prompt_embeds.detach().to("cpu")
            pooled_prompt_embeds = pooled_prompt_embeds.detach().to("cpu")
            text_ids_cpu = text_ids.detach().to("cpu") if text_ids is not None else None
            self._prompt_embed_cache[cache_key] = (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids_cpu,
            )
            if len(self._prompt_embed_cache) > 4:
                oldest = next(iter(self._prompt_embed_cache))
                del self._prompt_embed_cache[oldest]
        except Exception as exc:
            self.logger.warning("Prompt embed cache store skipped: %s", exc)
            text_ids_cpu = text_ids

        self._evict_text_encoders(pipeline)
        return prompt_embeds, pooled_prompt_embeds, text_ids_cpu, encode_s, False

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        reference_image: Optional[Image.Image] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
        guidance_scale: float = 2.5,
        seed: Optional[int] = 42,
        progress_callback: ProgressCallback = None,
        save_raw_path: Optional[str] = None,
        flux_input_audit_path: Optional[str] = None,
        color_trace: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        """Generate a garment with FLUX.1-Kontext image conditioning."""
        t_start = time.perf_counter()
        profile = self._profile_enabled()
        max_seq = self._max_sequence_length()

        def _progress(step: str, pct: int) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(step, pct)
                except Exception:
                    pass

        def _flux_mark(label: str, t_seg: float, *, end: bool = False) -> float:
            now = time.perf_counter()
            if end:
                msg = f"[FLUX] {label} END t={now:.2f} elapsed={now - t_seg:.2f}s"
            else:
                msg = f"[FLUX] {label} START t={now:.2f}"
            self.logger.info(msg)
            print(msg, flush=True)
            return now

        pipeline = getattr(self.model_loader, "pipeline", None)
        t_model_load_start = _flux_mark("model load", time.perf_counter())
        model_was_reused = pipeline is not None
        if pipeline is None and hasattr(self.model_loader, "load"):
            _progress("Loading model", 12)
            pipeline = self.model_loader.load()
            model_was_reused = False
        elif pipeline is not None:
            self.logger.info("[FLUX] Reusing loaded Kontext pipeline")
            if hasattr(self.model_loader, "load"):
                pipeline = self.model_loader.load()
        _flux_mark("model load", t_model_load_start, end=True)
        t_model_load_end = time.perf_counter()
        model_load_time = round(t_model_load_end - t_model_load_start, 3)

        if seed is not None:
            random.seed(seed)
            if torch is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        vram_before = (
            torch.cuda.memory_allocated() / (1024**2)
            if (torch and torch.cuda.is_available())
            else 0.0
        )
        ram_before = self._cpu_ram_mb()

        if pipeline is None:
            msg = "FLUX.1-Kontext pipeline not initialized or weights missing."
            self.logger.warning(msg)
            if not self.allow_fallback:
                raise RuntimeError(f"Real FLUX Kontext execution required but failed: {msg}")
            self.last_execution_stats = {
                "was_fallback_used": True,
                "was_real_flux_used": False,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_before,
                "peak_vram_mb": vram_before,
                "generation_time_s": 0.01,
                "model_load_time_s": 0.0,
                "prompt_encoding_time_s": 0.0,
                "inference_time_s": 0.0,
                "vae_decode_time_s": 0.0,
            }
            return self._generate_synthetic_preview(width, height, prompt)

        if reference_image is None:
            raise RuntimeError(
                "FLUX.1-Kontext requires a fabric conditioning image. "
                "Upload a fabric photo — text-only generation is not supported for this module."
            )

        runtime = {}
        if hasattr(self.model_loader, "get_runtime_info"):
            runtime = self.model_loader.get_runtime_info()

        physical_vram_mb = float(runtime.get("gpu_vram_mb") or 0.0)
        if physical_vram_mb <= 0 and torch is not None and torch.cuda.is_available():
            try:
                physical_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            except Exception:
                physical_vram_mb = 0.0

        self.logger.info("Kontext inference started (conditioning image present)")
        snap0 = self._vram_snapshot()
        self.logger.info(
            "[FLUX] Effective config: %sx%s steps=%s guidance=%s max_seq=%s "
            "preencode=%s offload=%s bnb4bit=%s dtype=%s alloc_conf=%s | "
            "CUDA allocated=%.1f MB reserved=%.1f MB (physical VRAM=%.0f MiB)",
            width,
            height,
            num_inference_steps,
            guidance_scale,
            max_seq,
            os.environ.get("FLUX_PREENCODE_PROMPT", "true"),
            runtime.get("offload_strategy"),
            runtime.get("bnb_4bit"),
            getattr(self.model_loader, "precision", "unknown"),
            os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            snap0["allocated_mb"],
            snap0["reserved_mb"],
            physical_vram_mb,
        )

        # Critical on 6GB: previous run may have left transformer/VAE on GPU.
        self._park_pipeline(pipeline)
        snap1 = self._vram_snapshot()
        self.logger.info(
            "[FLUX] After park_on_cpu: allocated=%.1f MB reserved=%.1f MB",
            snap1["allocated_mb"],
            snap1["reserved_mb"],
        )

        per_step_durations: list[float] = []
        step_wall: Dict[int, float] = {}
        last_step_t = time.perf_counter()
        encode_s = 0.0
        encode_cached = False
        resize_time = 0.0
        diffusion_s = 0.0
        decode_s = 0.0

        try:
            generator = None
            if seed is not None and torch is not None:
                # Prefer CPU generator — CUDA generators can force extra device syncs with offload
                generator = torch.Generator(device="cpu").manual_seed(seed)

            def step_callback(pipe: Any, step_index: int, timestep: Any, callback_kwargs: Any) -> Any:
                nonlocal last_step_t
                now = time.perf_counter()
                dur = round(now - last_step_t, 3)
                step_wall[step_index] = dur
                per_step_durations.append(dur)
                last_step_t = now
                self.logger.info(
                    "[FLUX PROFILE] Step %d/%d: %.3f sec",
                    step_index + 1,
                    num_inference_steps,
                    dur,
                )
                self._log_generation_stage(
                    "DENOISING_STEP",
                    step=step_index + 1,
                    steps=num_inference_steps,
                    step_s=dur,
                )
                pct = 50 + int(35 * (step_index + 1) / max(1, num_inference_steps))
                _progress(f"Generating (step {step_index + 1}/{num_inference_steps})", pct)
                return callback_kwargs

            _progress("Preparing fabric conditioning", 35)
            t_resize = _flux_mark("conditioning resize", time.perf_counter())
            if hasattr(reference_image, "resize"):
                if reference_image.size != (width, height):
                    cond_image = reference_image.resize((width, height), Image.Resampling.LANCZOS)
                else:
                    cond_image = reference_image
            else:
                cond_image = reference_image
            resize_time = round(time.perf_counter() - t_resize, 3)
            _flux_mark("conditioning resize", t_resize, end=True)
            cond_stats = self._log_pil_stats("IMAGE_CONDITIONING", cond_image)

            # Exact image that will be passed as pipeline(image=...) — not an earlier stage.
            flux_input_saved = None
            if flux_input_audit_path and hasattr(cond_image, "save"):
                try:
                    from pathlib import Path as _Path

                    _p = _Path(flux_input_audit_path)
                    _p.parent.mkdir(parents=True, exist_ok=True)
                    # Copy so later mutations cannot rewrite the audit artifact.
                    audit_img = cond_image.copy() if hasattr(cond_image, "copy") else cond_image
                    audit_img.save(_p)
                    flux_input_saved = str(_p)
                    self.logger.info(
                        "[FLUX COLOR TRACE] saved exact FluxKontext image= arg → %s",
                        flux_input_saved,
                    )
                    print(
                        f"[FLUX COLOR TRACE] conditioning_source_path={flux_input_saved}",
                        flush=True,
                    )
                except Exception as audit_exc:
                    self.logger.warning("flux_input audit save failed: %s", audit_exc)

            trace = color_trace or {}
            for line in (
                "[FLUX COLOR TRACE]",
                f"selected_ui_color={trace.get('selected_ui_color')}",
                f"color_mode={trace.get('color_mode')}",
                f"force_recolor={trace.get('force_recolor')}",
                f"target_color={trace.get('target_color')}",
                f"conditioning_recolored={trace.get('conditioning_recolored')}",
                f"conditioning_source_path={flux_input_saved}",
                f"conditioning_min={cond_stats.get('min')}",
                f"conditioning_max={cond_stats.get('max')}",
                f"conditioning_mean={cond_stats.get('mean')}",
                f"conditioning_std={cond_stats.get('std')}",
                f"flux_image_arg_is_recolored_conditioning={bool(trace.get('conditioning_recolored'))}",
            ):
                self.logger.info(line)
                print(line, flush=True)

            self.logger.info("=== FLUX KONTEXT CALL PROMPT ===\n%s", prompt)
            if negative_prompt:
                self.logger.info("=== FLUX KONTEXT NEGATIVE PROMPT ===\n%s", negative_prompt)

            signature = inspect.signature(pipeline.__call__)
            device = getattr(pipeline, "_execution_device", None)
            if device is None and torch is not None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Phase A — prefer letting the pipeline encode with a short T5 budget.
            # WHY not always pre-encode: holding T5 on GPU while transformer onloads OOMs 6GB.
            # max_sequence_length=128 alone dropped encode from ~360s → ~50s in measurement.
            _progress("Encoding prompt", 45)
            t_encode = _flux_mark("prompt encoding", time.perf_counter())
            prompt_embeds = pooled_prompt_embeds = None
            # Default true: measured encode ~50s at seq=128; cache + CPU eviction helps 6GB.
            use_preencode = os.environ.get("FLUX_PREENCODE_PROMPT", "true").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if use_preencode:
                try:
                    (
                        prompt_embeds,
                        pooled_prompt_embeds,
                        _text_ids,
                        encode_s,
                        encode_cached,
                    ) = self._encode_prompt_timed(pipeline, prompt, max_seq, device)
                    self.logger.info(
                        "[FLUX] Pre-encode done (%.3fs, cached=%s); encoders on CPU | %s",
                        encode_s,
                        encode_cached,
                        self._vram_snapshot(),
                    )
                except Exception as enc_exc:
                    # Do NOT fall through to inline encode while GPU is full —
                    # that double-OOMs. Park, retry once, then fail clearly.
                    if self._is_cuda_oom(enc_exc):
                        self.logger.warning(
                            "Pre-encode CUDA OOM (%s); parking pipeline and retrying once",
                            enc_exc,
                        )
                        self._park_pipeline(pipeline)
                        try:
                            (
                                prompt_embeds,
                                pooled_prompt_embeds,
                                _text_ids,
                                encode_s,
                                encode_cached,
                            ) = self._encode_prompt_timed(pipeline, prompt, max_seq, device)
                            self.logger.info(
                                "[FLUX] Pre-encode retry OK (%.3fs) | %s",
                                encode_s,
                                self._vram_snapshot(),
                            )
                        except Exception as retry_exc:
                            self._park_pipeline(pipeline)
                            raise RuntimeError(
                                f"CUDA out of memory during FLUX prompt encoding "
                                f"(T5/CLIP) after cleanup retry: {retry_exc}"
                            ) from retry_exc
                    else:
                        self.logger.warning(
                            "Explicit encode_prompt failed (%s); using inline encode",
                            enc_exc,
                        )
                        prompt_embeds = pooled_prompt_embeds = None
                        encode_s = 0.0

            _flux_mark("prompt encoding", t_encode, end=True)

            kwargs: Dict[str, Any] = {
                "image": cond_image,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "callback_on_step_end": step_callback,
                "max_area": int(height) * int(width),
                # Keep exact HxW — do not jump to 1024 preferred (OOM on 6GB).
                "_auto_resize": False,
                "max_sequence_length": max_seq,
            }

            if prompt_embeds is not None and pooled_prompt_embeds is not None:
                # Keep embeds on the execution device for the full denoise loop.
                # Leaving them on CPU caused step-2+ NF4 failures:
                # RuntimeError: invalid argument to getCurrentStream
                # (bitsandbytes saw A.device.index=None).
                try:
                    exec_dev = device
                    if exec_dev is None and torch is not None:
                        exec_dev = torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        )
                    if hasattr(prompt_embeds, "to"):
                        prompt_embeds = prompt_embeds.to(exec_dev)
                    if hasattr(pooled_prompt_embeds, "to"):
                        pooled_prompt_embeds = pooled_prompt_embeds.to(exec_dev)
                    self.logger.info(
                        "[FLUX] Prompt embeds on %s before diffusion | %s",
                        exec_dev,
                        self._vram_snapshot(),
                    )
                except Exception as move_exc:
                    self.logger.warning("Prompt embed device move failed: %s", move_exc)
                kwargs["prompt_embeds"] = prompt_embeds
                kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds
                if "prompt" in signature.parameters:
                    kwargs["prompt"] = None
            else:
                kwargs["prompt"] = prompt
                # encode_s will be inferred from step callback / pipeline timing
                encode_s = 0.0
                encode_cached = False

            true_cfg = float(os.environ.get("FLUX_TRUE_CFG_SCALE", "1.0"))
            if negative_prompt and "negative_prompt" in signature.parameters:
                if true_cfg > 1.0 and "true_cfg_scale" in signature.parameters:
                    kwargs["negative_prompt"] = negative_prompt
                    kwargs["true_cfg_scale"] = true_cfg
                    self.logger.info("Kontext true_cfg_scale=%s (negatives active)", true_cfg)
                else:
                    self.logger.info(
                        "Negative prompt prepared but true_cfg_scale=1.0 "
                        "(set FLUX_TRUE_CFG_SCALE=1.5+ on higher VRAM to activate)"
                    )

            if generator is not None and "generator" in signature.parameters:
                kwargs["generator"] = generator

            # Drop kwargs not accepted by this diffusers version
            kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters or k.startswith("_")}

            # Large-resolution memory path (768²+): VAE tiling/slicing are supported on
            # FluxKontextPipeline and materially reduce decode/activation peaks on T4-class GPUs.
            # Keep model_cpu_offload (sequential_cpu_offload is unsafe with bitsandbytes NF4).
            large_res = int(height) * int(width) >= 768 * 768
            force_tile = os.environ.get("FLUX_VAE_TILING", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if large_res or force_tile:
                vae = getattr(pipeline, "vae", None)
                if vae is not None:
                    if hasattr(vae, "enable_slicing"):
                        vae.enable_slicing()
                        self.logger.info("[FLUX] VAE slicing enabled (large-res / OOM mitigation)")
                    if hasattr(vae, "enable_tiling"):
                        vae.enable_tiling()
                        self.logger.info("[FLUX] VAE tiling enabled (large-res / OOM mitigation)")

            self.logger.info(
                "Kontext inputs: steps=%s guidance=%s seed=%s size=%sx%s "
                "has_image=True model_reused=%s max_seq=%s embeds_precomputed=%s",
                num_inference_steps,
                guidance_scale,
                seed,
                width,
                height,
                model_was_reused,
                max_seq,
                prompt_embeds is not None,
            )
            device_s = str(device)
            print(f"[FLUX] pipeline device = {device_s}", flush=True)
            print(f"[FLUX] resolution = {width}x{height}", flush=True)
            print(f"[FLUX] steps = {num_inference_steps}", flush=True)
            print(f"[FLUX] guidance = {guidance_scale}", flush=True)
            print(
                f"[FLUX] conditioning image = {flux_input_saved or '(in-memory)'} "
                f"size={getattr(cond_image, 'size', None)} "
                f"mode={getattr(cond_image, 'mode', None)}",
                flush=True,
            )
            print("[FLUX] inference started", flush=True)
            self.logger.info(
                "[FLUX] FINAL kwargs before pipeline(): height=%s width=%s "
                "steps=%s guidance=%s device=%s auto_resize=%s",
                kwargs.get("height"),
                kwargs.get("width"),
                kwargs.get("num_inference_steps"),
                kwargs.get("guidance_scale"),
                device_s,
                kwargs.get("_auto_resize"),
            )

            self._clear_cuda()
            from src.features.custom_generator.inference.flux_vram_policy import (
                log_vram,
                recommend_oom_fallback,
            )

            log_vram("before generation")
            self._log_generation_stage("START")
            self._ensure_generation_devices(pipeline)
            if self.model_loader is not None and hasattr(self.model_loader, "stabilize_flux_vae"):
                stab = self.model_loader.stabilize_flux_vae(pipeline)
                self._log_generation_stage(
                    "VAE_STABILIZED",
                    upcasted=stab.get("upcasted"),
                    vae_dtype=stab.get("vae_dtype_after"),
                    vae_device=stab.get("vae_device"),
                )
            self._install_vae_decode_probe(pipeline)
            self._log_component_dtypes(pipeline)
            self._maybe_cuda_sync("DEVICES_READY")
            _progress("Generating", 50)
            self._log_generation_stage("PIPELINE_READY")
            t_diff_start = _flux_mark("inference", time.perf_counter())
            last_step_t = t_diff_start
            # Suppress SDPA MATH on the Flux *transformer only*. Wrapping the whole
            # pipeline(...) breaks VAE encode/decode on T4 (head dims need MATH).
            # On T4 (sm_75), BF16 efficient SDPA is unavailable — Q/K/V are cast to
            # FP16 only inside the temporary SDPA wrap during transformer.forward.
            from src.features.custom_generator.inference.flux_attention import (
                NoSupportedEfficientAttention,
                configure_memory_efficient_attention,
                merge_runtime_attention_state,
                transformer_only_memory_efficient_attention,
            )

            attn_diag = getattr(self.model_loader, "_attention_diag", None) or {}
            if not attn_diag.get("attention_config_ok"):
                # Re-apply / validate on this pipeline (covers reused loaders).
                attn_diag = configure_memory_efficient_attention(pipeline)
                if self.model_loader is not None:
                    try:
                        self.model_loader._attention_diag = attn_diag
                        if attn_diag.get("attention_config_ok"):
                            self.model_loader._attention_backend = "memory_efficient_sdpa"
                    except Exception:
                        pass
            if not attn_diag.get("attention_config_ok"):
                err = attn_diag.get("error") or "NO_SUPPORTED_EFFICIENT_ATTENTION_BACKEND"
                raise NoSupportedEfficientAttention(
                    f"Cannot run FLUX Kontext without memory-efficient SDPA ({err}). "
                    "MATH attention would OOM at 1024 on 16GB-class GPUs."
                )

            def _run_pipeline_once(pipe_kwargs: Dict[str, Any]):
                self._log_generation_stage("INPUT_PREPARED")
                self._maybe_cuda_sync("INPUT_PREPARED")
                self._log_vae_dtype_debug(pipeline, pipe_kwargs.get("image"))
                log_vram("before transformer inference")
                self._log_generation_stage("DENOISING_START")
                try:
                    with transformer_only_memory_efficient_attention(pipeline) as attn_runtime:
                        out = pipeline(**pipe_kwargs)
                except Exception as pipe_exc:
                    self._log_generation_stage(
                        "FAILED",
                        stage_hint="DENOISING_OR_VAE_ENCODE",
                        error=f"{type(pipe_exc).__name__}: {pipe_exc}",
                    )
                    raise
                self._log_generation_stage("DENOISING_COMPLETE")
                self._maybe_cuda_sync("DENOISING_COMPLETE")
                return out, attn_runtime

            try:
                output, attn_runtime = _run_pipeline_once(kwargs)
            except Exception as denoise_exc:
                if not self._is_cuda_oom(denoise_exc):
                    raise
                self.logger.error(
                    "[VRAM] OOM during transformer/diffusion at %sx%s steps=%s: %s",
                    width,
                    height,
                    num_inference_steps,
                    denoise_exc,
                )
                log_vram("after OOM before cleanup")
                self._park_pipeline(pipeline)
                log_vram("after cleanup")
                # Production lock (Kaggle): fail clearly — never silently resize
                # 768×12 → smaller and still report the locked config as success.
                no_oom_fallback = os.environ.get(
                    "FLUX_PRODUCTION_NO_OOM_FALLBACK", ""
                ).strip().lower() in ("1", "true", "yes", "on")
                if no_oom_fallback:
                    raise RuntimeError(
                        f"CUDA out of memory at requested Production settings "
                        f"{width}x{height} steps={num_inference_steps} "
                        f"(FLUX_PRODUCTION_NO_OOM_FALLBACK=1 — no silent resize). "
                        f"Original error: {denoise_exc}"
                    ) from denoise_exc
                fallback = recommend_oom_fallback(
                    height=int(height),
                    width=int(width),
                    num_inference_steps=int(num_inference_steps),
                )
                if fallback is None:
                    raise
                self.logger.warning(
                    "[VRAM] Retrying generation once at safer settings %sx%s steps=%s "
                    "(was %sx%s steps=%s)",
                    fallback["width"],
                    fallback["height"],
                    fallback["num_inference_steps"],
                    width,
                    height,
                    num_inference_steps,
                )
                height = int(fallback["height"])
                width = int(fallback["width"])
                num_inference_steps = int(fallback["num_inference_steps"])
                kwargs["height"] = height
                kwargs["width"] = width
                kwargs["num_inference_steps"] = num_inference_steps
                if "max_area" in kwargs:
                    kwargs["max_area"] = int(height) * int(width)
                # Rebuild step callback bounds for the safer step count.
                def step_callback_retry(pipe, step_index, timestep, callback_kwargs):
                    nonlocal last_step_t
                    now = time.perf_counter()
                    step_dt = now - last_step_t
                    last_step_t = now
                    self.logger.info(
                        "[FLUX] Diffusion step %s/%s (%.2fs)",
                        step_index + 1,
                        num_inference_steps,
                        step_dt,
                    )
                    self._log_generation_stage(
                        "DENOISING_STEP",
                        step=step_index + 1,
                        steps=num_inference_steps,
                        step_s=round(step_dt, 2),
                    )
                    pct = 50 + int(35 * (step_index + 1) / max(1, num_inference_steps))
                    _progress(
                        f"Generating (retry {step_index + 1}/{num_inference_steps})",
                        pct,
                    )
                    return callback_kwargs

                if "callback_on_step_end" in kwargs:
                    kwargs["callback_on_step_end"] = step_callback_retry
                _progress(
                    f"Retrying after CUDA OOM at {width}x{height}",
                    48,
                )
                self._clear_cuda()
                self._ensure_generation_devices(pipeline)
                log_vram("before generation retry")
                try:
                    output, attn_runtime = _run_pipeline_once(kwargs)
                except Exception as retry_exc:
                    self._park_pipeline(pipeline)
                    log_vram("after retry cleanup")
                    if self._is_cuda_oom(retry_exc):
                        raise RuntimeError(
                            f"CUDA out of memory during FLUX transformer inference "
                            f"even after fallback to {width}x{height} "
                            f"steps={num_inference_steps}: {retry_exc}"
                        ) from retry_exc
                    raise

            attn_diag = merge_runtime_attention_state(attn_diag, attn_runtime)
            if self.model_loader is not None:
                try:
                    self.model_loader._attention_diag = attn_diag
                except Exception:
                    pass
            self.logger.info(
                "[FLUX] Attention runtime: dtype_req=%s dtype_eff=%s qkv=%s "
                "fallback=%s reason=%s",
                attn_diag.get("attention_dtype_requested"),
                attn_diag.get("attention_dtype_effective"),
                attn_diag.get("attention_qkv_dtype"),
                attn_diag.get("attention_fallback_used"),
                attn_diag.get("attention_fallback_reason"),
            )
            t_diff_end = time.perf_counter()
            diffusion_s = round(t_diff_end - t_diff_start, 3)
            _flux_mark("inference", t_diff_start, end=True)

            log_vram("before VAE decode")
            self._log_generation_stage("VAE_DECODE_START")
            _progress("Decoding image", 88)
            t_dec = _flux_mark("decoding", time.perf_counter())
            try:
                self.logger.info(
                    "[FLUX DEBUG] PIPELINE_OUTPUT type=%s attrs=%s",
                    type(output).__name__,
                    [a for a in ("images", "nsfw_content_detected") if hasattr(output, a)],
                )
                images = getattr(output, "images", None)
                self.logger.info(
                    "[FLUX DEBUG] images type=%s count=%s",
                    type(images).__name__ if images is not None else None,
                    len(images) if images is not None else 0,
                )
                image = output.images[0]
                self._log_pil_stats("PIL_ARRAY", image)
                self._assert_non_black_pil(image, stage="pipeline_output")
            except Exception as dec_exc:
                self._log_generation_stage(
                    "FAILED",
                    stage_hint="VAE_DECODE",
                    error=f"{type(dec_exc).__name__}: {dec_exc}",
                )
                if self._is_cuda_oom(dec_exc):
                    log_vram("OOM during VAE decode")
                    self._park_pipeline(pipeline)
                    raise RuntimeError(
                        f"CUDA out of memory during FLUX VAE decode: {dec_exc}"
                    ) from dec_exc
                raise
            decode_s = round(time.perf_counter() - t_dec, 3)
            _flux_mark("decoding", t_dec, end=True)
            self._log_generation_stage("VAE_DECODE_COMPLETE")
            log_vram("after VAE decode")
            out_w, out_h = image.size if hasattr(image, "size") else (width, height)
            out_mode = getattr(image, "mode", "?")
            print("[FLUX] inference completed", flush=True)
            print(f"[FLUX] output size = {out_w}x{out_h}", flush=True)
            print(f"[FLUX] output mode = {out_mode}", flush=True)

            # Raw model output before any UI path — for blur root-cause isolation
            if save_raw_path:
                try:
                    from pathlib import Path

                    raw_p = Path(save_raw_path)
                    raw_p.parent.mkdir(parents=True, exist_ok=True)
                    self._log_generation_stage("SAVE_START", path=str(raw_p))
                    self._log_pil_stats("RAW_SAVE", image)
                    self._assert_non_black_pil(image, stage="raw_save")
                    image.save(raw_p, format="PNG", compress_level=3)
                    raw_bytes = int(raw_p.stat().st_size) if raw_p.exists() else 0
                    self.logger.info("[FLUX] Saved raw model output → %s (%sx%s)", raw_p, image.size[0], image.size[1])
                    print(f"[FLUX] output saved = {raw_p}", flush=True)
                    print(f"[FLUX] file size = {raw_bytes}", flush=True)
                    print("[FLUX] real FLUX output = true", flush=True)
                except Exception as raw_exc:
                    # Black-image / hard failures must not be swallowed.
                    if "completely black image" in str(raw_exc).lower():
                        raise
                    self.logger.warning("Raw output save failed: %s", raw_exc)

            self._log_generation_stage("COMPLETE")
            total_time = round(time.perf_counter() - t_start, 3)

            # If explicit encode was used, diffusion_s includes only denoise+vae inside pipeline;
            # pipeline still may decode VAE inside __call__, so diffusion_s ≈ denoise+vae.
            # Prefer step sum for denoise estimate when available.
            step_sum = round(sum(per_step_durations), 3) if per_step_durations else 0.0
            if step_sum > 0:
                infer_time = step_sum
                # Remainder of pipeline call after steps ≈ VAE + overhead
                vae_est = max(0.0, round(diffusion_s - step_sum, 3))
            else:
                infer_time = diffusion_s
                vae_est = decode_s

            vram_snap = self._vram_snapshot()
            vram_after = vram_snap.get("allocated_mb", 0.0)
            peak_allocated = vram_snap.get("max_allocated_mb", 0.0)
            peak_reserved = vram_snap.get("max_reserved_mb", 0.0)
            # Backward-compatible alias: peak CUDA *allocated* (not dedicated physical VRAM).
            peak_vram = peak_allocated
            ram_after = self._cpu_ram_mb()
            physical_vram_mb = 0.0
            try:
                if torch is not None and torch.cuda.is_available():
                    physical_vram_mb = round(
                        torch.cuda.get_device_properties(0).total_memory / (1024**2), 1
                    )
            except Exception:
                physical_vram_mb = 0.0

            if profile:
                step_lines = "\n".join(
                    f"Step {i + 1}: {d:.3f} sec" for i, d in enumerate(per_step_durations)
                )
                self.logger.info(
                    "\n[FLUX PROFILE]\n"
                    "Resolution: %sx%s\n"
                    "Steps: %s\n"
                    "Guidance: %s\n"
                    "max_sequence_length: %s\n"
                    "\n"
                    "Model loading: %.3f sec (reused=%s)\n"
                    "Image preprocess: %.3f sec\n"
                    "Prompt encoding: %.3f sec (cached=%s)\n"
                    "Diffusion (pipeline call): %.3f sec\n"
                    "%s\n"
                    "VAE/decode estimate: %.3f sec\n"
                    "\n"
                    "TOTAL: %.3f sec\n"
                    "\n"
                    "CUDA allocated before: %.1f MB\n"
                    "Peak CUDA allocated: %.1f MB\n"
                    "Peak CUDA reserved: %.1f MB\n"
                    "CUDA allocated after: %.1f MB\n"
                    "Physical GPU VRAM: %.1f MB (allocator peaks may exceed this under "
                    "CPU offload / Windows shared memory)\n"
                    "CPU RAM: %.1f → %.1f MB\n"
                    "Offload: %s\n"
                    "Quantization: %s\n"
                    "Attention: %s\n"
                    "torch.compile: %s\n"
                    "=======================================",
                    width,
                    height,
                    num_inference_steps,
                    guidance_scale,
                    max_seq,
                    model_load_time,
                    model_was_reused,
                    resize_time,
                    encode_s,
                    encode_cached,
                    diffusion_s,
                    step_lines or "(no per-step samples)",
                    vae_est,
                    total_time,
                    vram_before,
                    peak_allocated,
                    peak_reserved,
                    vram_after,
                    physical_vram_mb,
                    ram_before,
                    ram_after,
                    runtime.get(
                        "offload_strategy",
                        getattr(self.model_loader, "_offload_strategy", "unknown"),
                    ),
                    "nf4"
                    if runtime.get("bnb_4bit", getattr(self.model_loader, "_used_bnb_4bit", False))
                    else "full",
                    runtime.get("attention_backend", "unknown"),
                    runtime.get("torch_compile", False),
                )

            self.last_execution_stats = {
                "was_fallback_used": False,
                "was_real_flux_used": True,
                "model_kind": "flux-kontext",
                "model_reused": model_was_reused,
                "has_image": True,
                "vram_before_mb": round(vram_before, 2),
                "vram_after_mb": round(vram_after, 2),
                "peak_vram_mb": round(peak_vram, 2),
                "peak_cuda_allocated_mb": round(peak_allocated, 2),
                "peak_cuda_reserved_mb": round(peak_reserved, 2),
                "physical_gpu_vram_mb": physical_vram_mb,
                "vram_metric_note": (
                    "peak_vram_mb is torch.cuda.max_memory_allocated (CUDA allocator), "
                    "not dedicated physical VRAM occupancy under model_cpu_offload."
                ),
                "cpu_ram_before_mb": ram_before,
                "cpu_ram_after_mb": ram_after,
                "generation_time_s": total_time,
                "model_load_time_s": model_load_time,
                "image_preprocess_time_s": resize_time,
                "prompt_encoding_time_s": encode_s,
                "prompt_encode_cached": encode_cached,
                "inference_time_s": infer_time,
                "vae_decode_time_s": vae_est,
                "diffusion_pipeline_s": diffusion_s,
                "per_step_durations_s": per_step_durations,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
                "max_sequence_length": max_seq,
                "output_size": list(image.size) if hasattr(image, "size") else [width, height],
                "offload_strategy": runtime.get(
                    "offload_strategy",
                    getattr(self.model_loader, "_offload_strategy", None),
                ),
                "attention_backend": runtime.get("attention_backend"),
                "attention_diag": runtime.get("attention_diag")
                or getattr(self.model_loader, "_attention_diag", None),
                "torch_compile": runtime.get("torch_compile"),
                "bnb_4bit": runtime.get("bnb_4bit"),
                "runtime": runtime,
            }

            self._park_pipeline(pipeline)
            return image
        except Exception as exc:
            self.logger.exception("FLUX Kontext inference error: %s", exc)
            # Always reclaim GPU after failure so the next job is not stuck at ~6GB used.
            try:
                if pipeline is not None:
                    self._park_pipeline(pipeline)
                elif hasattr(self.model_loader, "park_on_cpu"):
                    self.model_loader.park_on_cpu()
            except Exception as park_exc:
                self.logger.warning("post-error park_on_cpu failed: %s", park_exc)
            if self._is_cuda_oom(exc):
                phys = self._physical_vram_mib()
                raise RuntimeError(
                    f"CUDA out of memory during FLUX generation "
                    f"(allocated≈{self._vram_snapshot().get('allocated_mb')} MB; "
                    f"physical VRAM={phys} MiB): {exc}"
                ) from exc
            if self._is_attention_kernel_error(exc):
                raise RuntimeError(
                    "ATTENTION_KERNEL_UNAVAILABLE: FLUX transformer SDPA has no usable "
                    "kernel (often BF16 Q/K/V on pre-Ampere GPUs with MATH disabled). "
                    "Expected path: transformer-only FP16 Q/K/V cast for efficient SDPA. "
                    f"Original: {exc}"
                ) from exc
            if not self.allow_fallback:
                raise RuntimeError(f"Real FLUX Kontext inference failed: {exc}") from exc
            self.last_execution_stats = {
                "was_fallback_used": True,
                "was_real_flux_used": False,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_before,
                "peak_vram_mb": vram_before,
                "generation_time_s": round(time.perf_counter() - t_start, 3),
            }
            return self._generate_synthetic_preview(width, height, prompt)

    def _generate_synthetic_preview(self, width: int, height: int, prompt: str) -> Image.Image:
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        margin_x, margin_y = int(width * 0.25), int(height * 0.2)
        garment_box = [margin_x, margin_y, width - margin_x, height - margin_y]
        draw.rectangle(garment_box, fill=(240, 240, 245), outline=(200, 200, 210), width=3)
        draw.text((margin_x + 20, margin_y + 40), "FabricVision-AI", fill=(80, 80, 90))
        draw.text((margin_x + 20, margin_y + 80), "FLUX Kontext Output", fill=(100, 100, 110))
        return img
