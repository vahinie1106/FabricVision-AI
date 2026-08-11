"""Load and manage FLUX.1-Kontext diffusion weights with 6GB VRAM safety."""

from __future__ import annotations

import gc
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Optional

from src.common.models.device_manager import DeviceManager

try:
    import torch
except ImportError:
    torch = None

ProgressCallback = Optional[Callable[[str, int], None]]


class FLUXModelLoader:
    """
    Load FLUX.1-Kontext for fabric→garment image-conditioned generation.

    Resolution order:
    1. Complete local dir at `models/flux-kontext`
    2. Hybrid: Kontext transformer under `models/flux-kontext/transformer`
       + shared CLIP/T5/VAE/tokenizers from legacy `models/flux` (schnell)
    3. Hugging Face download IDs (NF4 package preferred for 6GB GPUs)

    The pipeline is resident for the process lifetime. Callers must reuse
    `load()` - it returns the cached pipeline without reloading weights.
    """

    DEFAULT_HF_ID = "black-forest-labs/FLUX.1-Kontext-dev"
    DEFAULT_NF4_HF_ID = "eramth/flux-kontext-4bit"
    LEGACY_SCHNELL_PATH = Path("models/flux")

    def __init__(
        self,
        model_path: str | Path = "models/flux-kontext",
        device: str = "auto",
        precision: str = "bfloat16",
        hf_model_id: Optional[str] = None,
        allow_fallback: bool = True,
        enable_torch_compile: Optional[bool] = None,
        attention_backend: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device_setting = device
        self.precision = precision
        self.hf_model_id = hf_model_id or os.environ.get(
            "FLUX_KONTEXT_MODEL_ID", self.DEFAULT_NF4_HF_ID
        )
        self.allow_fallback = allow_fallback
        self.logger = logging.getLogger("fabricvision.garment_generation.model_loader")
        self.device_manager = DeviceManager()
        self._pipeline = None
        self._used_bnb_4bit = False
        self._offload_strategy = "none"
        self._model_kind = "flux-kontext"
        self._attention_backend = "default"
        self._attention_diag: dict = {}
        self._torch_compile_enabled = False
        self._init_time_s = 0.0
        self._download_time_s = 0.0
        self._pipeline_assemble_time_s = 0.0
        self._offload_config_time_s = 0.0
        self._cache_status = "unknown"  # hit | miss | hybrid | hub_direct
        self._load_count = 0
        self._reuse_count = 0
        self._progress_callback: ProgressCallback = None
        # Quantization profile: nf4 (default / low-VRAM) | full (bf16/fp16, quality path)
        quant_env = os.environ.get("FLUX_QUANTIZATION", "").strip().lower()
        disable_nf4 = os.environ.get("FLUX_DISABLE_NF4", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if quant_env in ("full", "bf16", "fp16", "none", "off"):
            self._want_nf4 = False
        elif quant_env in ("nf4", "4bit", "bnb"):
            self._want_nf4 = True
        elif disable_nf4:
            self._want_nf4 = False
        else:
            self._want_nf4 = True  # preserve RTX 3050 default

        # Env overrides win so operators can tune without code edits.
        env_compile = os.environ.get("FLUX_ENABLE_TORCH_COMPILE", "").strip().lower()
        if enable_torch_compile is not None:
            self._want_compile = bool(enable_torch_compile)
        elif env_compile in ("1", "true", "yes", "on"):
            self._want_compile = True
        else:
            self._want_compile = False  # default false until validated

        self._want_attention = (
            attention_backend
            or os.environ.get("FLUX_ATTENTION_BACKEND", "auto")
        ).strip().lower()

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        self._progress_callback = callback

    def _progress(self, step: str, pct: int) -> None:
        msg = f"[FLUX] {step}"
        self.logger.info(msg)
        print(msg, flush=True)
        if self._progress_callback is not None:
            try:
                self._progress_callback(step, int(pct))
            except Exception:
                pass

    def _mark(self, label: str, t_seg: float, *, end: bool = False, extra: str = "") -> float:
        now = time.perf_counter()
        if end:
            line = f"[FLUX] {label} END t={now:.2f} elapsed={now - t_seg:.2f}s"
        else:
            line = f"[FLUX] {label} START t={now:.2f}"
        if extra:
            line = f"{line} {extra}"
        self.logger.info(line)
        print(line, flush=True)
        return now

    def _is_complete_local_dir(self, path: Path) -> bool:
        if not path.exists() or not (path / "model_index.json").exists():
            return False
        weight_files = (
            list(path.rglob("*.safetensors"))
            + list(path.rglob("*.sft"))
            + list(path.rglob("*.bin"))
        )
        return len(weight_files) >= 2

    def _has_transformer_weights(self, path: Path) -> bool:
        trans = path / "transformer"
        if not trans.exists():
            return False
        return bool(
            list(trans.glob("*.safetensors"))
            + list(trans.glob("*.bin"))
            + list(trans.glob("*.sft"))
        )

    def _prepare_hybrid_kontext_dir(self) -> Optional[Path]:
        """
        Build a Diffusers-ready Kontext dir by combining:
        - Kontext transformer (required)
        - Shared encoders/VAE/tokenizers from installed schnell weights
        """
        if not self._has_transformer_weights(self.model_path):
            return None
        if not self._is_complete_local_dir(self.LEGACY_SCHNELL_PATH):
            self.logger.warning(
                "Kontext transformer found but shared schnell components missing at %s",
                self.LEGACY_SCHNELL_PATH,
            )
            return None

        self.logger.info(
            "Preparing hybrid Kontext package: transformer from %s, "
            "shared components from %s",
            self.model_path,
            self.LEGACY_SCHNELL_PATH,
        )

        index_src = self.model_path / "model_index.json"
        if not index_src.exists():
            schnell_index = self.LEGACY_SCHNELL_PATH / "model_index.json"
            if schnell_index.exists():
                import json

                data = json.loads(schnell_index.read_text(encoding="utf-8"))
                data["_class_name"] = "FluxKontextPipeline"
                index_src.write_text(json.dumps(data, indent=2), encoding="utf-8")

        shared = [
            "scheduler",
            "text_encoder",
            "text_encoder_2",
            "tokenizer",
            "tokenizer_2",
            "vae",
        ]
        for name in shared:
            dst = self.model_path / name
            src = self.LEGACY_SCHNELL_PATH / name
            if dst.exists() or not src.exists():
                continue
            self.logger.info("Linking shared component %s", name)
            try:
                os.symlink(src.resolve(), dst, target_is_directory=True)
            except OSError:
                shutil.copytree(src, dst)

        if self._is_complete_local_dir(self.model_path):
            return self.model_path
        return None

    def _local_t5_ready(self, path: Path) -> bool:
        te2 = path / "text_encoder_2"
        if not te2.exists():
            return False
        return bool(list(te2.glob("*.safetensors")) + list(te2.glob("*.bin")))

    def _hf_token_present(self) -> bool:
        return bool(
            (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
        )

    def _gpu_vram_mb(self) -> float:
        if torch is None:
            return 0.0
        try:
            if torch.cuda.is_available():
                return float(torch.cuda.get_device_properties(0).total_memory) / (1024**2)
        except Exception:
            return 0.0
        return 0.0

    def _ensure_hub_package(self, repo_id: str) -> str:
        """
        Ensure a complete Diffusers Kontext package exists under model_path.

        On Kaggle, ``models/`` is gitignored so a fresh clone has no weights.
        Download the configured HF package into ``model_path`` when incomplete.
        """
        import threading

        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        if self._is_complete_local_dir(self.model_path) and self._local_t5_ready(self.model_path):
            self._cache_status = "hit"
            self.logger.info("[FLUX] CACHE HIT local package ready at %s", self.model_path.resolve())
            print(f"[FLUX] CACHE HIT path={self.model_path.resolve()}", flush=True)
            return str(self.model_path)

        download_root = self.model_path
        if self._has_transformer_weights(self.model_path) and not self._local_t5_ready(
            self.model_path
        ):
            if not self._is_complete_local_dir(self.LEGACY_SCHNELL_PATH):
                download_root = self.model_path.parent / f"{self.model_path.name}-hub"
                self.logger.warning(
                    "Incomplete local Kontext tree at %s (transformer without T5/VAE). "
                    "Downloading hub package to %s",
                    self.model_path,
                    download_root,
                )

        if self._is_complete_local_dir(download_root) and self._local_t5_ready(download_root):
            self._cache_status = "hit"
            self.logger.info("[FLUX] CACHE HIT hub package at %s", download_root.resolve())
            print(f"[FLUX] CACHE HIT path={download_root.resolve()}", flush=True)
            return str(download_root)

        self._cache_status = "miss"
        self._progress(
            f"Downloading FLUX weights (CACHE MISS: {repo_id})",
            9,
        )
        self.logger.info(
            "[FLUX] CACHE MISS / DOWNLOAD starting repo=%s -> %s (HF_TOKEN present: %s)",
            repo_id,
            download_root,
            self._hf_token_present(),
        )
        print(
            f"[FLUX] CACHE MISS / DOWNLOAD repo={repo_id} dest={download_root}",
            flush=True,
        )
        download_root.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "MODEL_DEPENDENCY_ERROR: huggingface_hub is required to download FLUX weights"
            ) from exc

        token = (
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
        )
        t_dl = self._mark("weight download", time.perf_counter())
        stop_hb = threading.Event()

        def _heartbeat() -> None:
            started = time.perf_counter()
            while not stop_hb.wait(20.0):
                elapsed = int(time.perf_counter() - started)
                # Keep UI alive during multi-GB downloads (stay in loading band 9-11%).
                pct = min(11, 9 + elapsed // 120)
                self._progress(
                    f"Downloading FLUX weights ({elapsed}s elapsed, CACHE MISS)",
                    pct,
                )

        hb = threading.Thread(target=_heartbeat, name="flux-download-hb", daemon=True)
        hb.start()
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(download_root),
                resume_download=True,
                max_workers=2,
                token=token,
            )
        except Exception as exc:
            err = str(exc).lower()
            if any(k in err for k in ("401", "403", "unauthorized", "gated", "access")):
                raise RuntimeError(
                    "MODEL_AUTH_FAILED: Hugging Face authentication failed while downloading "
                    f"{repo_id}. Set Kaggle secret HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) if "
                    "the repo requires access. Do not commit tokens."
                ) from exc
            if any(k in err for k in ("404", "not found", "repository not found")):
                raise RuntimeError(
                    f"MODEL_NOT_FOUND: Hugging Face repo '{repo_id}' was not found."
                ) from exc
            raise RuntimeError(
                f"MODEL_DOWNLOAD_FAILED: Could not download '{repo_id}' into "
                f"'{download_root}': {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            stop_hb.set()
            hb.join(timeout=2.0)

        self._download_time_s = round(time.perf_counter() - t_dl, 2)
        self._mark("weight download", t_dl, end=True)

        if not (
            self._is_complete_local_dir(download_root) and self._local_t5_ready(download_root)
        ):
            raise RuntimeError(
                f"MODEL_DOWNLOAD_FAILED: Download of '{repo_id}' completed but package at "
                f"'{download_root}' is still incomplete (missing T5/VAE/weights)."
            )
        self.logger.info("FLUX hub package ready at %s", download_root.resolve())
        self._progress("FLUX weights downloaded - initializing pipeline", 12)
        return str(download_root)

    def _resolve_model_source(self) -> tuple[str, Optional[str]]:
        """
        Returns (pipeline_root, transformer_root_or_None).

        Prefer a complete Kontext directory. Otherwise load shared components from
        schnell (`models/flux`) and the Kontext transformer from `models/flux-kontext`.
        On Kaggle (no gitignored weights), download the configured HF package.
        """
        self.logger.info("FLUX model ID: %s", self.hf_model_id)
        self.logger.info(
            "FLUX resolve: model_path=%s exists=%s hf_token_present=%s cuda=%s gpu_vram_mb=%.0f",
            self.model_path,
            self.model_path.exists(),
            self._hf_token_present(),
            bool(torch is not None and torch.cuda.is_available()),
            self._gpu_vram_mb(),
        )

        if self._is_complete_local_dir(self.model_path) and self._has_transformer_weights(
            self.model_path
        ):
            if self._local_t5_ready(self.model_path):
                self._cache_status = "hit"
                self.logger.info(
                    "[FLUX] CACHE HIT complete local Kontext weights at %s", self.model_path
                )
                print(f"[FLUX] CACHE HIT path={self.model_path}", flush=True)
                return str(self.model_path), None

        if self._has_transformer_weights(self.model_path) and self._is_complete_local_dir(
            self.LEGACY_SCHNELL_PATH
        ):
            self._cache_status = "hybrid"
            self.logger.info(
                "[FLUX] CACHE HIT hybrid Kontext load: transformer=%s shared_from=%s",
                self.model_path,
                self.LEGACY_SCHNELL_PATH,
            )
            return str(self.LEGACY_SCHNELL_PATH), str(self.model_path)

        if not self.hf_model_id:
            raise RuntimeError(
                "MODEL_NOT_FOUND: No FLUX.1-Kontext weights found. "
                "Run: python scripts/download_flux_kontext.py"
            )

        skip_prefetch = os.environ.get("FLUX_SKIP_HUB_PREFETCH", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if skip_prefetch:
            self._cache_status = "hub_direct"
            self.logger.info("Using Hugging Face Kontext source directly: %s", self.hf_model_id)
            return self.hf_model_id, None

        local_pkg = self._ensure_hub_package(self.hf_model_id)
        return local_pkg, None

    def _configure_attention(self, pipeline: Any) -> str:
        """
        Prefer memory-efficient SDPA and disable MATH when the installed torch
        APIs allow it (required for Kontext 1024 on T4-class 16GB GPUs).

        ``ATTENTION_BACKEND=sdpa`` alone is insufficient: PyTorch may still pick
        the MATH kernel and allocate ~6 GiB for the joint Kontext sequence.
        xFormers / packaged FlashAttention are not installed here (explicitly
        deferred); this path uses only PyTorch SDPA + optional Diffusers
        ``transformer.set_attention_backend`` when present.
        """
        from src.features.custom_generator.inference.flux_attention import (
            configure_memory_efficient_attention,
        )

        wanted = self._want_attention
        if wanted == "default":
            self._attention_backend = "default"
            self._attention_diag = {
                "attention_backend_requested": "default",
                "attention_config_ok": True,
            }
            return self._attention_backend

        # auto / sdpa / flash / memory_efficient → memory-efficient SDPA path.
        # xformers is intentionally not auto-installed; only use if already present
        # and explicitly requested (not part of the T4 1024 first fix).
        if wanted == "xformers":
            try:
                import xformers  # noqa: F401

                if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                    pipeline.enable_xformers_memory_efficient_attention()
                    self._attention_backend = "xformers"
                    self._attention_diag = {
                        "attention_backend_requested": "xformers",
                        "attention_backend_effective": "xformers",
                        "attention_config_ok": True,
                    }
                    self.logger.info("[FLUX] Attention backend: xformers")
                    return self._attention_backend
            except Exception as exc:
                self.logger.warning(
                    "[FLUX] xFormers requested but unavailable (%s); "
                    "falling back to memory-efficient SDPA",
                    exc,
                )

        if wanted in ("auto", "sdpa", "flash", "memory_efficient", "mem_efficient"):
            diag = configure_memory_efficient_attention(pipeline)
            self._attention_diag = diag
            if diag.get("attention_config_ok"):
                self._attention_backend = "memory_efficient_sdpa"
                self.logger.info(
                    "[FLUX] Attention backend: memory_efficient_sdpa (%s)",
                    diag.get("attention_backend_effective"),
                )
                return self._attention_backend
            err = diag.get("error") or "NO_SUPPORTED_EFFICIENT_ATTENTION_BACKEND"
            self.logger.error("[FLUX] Memory-efficient attention unavailable: %s", err)
            # Do not silently fall back to MATH-capable generic "sdpa".
            self._attention_backend = "unavailable"
            return self._attention_backend

        self._attention_backend = "default"
        self._attention_diag = {
            "attention_backend_requested": wanted,
            "attention_config_ok": False,
            "error": f"unsupported_want={wanted}",
        }
        self.logger.info("[FLUX] Attention backend: default")
        return self._attention_backend

    def _maybe_torch_compile(self, pipeline: Any) -> bool:
        """
        Optionally compile the transformer only.

        Why transformer-only: bitsandbytes NF4 + model_cpu_offload conflict with
        compiling the full pipeline. Default remains disabled until measured wins.
        """
        if not self._want_compile:
            self._torch_compile_enabled = False
            return False
        if torch is None or not hasattr(torch, "compile"):
            self.logger.warning("[FLUX] torch.compile requested but unavailable")
            self._torch_compile_enabled = False
            return False
        if self._used_bnb_4bit:
            # Compiling Linear4bit modules is commonly unsupported / unstable.
            self.logger.warning(
                "[FLUX] torch.compile skipped: incompatible with bitsandbytes NF4 "
                "transformer on this path"
            )
            self._torch_compile_enabled = False
            return False
        if self._offload_strategy == "model_cpu_offload":
            self.logger.warning(
                "[FLUX] torch.compile skipped: conflicts with model_cpu_offload "
                "(module device movement)"
            )
            self._torch_compile_enabled = False
            return False

        try:
            transformer = getattr(pipeline, "transformer", None)
            if transformer is None:
                self.logger.warning("[FLUX] torch.compile skipped: no transformer")
                return False
            pipeline.transformer = torch.compile(transformer, mode="reduce-overhead")
            self._torch_compile_enabled = True
            self.logger.info("[FLUX] torch.compile enabled on transformer")
            return True
        except Exception as exc:
            self.logger.warning(
                "[FLUX] torch.compile failed (%s); continuing without compile", exc
            )
            self._torch_compile_enabled = False
            return False

    def load(self, progress_callback: ProgressCallback = None) -> Any | None:
        """Load FluxKontextPipeline once; subsequent calls reuse the resident pipeline."""
        if progress_callback is not None:
            self._progress_callback = progress_callback
        if self._pipeline is not None:
            self._reuse_count += 1
            self.logger.info(
                "[FLUX] Reusing loaded Kontext pipeline (reuse_count=%s)",
                self._reuse_count,
            )
            print(
                f"[FLUX] REUSE pipeline resident=True reuse_count={self._reuse_count}",
                flush=True,
            )
            self._progress("Reusing loaded FLUX.1-Kontext pipeline", 15)
            return self._pipeline

        if os.environ.get("PYTEST_CURRENT_TEST") and self.allow_fallback:
            self.logger.info("Pytest environment detected; skipping Kontext weight load.")
            return None

        t0 = self._mark("pipeline initialization", time.perf_counter())
        self._progress("Loading model - resolving FLUX weights", 8)
        target_device = self.device_manager.resolve_device(self.device_setting)
        vram_before = self._gpu_vram_mb()
        print(
            f"[FLUX] START pipeline initialization device={target_device} "
            f"gpu_vram_mb={vram_before:.0f} cuda={bool(torch and torch.cuda.is_available())}",
            flush=True,
        )

        # NF4 Diffusers packages require bitsandbytes + valid package metadata.
        if bool(getattr(self, "_want_nf4", True)):
            try:
                from src.common.utils.ensure_bitsandbytes import ensure_bitsandbytes

                t_bnb = self._mark("bitsandbytes check", time.perf_counter())
                bnb_ver = ensure_bitsandbytes(auto_install=True)
                self._mark("bitsandbytes check", t_bnb, end=True, extra=f"ver={bnb_ver}")
            except Exception as exc:
                msg = str(exc) or f"{type(exc).__name__}"
                self.logger.error("%s", msg)
                if not self.allow_fallback:
                    raise RuntimeError(msg) from exc
                return None

        try:
            from diffusers import FluxKontextPipeline, FluxTransformer2DModel
        except ImportError as exc:
            msg = (
                "MODEL_DEPENDENCY_ERROR: Diffusers FluxKontextPipeline unavailable: "
                f"{exc}. Install a recent diffusers build that exports FluxKontextPipeline."
            )
            self.logger.error(msg)
            if not self.allow_fallback:
                raise RuntimeError(msg) from exc
            return None

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        try:
            self._progress("Resolving FLUX model source", 8)
            pipeline_root, transformer_root = self._resolve_model_source()
        except Exception as exc:
            self.logger.error("%s", exc)
            if not self.allow_fallback:
                raise
            return None

        dtype = None
        if torch is not None:
            dtype = (
                torch.bfloat16
                if (self.precision == "bfloat16" and torch.cuda.is_available())
                else torch.float16
            )

        use_low_cpu_mem = False if os.environ.get("PYTEST_CURRENT_TEST") else True
        pipeline = None
        used_bnb_4bit = False
        model_source = pipeline_root

        try:
            transformer = None
            want_nf4 = bool(getattr(self, "_want_nf4", True))
            self.logger.info(
                "[FLUX] Quantization profile: %s dtype=%s cache=%s",
                "nf4" if want_nf4 else "full_precision",
                dtype,
                self._cache_status,
            )
            if transformer_root is not None:
                self._progress("Loading FLUX transformer", 13)
                t_tr = self._mark("transformer load", time.perf_counter())
                self.logger.info(
                    "Loading Kontext transformer from %s (shared root %s)",
                    transformer_root,
                    pipeline_root,
                )
                if want_nf4:
                    try:
                        from transformers import BitsAndBytesConfig

                        quant_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=dtype or torch.bfloat16,
                        )
                        transformer = FluxTransformer2DModel.from_pretrained(
                            transformer_root,
                            subfolder="transformer",
                            quantization_config=quant_config,
                            torch_dtype=dtype or torch.bfloat16,
                            low_cpu_mem_usage=use_low_cpu_mem,
                        )
                        used_bnb_4bit = True
                    except Exception as q_exc:
                        self.logger.info(
                            "NF4 transformer load deferred (%s); loading as-is...", q_exc
                        )
                        transformer = FluxTransformer2DModel.from_pretrained(
                            transformer_root,
                            subfolder="transformer",
                            torch_dtype=dtype,
                            low_cpu_mem_usage=use_low_cpu_mem,
                        )
                        used_bnb_4bit = bool(
                            any(
                                "Linear4bit" in type(m).__name__
                                for m in list(transformer.modules())[:50]
                            )
                        )
                else:
                    transformer = FluxTransformer2DModel.from_pretrained(
                        transformer_root,
                        subfolder="transformer",
                        torch_dtype=dtype,
                        low_cpu_mem_usage=use_low_cpu_mem,
                    )
                    used_bnb_4bit = False
                self._mark("transformer load", t_tr, end=True)

            self._progress("Loading FluxKontextPipeline (T5/CLIP/VAE)", 14)
            t_pipe = self._mark("pipeline assembly", time.perf_counter())
            self.logger.info("Loading FluxKontextPipeline from %s ...", pipeline_root)
            kwargs = {
                "torch_dtype": dtype,
                "low_cpu_mem_usage": use_low_cpu_mem,
            }
            if transformer is not None:
                kwargs["transformer"] = transformer

            try:
                pipeline = FluxKontextPipeline.from_pretrained(pipeline_root, **kwargs)
                if transformer is None:
                    trans = getattr(pipeline, "transformer", None)
                    used_bnb_4bit = bool(
                        trans is not None
                        and any(
                            "Linear4bit" in type(m).__name__
                            for m in list(trans.modules())[:50]
                        )
                    )
                    if not want_nf4 and used_bnb_4bit:
                        raise RuntimeError(
                            "Requested full-precision FLUX but pipeline transformer "
                            "still contains Linear4bit modules (likely an NF4 checkpoint). "
                            "Point FLUX model path at a non-quantized Kontext root, or "
                            "set FLUX_QUANTIZATION=nf4."
                        )
            except Exception as direct_exc:
                if not want_nf4:
                    raise
                self.logger.info(
                    "Direct Kontext load deferred (%s); trying explicit NF4 transformer...",
                    direct_exc,
                )
                pipeline = None

            if (
                pipeline is None
                and want_nf4
                and torch is not None
                and torch.cuda.is_available()
            ):
                from transformers import BitsAndBytesConfig

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype or torch.bfloat16,
                )
                transformer = FluxTransformer2DModel.from_pretrained(
                    pipeline_root if transformer_root is None else transformer_root,
                    subfolder="transformer",
                    quantization_config=quant_config,
                    torch_dtype=dtype or torch.bfloat16,
                    low_cpu_mem_usage=use_low_cpu_mem,
                )
                pipeline = FluxKontextPipeline.from_pretrained(
                    pipeline_root,
                    transformer=transformer,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=use_low_cpu_mem,
                )
                used_bnb_4bit = True

            if pipeline is None:
                raise RuntimeError(
                    f"Unable to construct FluxKontextPipeline from {pipeline_root}"
                )
            self._pipeline_assemble_time_s = round(time.perf_counter() - t_pipe, 2)
            self._mark("pipeline assembly", t_pipe, end=True)

            self._used_bnb_4bit = used_bnb_4bit

            if target_device == "cuda":
                if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                    self.logger.info(
                        "[FLUX] PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
                    )

                # Offload policy:
                # - FLUX_MODEL_CPU_OFFLOAD=true → model_cpu_offload (6GB path)
                # - FLUX_MODEL_CPU_OFFLOAD=false → GPU-resident
                # - auto: offload on <14GB; GPU-resident on T4/16GB+ (NF4 or full)
                offload_env = os.environ.get("FLUX_MODEL_CPU_OFFLOAD", "").strip().lower()
                physical_mb = self._gpu_vram_mb()
                prefer_offload = True
                if offload_env in ("0", "false", "no", "off"):
                    prefer_offload = False
                elif offload_env in ("1", "true", "yes", "on"):
                    prefer_offload = True
                else:
                    # Auto: T4 15GB / 16GB+ should keep NF4 resident - CPU offload
                    # thrash was measured at multi-minute per diffusion step on 6GB.
                    prefer_offload = physical_mb < 14000

                self._progress("Configuring device / offload", 16)
                t_off = self._mark("offload configuration", time.perf_counter())
                # WHY model_cpu_offload (not sequential): sequential + bnb NF4 previously
                # raised "Cannot copy out of meta tensor; no data!" on this stack.
                if prefer_offload and hasattr(pipeline, "enable_model_cpu_offload"):
                    pipeline.enable_model_cpu_offload()
                    self._offload_strategy = "model_cpu_offload"
                    self.logger.info(
                        "Kontext model CPU offload enabled (%s, vram=%.0fMB)",
                        "bnb-safe" if used_bnb_4bit else "standard",
                        physical_mb,
                    )
                else:
                    if hasattr(pipeline, "to"):
                        pipeline.to(target_device)
                    self._offload_strategy = "gpu_resident" if not prefer_offload else "none"
                    self.logger.info(
                        "Kontext GPU-resident load (offload=%s, nf4=%s, vram=%.0fMB)",
                        self._offload_strategy,
                        used_bnb_4bit,
                        physical_mb,
                    )

                if hasattr(pipeline, "vae") and pipeline.vae is not None:
                    if hasattr(pipeline.vae, "enable_slicing"):
                        pipeline.vae.enable_slicing()
                    want_tile = os.environ.get("FLUX_VAE_TILING", "false").strip().lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                    # Auto-enable VAE tiling on high-res T4 path when not forced off.
                    if (
                        not want_tile
                        and physical_mb >= 14000
                        and os.environ.get("FLUX_VAE_TILING", "").strip() == ""
                    ):
                        want_tile = True
                    if want_tile and hasattr(pipeline.vae, "enable_tiling"):
                        pipeline.vae.enable_tiling()
                        self.logger.info("[FLUX] VAE tiling enabled (OOM mitigation)")
                    elif hasattr(pipeline.vae, "disable_tiling"):
                        try:
                            pipeline.vae.disable_tiling()
                        except Exception:
                            pass
                        self.logger.info(
                            "[FLUX] VAE tiling disabled (sharper decode on 6GB Standard/Production)"
                        )

                if torch is not None:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    try:
                        torch.set_float32_matmul_precision("high")
                    except Exception:
                        pass
                self._offload_config_time_s = round(time.perf_counter() - t_off, 2)
                self._mark("offload configuration", t_off, end=True)
            elif hasattr(pipeline, "to"):
                pipeline.to(target_device)
                self._offload_strategy = "none"

            self._configure_attention(pipeline)
            self._maybe_torch_compile(pipeline)

            self._pipeline = pipeline
            self._load_count += 1
            self._init_time_s = round(time.perf_counter() - t0, 2)
            alloc_after = 0.0
            if torch is not None and torch.cuda.is_available():
                alloc_after = round(torch.cuda.memory_allocated() / (1024**2), 1)
            self._mark("pipeline initialization", t0, end=True)
            print(
                f"[FLUX] FLUX READY elapsed={self._init_time_s}s "
                f"cache={self._cache_status} download_s={self._download_time_s} "
                f"assemble_s={self._pipeline_assemble_time_s} "
                f"offload={self._offload_strategy} bnb4bit={used_bnb_4bit} "
                f"vram_alloc_mb={alloc_after}",
                flush=True,
            )
            self._progress("FLUX READY", 18)
            self.logger.info(
                "[FLUX] Model initialization completed (%.2fs, bnb4bit=%s offload=%s "
                "attention=%s compile=%s cache=%s)",
                self._init_time_s,
                used_bnb_4bit,
                self._offload_strategy,
                self._attention_backend,
                self._torch_compile_enabled,
                self._cache_status,
            )
            return pipeline
        except Exception as exc:
            msg = (
                f"Failed to load FLUX.1-Kontext from '{model_source}': "
                f"{type(exc).__name__}: {exc}"
            )
            self.logger.exception(msg)
            if not self.allow_fallback:
                raise RuntimeError(msg) from exc
            return None

    @property
    def pipeline(self) -> Any | None:
        return self._pipeline

    def park_on_cpu(self) -> dict[str, float]:
        """
        Free GPU residency after / before FLUX jobs without breaking NF4 state.

        CRITICAL findings (Day-18 validation):
        - Never call ``transformer.to("cpu")`` on bitsandbytes NF4 modules
          (corrupts quant_state / device.index → ``getCurrentStream`` errors).
        - Do not call ``enable_model_cpu_offload()`` again here: ``maybe_free_model_hooks``
          already re-applies it. Double-enabling stacks hooks and breaks step 2+.
        - Prefer accelerate's ``maybe_free_model_hooks()`` as the sole offload reset.
        """
        before = 0.0
        reserved_before = 0.0
        if torch is not None and torch.cuda.is_available():
            before = round(torch.cuda.memory_allocated() / (1024**2), 1)
            reserved_before = round(torch.cuda.memory_reserved() / (1024**2), 1)

        pipe = self._pipeline
        if pipe is not None and hasattr(pipe, "maybe_free_model_hooks"):
            try:
                pipe.maybe_free_model_hooks()
            except Exception as exc:
                self.logger.debug("maybe_free_model_hooks skipped: %s", exc)

        # Optional: park non-quantized encoders/VAE only when NOT using model_cpu_offload,
        # so we do not fight accelerate hooks on the NF4 path.
        if pipe is not None and self._offload_strategy != "model_cpu_offload":
            for name in ("text_encoder", "text_encoder_2", "vae", "transformer"):
                if self._used_bnb_4bit and name == "transformer":
                    continue
                mod = getattr(pipe, name, None)
                if mod is None or not hasattr(mod, "to"):
                    continue
                try:
                    mod.to("cpu")
                except Exception as exc:
                    self.logger.debug("park %s → cpu skipped: %s", name, exc)

        gc.collect()
        after = before
        reserved_after = reserved_before
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
            after = round(torch.cuda.memory_allocated() / (1024**2), 1)
            reserved_after = round(torch.cuda.memory_reserved() / (1024**2), 1)

        self.logger.info(
            "[FLUX] park_on_cpu: allocated %.1f→%.1f MB, reserved %.1f→%.1f MB "
            "(physical GPU=%.0f MiB, bnb4bit=%s, offload=%s)",
            before,
            after,
            reserved_before,
            reserved_after,
            self._gpu_vram_mb(),
            self._used_bnb_4bit,
            self._offload_strategy,
        )
        return {
            "allocated_before_mb": before,
            "allocated_after_mb": after,
            "reserved_before_mb": reserved_before,
            "reserved_after_mb": reserved_after,
        }

    def get_runtime_info(self) -> dict[str, Any]:
        """Expose loader state for profiling / job metadata."""
        info: dict[str, Any] = {
            "model_kind": self._model_kind,
            "bnb_4bit": self._used_bnb_4bit,
            "want_nf4": bool(getattr(self, "_want_nf4", True)),
            "quantization_profile": (
                "nf4" if self._used_bnb_4bit else "full_precision"
            ),
            "offload_strategy": self._offload_strategy,
            "attention_backend": self._attention_backend,
            "attention_diag": dict(getattr(self, "_attention_diag", {}) or {}),
            "torch_compile": self._torch_compile_enabled,
            "init_time_s": self._init_time_s,
            "download_time_s": self._download_time_s,
            "pipeline_assemble_time_s": self._pipeline_assemble_time_s,
            "offload_config_time_s": self._offload_config_time_s,
            "cache_status": self._cache_status,
            "load_count": self._load_count,
            "reuse_count": self._reuse_count,
            "pipeline_resident": self._pipeline is not None,
            "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "hf_model_id": self.hf_model_id,
            "model_path": str(self.model_path),
            "hf_token_present": self._hf_token_present(),
            "gpu_vram_mb": round(self._gpu_vram_mb(), 1),
        }
        if torch is not None:
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 1)
                info["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024**2), 1)
        try:
            import diffusers

            info["diffusers_version"] = diffusers.__version__
        except Exception:
            pass
        try:
            import bitsandbytes as bnb

            info["bitsandbytes_version"] = getattr(bnb, "__version__", "unknown")
        except Exception:
            info["bitsandbytes_version"] = None
        return info
