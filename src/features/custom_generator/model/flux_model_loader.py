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
    # Reject git-LFS pointer stubs and truncated downloads. Real NF4 transformer
    # weights for eramth/flux-kontext-4bit are ~6.7 GiB; 50 MiB catches pointers.
    _MIN_TRANSFORMER_WEIGHT_BYTES = 50 * 1024 * 1024
    _MIN_COMPONENT_WEIGHT_BYTES = 1 * 1024 * 1024

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
        self._load_phase: str = "idle"
        self._load_phase_pct: int = 0
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
        self._load_phase = step
        self._load_phase_pct = int(pct)
        msg = f"[FLUX] {step} ({pct}%)"
        self.logger.info(msg)
        print(msg, flush=True)
        if self._progress_callback is not None:
            try:
                self._progress_callback(step, int(pct))
            except Exception as exc:
                self.logger.warning("progress_callback failed: %s", exc)

    def _mark(self, label: str, t_seg: float, *, end: bool = False, extra: str = "") -> float:
        now = time.perf_counter()
        pipe_exists = self._pipeline is not None
        cache = getattr(self, "_cache_status", None) or "unknown"
        base = (
            f"[FLUX TIMING] {label}_{'END' if end else 'START'} t={now:.2f}"
            + (f" duration={now - t_seg:.2f}s" if end else "")
            + f" process_id={os.getpid()}"
            + f" elapsed_seconds={now - t_seg:.2f}"
            + f" cache_state={cache}"
            + f" pipeline_exists={pipe_exists}"
        )
        if extra:
            base = f"{base} {extra}"
        self.logger.info(base)
        print(base, flush=True)
        return now

    REQUIRED_COMPONENT_DIRS = (
        "transformer",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "scheduler",
    )
    TRANSFORMER_WEIGHT_CANDIDATES = (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "diffusion_pytorch_model.sft",
    )

    @staticmethod
    def _is_lfs_pointer_file(path: Path) -> bool:
        """True when ``path`` is a git-LFS pointer, not real weight bytes."""
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size > 1024:
            return False
        try:
            with path.open("rb") as fh:
                head = fh.read(256)
        except OSError:
            return False
        return (
            b"git-lfs.github.com/spec" in head
            or head.startswith(b"version https://git-lfs")
        )

    def _component_weight_bytes(self, component_dir: Path) -> int:
        """Total non-pointer weight bytes under a Diffusers component directory."""
        if not component_dir.is_dir():
            return 0
        total = 0
        index = component_dir / "diffusion_pytorch_model.safetensors.index.json"
        if not index.exists():
            index = component_dir / "model.safetensors.index.json"
        if index.exists():
            try:
                import json

                data = json.loads(index.read_text(encoding="utf-8"))
                shards = {
                    component_dir / name
                    for name in (data.get("weight_map") or {}).values()
                }
                for shard in shards:
                    if shard.is_file() and not self._is_lfs_pointer_file(shard):
                        total += shard.stat().st_size
                if total > 0:
                    return total
            except Exception:
                pass
        for pattern in ("*.safetensors", "*.bin", "*.sft"):
            for weight in component_dir.glob(pattern):
                if not weight.is_file() or self._is_lfs_pointer_file(weight):
                    continue
                total += weight.stat().st_size
        return total

    def _find_transformer_weight_file(self, path: Path) -> Optional[Path]:
        """Return the canonical transformer weight file, or None if missing/invalid."""
        trans = path / "transformer"
        if not trans.is_dir():
            return None
        for name in self.TRANSFORMER_WEIGHT_CANDIDATES:
            candidate = trans / name
            if not candidate.is_file():
                continue
            if self._is_lfs_pointer_file(candidate):
                continue
            if candidate.stat().st_size < self._MIN_TRANSFORMER_WEIGHT_BYTES:
                continue
            return candidate
        # Sharded Diffusers layout.
        index = trans / "diffusion_pytorch_model.safetensors.index.json"
        if index.exists() and (
            self._component_weight_bytes(trans) >= self._MIN_TRANSFORMER_WEIGHT_BYTES
        ):
            return index
        return None

    def _has_transformer_weights(self, path: Path) -> bool:
        """True only when transformer/ has real Diffusers weight tensors."""
        return self._find_transformer_weight_file(path) is not None

    def _local_t5_ready(self, path: Path) -> bool:
        te2 = path / "text_encoder_2"
        return self._component_weight_bytes(te2) >= self._MIN_COMPONENT_WEIGHT_BYTES

    def _local_vae_ready(self, path: Path) -> bool:
        return (
            self._component_weight_bytes(path / "vae")
            >= self._MIN_COMPONENT_WEIGHT_BYTES
        )

    def _structure_missing(self, path: Path) -> list[str]:
        missing: list[str] = []
        if not (path / "model_index.json").exists():
            missing.append("model_index.json")
        for name in self.REQUIRED_COMPONENT_DIRS:
            if not (path / name).is_dir():
                missing.append(f"{name}/")
        return missing

    def _is_complete_local_dir(self, path: Path) -> bool:
        """
        Structural + weight completeness for a Diffusers FLUX Kontext tree.

        IMPORTANT: A directory with only ``model_index.json`` and a couple of
        small/partial weight files must NOT be treated as loadable.
        """
        if not path.exists() or not (path / "model_index.json").exists():
            return False
        if self._structure_missing(path):
            return False
        if not self._has_transformer_weights(path):
            return False
        if not self._local_vae_ready(path):
            return False
        if self._component_weight_bytes(path / "text_encoder") < self._MIN_COMPONENT_WEIGHT_BYTES:
            return False
        return True

    def _package_ready_for_pipeline(self, path: Path) -> bool:
        """Full package that FluxKontextPipeline.from_pretrained can load."""
        return (
            self._is_complete_local_dir(path)
            and self._has_transformer_weights(path)
            and self._local_t5_ready(path)
        )

    def _is_hub_repo_id(self, value: str) -> bool:
        text = (value or "").strip().replace("\\", "/")
        if not text or text.startswith("/") or text.startswith("."):
            return False
        # Local Windows/Unix paths are not hub IDs.
        if ":" in text[:3] or text.startswith("models/") or "/models/" in text:
            return False
        return "/" in text and not Path(text).exists()

    def preflight_validate_package(
        self, path: Optional[Path] = None, *, source: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Report whether a local Kontext package is loadable before from_pretrained.

        Prints ``[FLUX PREFLIGHT]`` lines for Kaggle logs.
        """
        root = Path(path) if path is not None else Path(self.model_path)
        src = source or self.hf_model_id or "(local)"
        missing_structure = self._structure_missing(root) if root.exists() else ["(directory missing)"]
        transformer_file = self._find_transformer_weight_file(root) if root.exists() else None
        transformer_path = root / "transformer" / "diffusion_pytorch_model.safetensors"
        transformer_lfs = (
            transformer_path.is_file() and self._is_lfs_pointer_file(transformer_path)
        )
        transformer_bytes = self._component_weight_bytes(root / "transformer") if root.exists() else 0
        t5_bytes = self._component_weight_bytes(root / "text_encoder_2") if root.exists() else 0
        vae_bytes = self._component_weight_bytes(root / "vae") if root.exists() else 0
        clip_bytes = self._component_weight_bytes(root / "text_encoder") if root.exists() else 0
        has_index = (root / "model_index.json").exists()
        transformer_ok = transformer_file is not None
        t5_ok = self._local_t5_ready(root) if root.exists() else False
        vae_ok = self._local_vae_ready(root) if root.exists() else False
        clip_ok = clip_bytes >= self._MIN_COMPONENT_WEIGHT_BYTES
        ready = self._package_ready_for_pipeline(root) if root.exists() else False
        missing_components: list[str] = []
        if missing_structure:
            missing_components.extend(missing_structure)
        if not transformer_ok:
            if transformer_lfs:
                missing_components.append(
                    "transformer/diffusion_pytorch_model.safetensors (git-LFS pointer only)"
                )
            elif transformer_path.exists() and transformer_path.stat().st_size < self._MIN_TRANSFORMER_WEIGHT_BYTES:
                missing_components.append(
                    f"transformer/diffusion_pytorch_model.safetensors "
                    f"(too small: {transformer_path.stat().st_size} bytes)"
                )
            else:
                missing_components.append(
                    "transformer/diffusion_pytorch_model.safetensors (or .bin/.sft)"
                )
        if not vae_ok:
            missing_components.append("vae weights")
        if not clip_ok:
            missing_components.append("text_encoder weights")
        if not t5_ok:
            missing_components.append("text_encoder_2 weights")

        report = {
            "path": str(root),
            "model_source": src,
            "model_index": has_index,
            "transformer_weights": transformer_ok,
            "transformer_file": str(transformer_file) if transformer_file else None,
            "transformer_bytes": transformer_bytes,
            "transformer_lfs_pointer": transformer_lfs,
            "text_encoder_2_weights": t5_ok,
            "text_encoder_2_bytes": t5_bytes,
            "vae_weights": vae_ok,
            "vae_bytes": vae_bytes,
            "text_encoder_weights": clip_ok,
            "text_encoder_bytes": clip_bytes,
            "missing": missing_components,
            "ready": ready,
        }

        print(
            f"[FLUX PREFLIGHT] model directory path={root} exists={root.exists()}",
            flush=True,
        )
        print(
            f"[FLUX PREFLIGHT] transformer weights ok={transformer_ok} "
            f"file={report['transformer_file']} bytes={transformer_bytes} "
            f"lfs_pointer={transformer_lfs}",
            flush=True,
        )
        print(
            f"[FLUX PREFLIGHT] required components "
            f"model_index={has_index} vae={vae_ok} clip={clip_ok} t5={t5_ok} "
            f"structure_missing={missing_structure or 'none'}",
            flush=True,
        )
        print(f"[FLUX PREFLIGHT] model source={src}", flush=True)
        if missing_components and not ready:
            print(
                f"[FLUX PREFLIGHT] missing={missing_components}",
                flush=True,
            )
        print(
            f"[FLUX PREFLIGHT] validation {'PASS' if ready else 'FAIL'}",
            flush=True,
        )
        self.logger.info("[FLUX PREFLIGHT] %s", report)
        return report

    def _assert_local_package_loadable(self, pipeline_root: str, *, source: str) -> None:
        """Hard gate: never call from_pretrained on an invalid local package."""
        if self._is_hub_repo_id(pipeline_root):
            print(
                f"[FLUX LOAD] hub repo id={pipeline_root} (Diffusers will fetch/cache)",
                flush=True,
            )
            return
        report = self.preflight_validate_package(Path(pipeline_root), source=source)
        if report["ready"]:
            print(
                f"[FLUX LOAD] package validated path={pipeline_root} "
                f"transformer={report.get('transformer_file')}",
                flush=True,
            )
            return
        raise RuntimeError(
            "[FLUX ERROR] refusing from_pretrained on incomplete package "
            f"path={pipeline_root} repo={source} missing={report.get('missing')}"
        )

    def _purge_incomplete_package(self, path: Path, *, reason: str) -> None:
        """Remove an invalid local tree so snapshot_download can recreate it cleanly."""
        if not path.exists():
            return
        print(
            f"[FLUX DOWNLOAD] purging incomplete package path={path} reason={reason}",
            flush=True,
        )
        self.logger.warning(
            "[FLUX DOWNLOAD] purging incomplete package at %s (%s)", path, reason
        )
        shutil.rmtree(path, ignore_errors=True)

    def _should_purge_before_download(self, path: Path, preflight: dict[str, Any]) -> bool:
        """
        Only wipe local trees that are corrupt (git-LFS pointers / junk).

        Never delete a tree that already has a real multi-GB transformer: that
        forces a multi-minute re-download after an interrupted snapshot. Prefer
        ``resume_download`` / hub-cache reuse instead.
        """
        if not path.exists():
            return False
        if preflight.get("transformer_lfs_pointer"):
            return True
        transformer_ok = bool(preflight.get("transformer_weights"))
        transformer_bytes = int(preflight.get("transformer_bytes") or 0)
        if transformer_ok and transformer_bytes >= self._MIN_TRANSFORMER_WEIGHT_BYTES:
            return False
        # Tiny stub weight files (not LFS) still poison Diffusers — wipe.
        tpath = path / "transformer" / "diffusion_pytorch_model.safetensors"
        if tpath.is_file() and tpath.stat().st_size < self._MIN_TRANSFORMER_WEIGHT_BYTES:
            return True
        return False

    def _dir_weight_bytes(self, path: Path) -> int:
        """Approximate on-disk weight bytes under a package (for download progress)."""
        if not path.is_dir():
            return 0
        total = 0
        for name in self.REQUIRED_COMPONENT_DIRS:
            total += self._component_weight_bytes(path / name)
        return total

    def _find_complete_hub_snapshot(self, repo_id: str) -> Optional[Path]:
        """
        Locate a complete Diffusers snapshot already present in the HF hub cache.

        This is the common Kaggle case after a prior download: ``models/`` is
        gitignored / empty on a fresh clone, but ``HF_HOME/hub`` still has blobs.
        """
        try:
            from src.common.utils.hf_cache_env import hub_repo_cache_dir

            repo_cache = hub_repo_cache_dir(repo_id)
        except Exception as exc:
            self.logger.warning("[FLUX CACHE] hub cache probe failed: %s", exc)
            return None

        candidates: list[Path] = []
        snaps = repo_cache / "snapshots"
        if snaps.is_dir():
            candidates.extend([p for p in snaps.iterdir() if p.is_dir()])
        # refs/main may point at the current revision
        ref = repo_cache / "refs" / "main"
        if ref.is_file():
            try:
                rev = ref.read_text(encoding="utf-8").strip()
                snap = snaps / rev
                if snap.is_dir():
                    candidates.insert(0, snap)
            except OSError:
                pass

        seen: set[str] = set()
        for snap in candidates:
            key = str(snap.resolve()) if snap.exists() else str(snap)
            if key in seen:
                continue
            seen.add(key)
            if self._package_ready_for_pipeline(snap):
                print(f"[FLUX CACHE] HIT hub_snapshot={snap}", flush=True)
                return snap
        return None

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
        Ensure a complete Diffusers Kontext package exists under model_path
        (or reuse a complete Hugging Face hub snapshot).

        On Kaggle, ``models/`` is gitignored so a fresh clone has no weights.
        Resolution order:
        1. Complete local ``models/flux-kontext`` → CACHE HIT
        2. Complete HF hub snapshot for ``repo_id`` → CACHE HIT (no network)
        3. Download into ``models/flux-kontext`` (resume; do not wipe good weights)
        """
        import threading

        from src.common.utils.hf_cache_env import ensure_huggingface_cache_env

        cache_env = ensure_huggingface_cache_env()
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        # Never force offline mode for recovery.
        if os.environ.get("HF_HUB_OFFLINE", "").strip() in ("1", "true", "True"):
            print(
                "[FLUX DOWNLOAD] warning: HF_HUB_OFFLINE is set; clearing for recovery",
                flush=True,
            )
            os.environ.pop("HF_HUB_OFFLINE", None)

        self._progress("Checking FLUX model cache", 5)
        print(
            f"[FLUX CACHE] CHECKING_CACHE model_path={self.model_path} "
            f"HF_HOME={cache_env.get('HF_HOME') or os.environ.get('HF_HOME')!r} "
            f"HUGGINGFACE_HUB_CACHE="
            f"{cache_env.get('HUGGINGFACE_HUB_CACHE') or os.environ.get('HUGGINGFACE_HUB_CACHE')!r}",
            flush=True,
        )

        download_root = Path(self.model_path)
        preflight = self.preflight_validate_package(download_root, source=repo_id)
        if preflight["ready"]:
            self._cache_status = "hit"
            print(f"[FLUX CACHE] HIT path={download_root.resolve()}", flush=True)
            self.logger.info(
                "[FLUX] CACHE HIT local package ready at %s", download_root.resolve()
            )
            self._progress("FLUX cache hit — loading pipeline", 12)
            return str(download_root)

        hub_snap = self._find_complete_hub_snapshot(repo_id)
        if hub_snap is not None:
            self._cache_status = "hit"
            self.logger.info(
                "[FLUX] CACHE HIT Hugging Face hub snapshot at %s", hub_snap
            )
            self._progress("FLUX cache hit (hub snapshot) — loading pipeline", 12)
            return str(hub_snap)

        print(
            f"[FLUX CACHE] MISS path={download_root} missing={preflight.get('missing')}",
            flush=True,
        )
        self._cache_status = "miss"
        self._progress(f"CACHE MISS — downloading FLUX weights ({repo_id})", 9)

        # Corrupt LFS/stub trees must be wiped. Valid partial downloads resume.
        if self._should_purge_before_download(download_root, preflight):
            self._purge_incomplete_package(
                download_root,
                reason=f"corrupt/incomplete stub missing={preflight.get('missing')}",
            )
        elif download_root.exists() and preflight.get("transformer_weights"):
            print(
                "[FLUX CACHE] RESUME keeping existing transformer weights; "
                "filling missing components without purge",
                flush=True,
            )

        print(f"[FLUX DOWNLOAD] repo={repo_id}", flush=True)
        print(f"[FLUX DOWNLOAD] local_dir={download_root.resolve()}", flush=True)
        self.logger.info(
            "[FLUX DOWNLOAD] starting repo=%s -> %s (HF_TOKEN present: %s)",
            repo_id,
            download_root,
            self._hf_token_present(),
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
        t_dl = self._mark("MODEL_DOWNLOAD", time.perf_counter())
        stop_hb = threading.Event()
        # Approx size of eramth/flux-kontext-4bit Diffusers tree (transformer alone ~6.7GiB).
        expected_bytes = float(
            os.environ.get("FLUX_EXPECTED_PACKAGE_BYTES", str(12 * 1024**3))
        )

        def _heartbeat() -> None:
            started = time.perf_counter()
            while not stop_hb.wait(15.0):
                elapsed = int(time.perf_counter() - started)
                on_disk = self._dir_weight_bytes(download_root)
                if expected_bytes > 0 and on_disk > 0:
                    frac = min(1.0, on_disk / expected_bytes)
                    # Download window: 9% → 48% based on measured bytes on disk.
                    pct = 9 + int(39 * frac)
                    gb = on_disk / (1024**3)
                    self._progress(
                        f"Downloading FLUX weights ({gb:.2f} GiB on disk, "
                        f"{elapsed}s elapsed)",
                        pct,
                    )
                else:
                    pct = min(20, 9 + elapsed // 60)
                    self._progress(
                        f"Downloading FLUX weights ({elapsed}s elapsed, CACHE MISS)",
                        pct,
                    )

        hb = threading.Thread(target=_heartbeat, name="flux-download-hb", daemon=True)
        hb.start()
        try:
            # Explicit local_dir keeps a flat Diffusers tree under models/flux-kontext.
            # Hub blobs still land under HUGGINGFACE_HUB_CACHE for later CACHE HITs.
            snapshot_kwargs = {
                "repo_id": repo_id,
                "local_dir": str(download_root),
                "max_workers": 2,
                "token": token,
            }
            # resume_download deprecated but still accepted on older hub; ignore TypeError.
            try:
                snapshot_download(**snapshot_kwargs, resume_download=True)  # type: ignore[call-arg]
            except TypeError:
                snapshot_download(**snapshot_kwargs)
        except Exception as exc:
            err = str(exc).lower()
            print(
                f"[FLUX ERROR] download failed repo={repo_id} "
                f"path={download_root} exc={type(exc).__name__}: {exc}",
                flush=True,
            )
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
        self._mark("MODEL_DOWNLOAD", t_dl, end=True)
        print("[FLUX DOWNLOAD] completed", flush=True)
        print("[FLUX PREFLIGHT] validating package", flush=True)

        post = self.preflight_validate_package(download_root, source=repo_id)
        if not post["ready"]:
            # Prefer a hub snapshot if local_dir materialization is incomplete but
            # the hub cache already has a full tree (common after interrupted copy).
            hub_snap = self._find_complete_hub_snapshot(repo_id)
            if hub_snap is not None:
                self._cache_status = "hit"
                print(
                    f"[FLUX CACHE] HIT hub_snapshot after download validate "
                    f"path={hub_snap}",
                    flush=True,
                )
                self._progress("FLUX weights ready (hub snapshot) — initializing pipeline", 12)
                return str(hub_snap)
            print(
                f"[FLUX ERROR] download finished but package still invalid "
                f"missing={post.get('missing')} transformer_bytes={post.get('transformer_bytes')}",
                flush=True,
            )
            raise RuntimeError(
                f"MODEL_DOWNLOAD_FAILED: Download of '{repo_id}' completed but package at "
                f"'{download_root}' is still incomplete "
                f"(missing={post.get('missing')}, "
                f"transformer_bytes={post.get('transformer_bytes')}, "
                f"lfs_pointer={post.get('transformer_lfs_pointer')})."
            )
        print("[FLUX PREFLIGHT] PASS", flush=True)
        self.logger.info("FLUX hub package ready at %s", download_root.resolve())
        self._progress("FLUX weights downloaded — initializing pipeline", 12)
        return str(download_root)

    def _resolve_model_source(self) -> tuple[str, Optional[str]]:
        """
        Returns (pipeline_root, transformer_root_or_None).

        Prefer a complete Kontext directory. Otherwise load shared components from
        schnell (`models/flux`) and the Kontext transformer from `models/flux-kontext`.
        On Kaggle (no gitignored weights), download the configured HF package.

        Incomplete local trees (model_index + partial weights, missing transformer
        tensors / git-LFS pointers) must NOT short-circuit to from_pretrained.
        """
        from src.common.utils.hf_cache_env import ensure_huggingface_cache_env

        ensure_huggingface_cache_env()
        self.logger.info("FLUX model ID: %s", self.hf_model_id)
        self.logger.info(
            "FLUX resolve: model_path=%s exists=%s hf_token_present=%s cuda=%s gpu_vram_mb=%.0f",
            self.model_path,
            self.model_path.exists(),
            self._hf_token_present(),
            bool(torch is not None and torch.cuda.is_available()),
            self._gpu_vram_mb(),
        )

        local_preflight = self.preflight_validate_package(
            self.model_path, source=self.hf_model_id or "(local)"
        )
        if local_preflight["ready"]:
            self._cache_status = "hit"
            self.logger.info(
                "[FLUX] CACHE HIT complete local Kontext weights at %s", self.model_path
            )
            print(f"[FLUX CACHE] HIT path={self.model_path}", flush=True)
            return str(self.model_path), None

        if self._has_transformer_weights(self.model_path) and (
            self._package_ready_for_pipeline(self.LEGACY_SCHNELL_PATH)
            or (
                self._is_complete_local_dir(self.LEGACY_SCHNELL_PATH)
                and self._local_t5_ready(self.LEGACY_SCHNELL_PATH)
            )
        ):
            self._cache_status = "hybrid"
            self.logger.info(
                "[FLUX] CACHE HIT hybrid Kontext load: transformer=%s shared_from=%s",
                self.model_path,
                self.LEGACY_SCHNELL_PATH,
            )
            print(
                f"[FLUX CACHE] HIT hybrid transformer={self.model_path} "
                f"shared={self.LEGACY_SCHNELL_PATH}",
                flush=True,
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
            # Hub-direct still allowed, but never pretend a broken local tree is ready.
            self._cache_status = "hub_direct"
            self.logger.info("Using Hugging Face Kontext source directly: %s", self.hf_model_id)
            print(
                f"[FLUX CACHE] MISS local incomplete → hub_direct source={self.hf_model_id}",
                flush=True,
            )
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
                f"[FLUX LOAD TRACE] process_id={os.getpid()} request_id=loader "
                f"loader_instance={id(self)} "
                f"cache_state={self._cache_status} pipeline_exists=true "
                f"pipeline_load_end model_reused=true",
                flush=True,
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

        print(
            f"[FLUX LOAD TRACE] process_id={os.getpid()} request_id=loader "
            f"loader_instance={id(self)} cache_state=pending "
            f"pipeline_exists=false pipeline_load_start model_reused=false",
            flush=True,
        )

        import threading

        t0 = self._mark("MODEL_INIT", time.perf_counter())
        self._progress("Initializing FLUX (cache check / dependencies)", 8)
        target_device = self.device_manager.resolve_device(self.device_setting)
        vram_before = self._gpu_vram_mb()
        print(
            f"[FLUX] START pipeline initialization device={target_device} "
            f"gpu_vram_mb={vram_before:.0f} cuda={bool(torch and torch.cuda.is_available())} "
            f"pid={os.getpid()}",
            flush=True,
        )

        stop_hb = threading.Event()

        def _load_heartbeat() -> None:
            started = time.perf_counter()
            while not stop_hb.wait(15.0):
                elapsed = int(time.perf_counter() - started)
                phase = self._load_phase or "Loading FLUX"
                pct = max(8, min(17, int(self._load_phase_pct or 8)))
                # Keep /status polls alive during multi-minute from_pretrained gaps.
                try:
                    self._progress(f"{phase} ({elapsed}s elapsed)", pct)
                except Exception:
                    pass

        hb = threading.Thread(target=_load_heartbeat, name="flux-load-hb", daemon=True)
        hb.start()

        try:
            result = self._load_after_heartbeat_start(
                t0=t0, target_device=target_device
            )
            print(
                f"[FLUX LOAD TRACE] process_id={os.getpid()} request_id=loader "
                f"loader_instance={id(self)} "
                f"cache_state={self._cache_status} "
                f"pipeline_exists={result is not None} "
                f"pipeline_load_end model_reused=false",
                flush=True,
            )
            return result
        finally:
            stop_hb.set()
            hb.join(timeout=2.0)

    def _load_after_heartbeat_start(self, *, t0: float, target_device: str) -> Any | None:
        # NF4 Diffusers packages require bitsandbytes + valid package metadata.
        if bool(getattr(self, "_want_nf4", True)):
            try:
                from src.common.utils.ensure_bitsandbytes import ensure_bitsandbytes

                t_bnb = self._mark("BITSANDBYTES_CHECK", time.perf_counter())
                bnb_ver = ensure_bitsandbytes(auto_install=True)
                self._mark("BITSANDBYTES_CHECK", t_bnb, end=True, extra=f"ver={bnb_ver}")
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
            self._progress("Resolving FLUX model source (disk cache vs download)", 10)
            t_src = self._mark("MODEL_SOURCE_RESOLVE", time.perf_counter())
            print(
                f"[FLUX] HF_HOME={os.environ.get('HF_HOME')!r} "
                f"HUGGINGFACE_HUB_CACHE={os.environ.get('HUGGINGFACE_HUB_CACHE')!r} "
                f"TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE')!r} "
                f"model_path={self.model_path}",
                flush=True,
            )
            pipeline_root, transformer_root = self._resolve_model_source()
            self._mark(
                "MODEL_SOURCE_RESOLVE",
                t_src,
                end=True,
                extra=(
                    f"disk_cache={self._cache_status} "
                    f"pipeline_root={pipeline_root} transformer_root={transformer_root}"
                ),
            )
            print(
                f"[FLUX] DISK_CACHE={'HIT' if self._cache_status in ('hit', 'hybrid') else 'MISS'} "
                f"cache_status={self._cache_status} "
                f"(FluxManager 'need_from_pretrained' means in-memory empty, not disk miss)",
                flush=True,
            )
        except Exception as exc:
            self.logger.error("%s", exc)
            if not self.allow_fallback:
                raise
            return None

        dtype = self._resolve_torch_dtype()

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

            # Hard gate: never call from_pretrained on an incomplete local package.
            if transformer_root is not None:
                tr_report = self.preflight_validate_package(
                    Path(transformer_root),
                    source=f"{self.hf_model_id or 'hybrid'}#transformer",
                )
                if not tr_report["transformer_weights"]:
                    raise RuntimeError(
                        "[FLUX ERROR] refusing transformer from_pretrained; "
                        f"missing={tr_report.get('missing')} path={transformer_root}"
                    )
            else:
                self._assert_local_package_loadable(
                    str(pipeline_root),
                    source=self.hf_model_id or str(pipeline_root),
                )

            if transformer_root is not None:
                self._progress("Loading FLUX transformer weights (from_pretrained)", 13)
                t_tr = self._mark("TRANSFORMER_LOAD", time.perf_counter())
                t_fp = self._mark(
                    "FROM_PRETRAINED",
                    time.perf_counter(),
                    extra="component=transformer",
                )
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
                self._mark(
                    "FROM_PRETRAINED",
                    t_fp,
                    end=True,
                    extra="component=transformer",
                )
                self._mark("TRANSFORMER_LOAD", t_tr, end=True)

            self._progress(
                "Loading FluxKontextPipeline (T5/CLIP/VAE) via from_pretrained",
                14,
            )
            t_pipe = self._mark("PIPELINE_LOAD", time.perf_counter())
            t_fp_pipe = self._mark(
                "FROM_PRETRAINED",
                time.perf_counter(),
                extra="component=FluxKontextPipeline",
            )
            self.logger.info("Loading FluxKontextPipeline from %s ...", pipeline_root)
            print(
                f"[FLUX LOAD] FluxKontextPipeline.from_pretrained root={pipeline_root}",
                flush=True,
            )
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
                err_l = str(direct_exc).lower()
                missing_weight = any(
                    s in err_l
                    for s in (
                        "diffusion_pytorch_model.safetensors",
                        "diffusion_pytorch_model.bin",
                        "no file named",
                    )
                )
                print(
                    f"[FLUX ERROR] from_pretrained failed root={pipeline_root} "
                    f"exc={type(direct_exc).__name__}: {direct_exc}",
                    flush=True,
                )
                # One controlled recovery — never retry the same broken local tree.
                if (
                    missing_weight
                    and self.hf_model_id
                    and not self._is_hub_repo_id(str(pipeline_root))
                    and transformer_root is None
                ):
                    print(
                        "[FLUX DOWNLOAD] recovery after load failure: purge+redownload",
                        flush=True,
                    )
                    repaired = self._ensure_hub_package(self.hf_model_id)
                    self._assert_local_package_loadable(
                        repaired, source=self.hf_model_id
                    )
                    pipeline_root = repaired
                    model_source = repaired
                    pipeline = FluxKontextPipeline.from_pretrained(
                        pipeline_root,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=use_low_cpu_mem,
                    )
                    trans = getattr(pipeline, "transformer", None)
                    used_bnb_4bit = bool(
                        trans is not None
                        and any(
                            "Linear4bit" in type(m).__name__
                            for m in list(trans.modules())[:50]
                        )
                    )
                else:
                    raise RuntimeError(
                        f"[FLUX ERROR] FluxKontextPipeline.from_pretrained failed for "
                        f"'{pipeline_root}': {type(direct_exc).__name__}: {direct_exc}"
                    ) from direct_exc

            if pipeline is None:
                raise RuntimeError(
                    f"Unable to construct FluxKontextPipeline from {pipeline_root}"
                )
            self._pipeline_assemble_time_s = round(time.perf_counter() - t_pipe, 2)
            self._mark(
                "FROM_PRETRAINED",
                t_fp_pipe,
                end=True,
                extra="component=FluxKontextPipeline",
            )
            self._mark("PIPELINE_LOAD", t_pipe, end=True)

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
                # - auto: offload on <14GB OR pre-Ampere (Tesla T4 sm_75).
                #   GPU-resident NF4 on T4 parks VAE to CPU then fails during
                #   Kontext image encode / denoise with deferred CUDA errors.
                #   Generation resolution is gated separately by flux_vram_policy
                #   (completion-first; 768 is NOT assumed safe on T4).
                offload_env = os.environ.get("FLUX_MODEL_CPU_OFFLOAD", "").strip().lower()
                physical_mb = self._gpu_vram_mb()
                prefer_offload = True
                if offload_env in ("0", "false", "no", "off"):
                    prefer_offload = False
                elif offload_env in ("1", "true", "yes", "on"):
                    prefer_offload = True
                else:
                    prefer_offload = physical_mb < 14000 or self._is_pre_ampere_gpu()

                self._progress("Configuring device / offload", 16)
                t_dev = self._mark("PIPELINE_DEVICE_SETUP", time.perf_counter())
                t_off = self._mark("OFFLOAD_SETUP", time.perf_counter())
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
                self._mark("OFFLOAD_SETUP", t_off, end=True)
                self._mark("PIPELINE_DEVICE_SETUP", t_dev, end=True)
                # After device/offload: keep VAE in fp32 when pipeline dtype is fp16.
                self.stabilize_flux_vae(pipeline)
            elif hasattr(pipeline, "to"):
                pipeline.to(target_device)
                self._offload_strategy = "none"
                self.stabilize_flux_vae(pipeline)

            self._configure_attention(pipeline)
            self._maybe_torch_compile(pipeline)

            self._pipeline = pipeline
            self._load_count += 1
            self._init_time_s = round(time.perf_counter() - t0, 2)
            alloc_after = 0.0
            if torch is not None and torch.cuda.is_available():
                alloc_after = round(torch.cuda.memory_allocated() / (1024**2), 1)
            self._mark("MODEL_INIT", t0, end=True)
            print(
                f"[FLUX] FLUX READY elapsed={self._init_time_s}s "
                f"cache={self._cache_status} download_s={self._download_time_s} "
                f"assemble_s={self._pipeline_assemble_time_s} "
                f"offload={self._offload_strategy} bnb4bit={used_bnb_4bit} "
                f"vram_alloc_mb={alloc_after}",
                flush=True,
            )
            self._progress("FLUX READY (in-memory)", 18)
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

    def _is_pre_ampere_gpu(self) -> bool:
        """True for Turing/Volta (e.g. Tesla T4 sm_75) where BF16 kernels are weak."""
        if torch is None or not torch.cuda.is_available():
            return False
        try:
            major, _minor = torch.cuda.get_device_capability(0)
            return int(major) < 8
        except Exception:
            return False

    def _resolve_torch_dtype(self) -> Any:
        """Pick pipeline / bnb compute dtype. Prefer FP16 on pre-Ampere (T4)."""
        if torch is None:
            return None
        forced = os.environ.get("FLUX_TORCH_DTYPE", "").strip().lower()
        if forced in ("fp16", "float16", "half"):
            return torch.float16
        if forced in ("bf16", "bfloat16"):
            return torch.bfloat16
        if not torch.cuda.is_available():
            return torch.float32
        # Tesla T4 (sm_75): native FP16; BF16 is emulated and breaks efficient
        # SDPA / some bitsandbytes compute paths → CUDA_ERROR during denoise.
        if self._is_pre_ampere_gpu():
            self.logger.info(
                "[FLUX] Using float16 compute dtype (pre-Ampere GPU; BF16 unsafe)"
            )
            return torch.float16
        if self.precision == "bfloat16":
            return torch.bfloat16
        return torch.float16

    def stabilize_flux_vae(self, pipeline: Any | None = None) -> dict[str, Any]:
        """
        Keep Flux VAE in float32 when the pipeline compute dtype is float16, and
        wrap encode/decode so tensors Diffusers casts to fp16 are re-aligned to
        the VAE dtype at the VAE boundary.

        Evidence (Kaggle T4):
        - FP16 VAE → NaN → completely black PNG (Diffusers #9096).
        - FP32 VAE + Diffusers ``image.to(dtype=prompt_embeds.dtype)`` (fp16)
          → ``Input type (c10::Half) and bias type (float) should be the same``
          at ``vae.encode`` / ``conv_in``.
        Transformer stays NF4 + float16; only the VAE path is adjusted.
        """
        info: dict[str, Any] = {
            "vae_dtype_before": None,
            "vae_dtype_after": None,
            "vae_device": None,
            "upcasted": False,
            "decode_wrapped": False,
            "encode_wrapped": False,
        }
        pipe = pipeline if pipeline is not None else self._pipeline
        if pipe is None or torch is None:
            return info
        vae = getattr(pipe, "vae", None)
        if vae is None:
            return info

        try:
            param = next(vae.parameters())
            info["vae_dtype_before"] = str(param.dtype)
            info["vae_device"] = str(param.device)
        except Exception:
            param = None

        try:
            if param is not None and param.dtype == torch.float16:
                vae.to(dtype=torch.float32)
                info["upcasted"] = True
                self.logger.info(
                    "[FLUX] VAE upcast float16→float32 "
                    "(Flux VAE fp16 yields NaN/black images on T4-class GPUs)"
                )
                print(
                    "[FLUX] VAE upcast float16→float32 (black-image mitigation)",
                    flush=True,
                )
        except Exception as exc:
            self.logger.warning("[FLUX] VAE fp32 upcast failed: %s", exc)

        def _cast_to_vae_dtype(sample: Any) -> Any:
            try:
                target = next(vae.parameters()).dtype
                if hasattr(sample, "to") and getattr(sample, "dtype", None) != target:
                    return sample.to(dtype=target)
            except Exception:
                pass
            return sample

        # Diffusers FluxKontext prepare_latents does:
        #   image = image.to(device=device, dtype=<pipeline/prompt dtype>)  # often fp16
        #   self.vae.encode(image)
        # With an FP32 VAE that raises:
        #   Input type (c10::Half) and bias type (float) should be the same
        # Cast at the VAE encode boundary only — do not change transformer dtype.
        if hasattr(vae, "encode") and not getattr(
            vae, "_fabricvision_fp32_encode_wrapped", False
        ):
            original_encode = vae.encode

            def _encode_matching_dtype(sample, *args, **kwargs):  # noqa: ANN001
                return original_encode(_cast_to_vae_dtype(sample), *args, **kwargs)

            vae.encode = _encode_matching_dtype  # type: ignore[method-assign]
            vae._fabricvision_fp32_encode_wrapped = True
            info["encode_wrapped"] = True
            self.logger.info("[FLUX] VAE.encode wrapped to cast inputs to VAE dtype")

        # Ensure decode latents match VAE dtype (pipeline may pass float16 latents).
        if hasattr(vae, "decode") and not getattr(
            vae, "_fabricvision_fp32_decode_wrapped", False
        ):
            original_decode = vae.decode

            def _decode_matching_dtype(latents, *args, **kwargs):  # noqa: ANN001
                return original_decode(_cast_to_vae_dtype(latents), *args, **kwargs)

            vae.decode = _decode_matching_dtype  # type: ignore[method-assign]
            vae._fabricvision_fp32_decode_wrapped = True
            info["decode_wrapped"] = True

        try:
            info["vae_dtype_after"] = str(next(vae.parameters()).dtype)
            info["vae_device"] = str(next(vae.parameters()).device)
        except Exception:
            pass
        return info

    def ensure_generation_devices(self) -> dict[str, Any]:
        """
        Restore modules required for Kontext generation after ``park_on_cpu``.

        GPU-resident NF4 parks VAE/encoders to CPU to free allocator headroom, but
        FluxKontext image conditioning + VAE decode require the VAE on CUDA.
        Leaving VAE on CPU after park caused deferred CUDA failures at ~50%
        (right as ``pipeline()`` starts conditioning encode / first denoise).

        Always runs ``stabilize_flux_vae`` (including model_cpu_offload) so T4
        float16 pipelines do not leave the VAE in the NaN/black fp16 path.
        """
        info: dict[str, Any] = {
            "offload_strategy": self._offload_strategy,
            "vae_device": None,
            "transformer_device": None,
            "restored": [],
            "vae_stabilized": None,
        }
        pipe = self._pipeline
        if pipe is None or torch is None:
            return info

        # VAE dtype stability is independent of offload strategy.
        info["vae_stabilized"] = self.stabilize_flux_vae(pipe)

        if not torch.cuda.is_available():
            return info
        if self._offload_strategy == "model_cpu_offload":
            # Accelerate hooks move modules on demand — do not fight them.
            try:
                info["vae_device"] = str(next(pipe.vae.parameters()).device)
            except Exception:
                pass
            return info

        target = torch.device("cuda")
        for name in ("vae",):
            mod = getattr(pipe, name, None)
            if mod is None or not hasattr(mod, "to"):
                continue
            try:
                before = None
                try:
                    before = next(mod.parameters()).device
                except Exception:
                    before = None
                # Preserve float32 after stabilize_flux_vae (do not cast back to fp16).
                mod.to(device=target)
                info["restored"].append(name)
                info[f"{name}_device_before"] = str(before)
                try:
                    info[f"{name}_device"] = str(next(mod.parameters()).device)
                except Exception:
                    info[f"{name}_device"] = "cuda"
                self.logger.info(
                    "[FLUX GENERATION] Restored %s → CUDA (was %s, strategy=%s, dtype=%s)",
                    name,
                    before,
                    self._offload_strategy,
                    next(mod.parameters()).dtype,
                )
            except Exception as exc:
                self.logger.warning(
                    "[FLUX GENERATION] Failed to restore %s to CUDA: %s", name, exc
                )
        try:
            tr = getattr(pipe, "transformer", None)
            if tr is not None:
                info["transformer_device"] = str(next(tr.parameters()).device)
        except Exception:
            pass
        return info

    def park_on_cpu(self) -> dict[str, float]:
        """
        Free GPU residency after / before FLUX jobs without breaking NF4 state.

        CRITICAL findings (Day-18 validation):
        - Never call ``transformer.to("cpu")`` on bitsandbytes NF4 modules
          (corrupts quant_state / device.index → ``getCurrentStream`` errors).
        - Do not call ``enable_model_cpu_offload()`` again here: ``maybe_free_model_hooks``
          already re-applies it. Double-enabling stacks hooks and breaks step 2+.
        - Prefer accelerate's ``maybe_free_model_hooks()`` as the sole offload reset.
        - For gpu_resident: parking VAE is OK for allocator cleanup, but callers MUST
          call ``ensure_generation_devices()`` before the next ``pipeline()`` call.
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
