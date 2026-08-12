from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.common.utils.utils import load_yaml_config, serialize_json
from src.features.custom_generator.inference.fabric_conditioning import (
    build_garment_conditioning_image,
    is_match_fabric_color,
    normalize_color_key,
    recolor_fabric_base_preserving_motifs,
    save_fabric_recolor_audit,
)
from src.features.custom_generator.inference.garment_output import (
    persist_and_verify_garment_png,
)
from src.features.custom_generator.inference.flux_inference import FLUXInferenceEngine
from src.features.custom_generator.model.flux_model_loader import FLUXModelLoader
from src.features.custom_generator.prompting.garment_prompt_builder import GarmentPromptBuilder
from src.features.custom_generator.validation.garment_validator import GarmentValidator


ProgressCallback = Optional[Callable[[str, int], None]]


def normalize_generation_mode(mode: Optional[str]) -> str:
    """
    Map UI / API mode labels onto canonical keys.

    Low-VRAM path (RTX 3050 6GB): preview | standard | production
    Quality path (~16GB+): quality_15 | quality_20 | quality_30 | quality_768
    """
    raw = (mode or "").strip().lower().replace("-", " ").replace("_", " ")
    raw = " ".join(raw.split())
    if raw in ("preview", "fast preview", "fast", "fastpreview", "low vram", "lowvram"):
        return "preview"
    if raw in ("production", "high quality", "hq", "high", "highquality", "prod"):
        return "production"
    if raw in ("standard", "default", "normal", ""):
        return "standard"
    # Quality-first presets (do not collapse into production)
    if raw in ("quality 15", "quality15", "q15", "steps 15"):
        return "quality_15"
    if raw in ("quality 20", "quality20", "q20", "steps 20"):
        return "quality_20"
    if raw in ("quality 30", "quality30", "q30", "steps 30"):
        return "quality_30"
    if raw in ("quality 768", "quality768", "q768", "res 768"):
        return "quality_768"
    if raw in ("quality", "quality first", "qualityfirst"):
        return "quality_20"
    # Unknown labels default to standard (quality-safe default for low-VRAM)
    return "standard"


@dataclass
class GarmentGenerationConfig:
    """Runtime configuration for FLUX.1-Kontext garment generation."""

    config_dir: str = "configs"
    model_path: str = "models/flux-kontext"
    device: str = "auto"
    precision: str = "bfloat16"
    output_root: str = "outputs/generated_garments"
    experiments_root: str = "experiments"
    height: int = 512
    width: int = 512
    num_inference_steps: int = 4
    guidance_scale: float = 2.5
    seed: int = 42
    prompt_version: str = "v2.0"
    generation_mode: str = "standard"
    config_path: Optional[str] = None
    allow_fallback: bool = True
    png_compress_level: int = 3
    png_optimize: bool = False
    enable_torch_compile: bool = False
    attention_backend: str = "auto"
    profile: bool = True
    # Internal: resolved canonical mode key after config load
    mode_key: str = field(default="standard", repr=False)


class GarmentGenerationPipeline:
    """Fabric-conditioned FLUX.1-Kontext garment synthesis pipeline."""

    def __init__(
        self,
        config: Optional[GarmentGenerationConfig] = None,
        model_loader: Optional[FLUXModelLoader] = None,
        inference_engine: Optional[FLUXInferenceEngine] = None,
        validator: Optional[GarmentValidator] = None,
    ) -> None:
        self.config = config or GarmentGenerationConfig()
        self.logger = logging.getLogger("fabricvision.garment_generation.pipeline")
        self._load_config_files()

        self.prompt_builder = GarmentPromptBuilder(self.config.config_dir)
        self.validator = validator or GarmentValidator()
        self.model_loader = model_loader or FLUXModelLoader(
            model_path=self.config.model_path,
            device=self.config.device,
            precision=self.config.precision,
            allow_fallback=self.config.allow_fallback,
            enable_torch_compile=self.config.enable_torch_compile,
            attention_backend=self.config.attention_backend,
        )
        self.inference_engine = inference_engine or FLUXInferenceEngine(
            self.model_loader,
            allow_fallback=self.config.allow_fallback,
        )

    def _mode_config_key(self, mode: str) -> str:
        canonical = normalize_generation_mode(mode)
        # Prefer canonical blocks; fall back to legacy aliases in YAML
        return canonical

    def _load_config_files(self) -> None:
        flux_cfg: Dict[str, Any] = {}
        if self.config.config_path:
            flux_cfg = load_yaml_config(self.config.config_path) or {}
            self.config.model_path = flux_cfg.get("model_path", self.config.model_path)
            self.config.device = flux_cfg.get("device", self.config.device)
            self.config.precision = flux_cfg.get("precision", self.config.precision)
            self.config.enable_torch_compile = bool(
                flux_cfg.get("enable_torch_compile", self.config.enable_torch_compile)
            )
            self.config.attention_backend = str(
                flux_cfg.get("attention_backend", self.config.attention_backend)
            )
            self.config.profile = bool(flux_cfg.get("profile", self.config.profile))
            # Propagate T5 sequence budget — measured 512 → ~360s encode on RTX 3050
            max_seq = flux_cfg.get("max_sequence_length")
            if max_seq is not None and not os.environ.get("FLUX_MAX_SEQUENCE_LENGTH"):
                os.environ["FLUX_MAX_SEQUENCE_LENGTH"] = str(int(max_seq))
            if flux_cfg.get("enable_vae_tiling") is False:
                os.environ.setdefault("FLUX_VAE_TILING", "false")
            # Day-17: preencode at seq=128 ≈ 48–50s and frees T5 before diffusion.
            if "FLUX_PREENCODE_PROMPT" not in os.environ:
                preencode = flux_cfg.get("preencode_prompt", True)
                os.environ["FLUX_PREENCODE_PROMPT"] = (
                    "true" if bool(preencode) else "false"
                )

        # Env overrides (Phase 19)
        env_compile = os.environ.get("FLUX_ENABLE_TORCH_COMPILE", "").strip().lower()
        if env_compile in ("1", "true", "yes", "on"):
            self.config.enable_torch_compile = True
        elif env_compile in ("0", "false", "no", "off"):
            self.config.enable_torch_compile = False

        env_attn = os.environ.get("FLUX_ATTENTION_BACKEND", "").strip()
        if env_attn:
            self.config.attention_backend = env_attn

        env_profile = os.environ.get("FLUX_PROFILE", "").strip().lower()
        if env_profile in ("1", "true", "yes", "on"):
            self.config.profile = True
            os.environ["FLUX_PROFILE"] = "true"
        elif env_profile in ("0", "false", "no", "off"):
            self.config.profile = False
            os.environ["FLUX_PROFILE"] = "false"
        elif self.config.profile:
            os.environ.setdefault("FLUX_PROFILE", "true")

        mode_key = self._mode_config_key(self.config.generation_mode)
        self.config.mode_key = mode_key
        self.config.generation_mode = {
            "preview": "Preview",
            "standard": "Standard",
            "production": "Production",
            "quality_15": "Quality_15",
            "quality_20": "Quality_20",
            "quality_30": "Quality_30",
            "quality_768": "Quality_768",
        }.get(mode_key, mode_key.replace("_", " ").title())

        mode_cfg = None
        if isinstance(flux_cfg, dict):
            # Try canonical key, then legacy aliases
            aliases = [mode_key]
            if mode_key == "preview":
                aliases.extend(["fast_preview"])
            elif mode_key == "production":
                aliases.extend(["high_quality"])
            for key in aliases:
                if isinstance(flux_cfg.get(key), dict):
                    mode_cfg = flux_cfg[key]
                    break

        if isinstance(mode_cfg, dict):
            self.config.height = int(mode_cfg.get("height", self.config.height))
            self.config.width = int(mode_cfg.get("width", self.config.width))
            self.config.num_inference_steps = int(
                mode_cfg.get("num_inference_steps", self.config.num_inference_steps)
            )
            self.config.guidance_scale = float(
                mode_cfg.get("guidance_scale", self.config.guidance_scale)
            )

        # Resolution knobs:
        # - FLUX_GENERATION_RESOLUTION → Preview/Standard (and Production fallback)
        # - FLUX_PRODUCTION_RESOLUTION / FLUX_PRODUCTION_SIZE → Production only (≥700 on T4)
        from src.features.custom_generator.inference.flux_inference import (
            resolve_flux_generation_resolution,
            resolve_flux_production_guidance,
            resolve_flux_production_resolution,
            resolve_flux_production_steps,
        )

        if mode_key == "production":
            size = resolve_flux_production_resolution(default=self.config.height)
            # On low-VRAM the production VRAM policy will clamp to 512 later.
            self.config.height = size
            self.config.width = size
            self.config.num_inference_steps = resolve_flux_production_steps(
                default=self.config.num_inference_steps
            )
            self.config.guidance_scale = resolve_flux_production_guidance(
                default=self.config.guidance_scale
            )
        else:
            res_env = os.environ.get("FLUX_GENERATION_RESOLUTION", "").strip()
            if res_env.isdigit():
                size = resolve_flux_generation_resolution(default=self.config.height)
                self.config.height = size
                self.config.width = size

        std_steps = os.environ.get("FLUX_STANDARD_STEPS", "").strip()
        if mode_key == "standard" and std_steps.isdigit():
            self.config.num_inference_steps = max(1, int(std_steps))

        preview_steps = os.environ.get("FLUX_PREVIEW_STEPS", "").strip()
        if mode_key == "preview" and preview_steps.isdigit():
            self.config.num_inference_steps = max(1, int(preview_steps))

        # T4 / 16GB+ quality path: Standard UI mode must not stay at the RTX 3050
        # 512×3 preset (known soft/blurry). Prefer 768 / 12 steps unless overridden.
        self._apply_high_vram_standard_defaults(mode_key)
        # Production: 700+ on Kaggle T4; 512 clamp on local low-VRAM.
        self._apply_production_vram_defaults(mode_key)

        gen_cfg_path = Path(self.config.config_dir) / "generation_config.yaml"
        if not gen_cfg_path.exists():
            gen_cfg_path = Path(self.config.config_dir) / "custom_generator" / "generation_config.yaml"
        if gen_cfg_path.exists():
            loaded_gen = load_yaml_config(gen_cfg_path) or {}
            if not mode_cfg:
                self.config.num_inference_steps = loaded_gen.get(
                    "default_num_inference_steps", self.config.num_inference_steps
                )
                self.config.guidance_scale = loaded_gen.get(
                    "default_guidance_scale", self.config.guidance_scale
                )
                self.config.height = loaded_gen.get("default_height", self.config.height)
                self.config.width = loaded_gen.get("default_width", self.config.width)
            # Only apply global defaults when caller left these at the dataclass
            # default. Otherwise an explicit output_root (e.g. pytest tmp_path
            # isolation, benchmark scripts) was silently overwritten — a real
            # stabilization bug, not a feature.
            _defaults = GarmentGenerationConfig.__dataclass_fields__
            if self.config.seed == _defaults["seed"].default:
                self.config.seed = loaded_gen.get("seed", self.config.seed)
            if self.config.output_root == _defaults["output_root"].default:
                self.config.output_root = loaded_gen.get("output_root", self.config.output_root)
            self.config.png_compress_level = int(
                loaded_gen.get("png_compress_level", self.config.png_compress_level)
            )
            self.config.png_optimize = bool(
                loaded_gen.get("png_optimize", self.config.png_optimize)
            )

    def _gpu_vram_mb(self) -> float:
        try:
            import torch

            if torch.cuda.is_available():
                return float(torch.cuda.get_device_properties(0).total_memory) / (1024**2)
        except Exception:
            return 0.0
        return 0.0

    def _apply_high_vram_standard_defaults(self, mode_key: str) -> None:
        """
        Completion-first Standard policy from measured VRAM headroom.

        Do NOT assume T4 ≡ 768×768 GPU-resident. That path OOMs when NF4 +
        Kontext activations exceed free headroom. Env overrides still win.
        """
        if mode_key != "standard":
            return

        from src.features.custom_generator.inference.flux_vram_policy import (
            log_vram,
            select_standard_generation_policy,
        )

        diag = log_vram("before_standard_policy")
        offload = None
        loader = getattr(self, "model_loader", None) or getattr(
            getattr(self, "inference_engine", None), "model_loader", None
        )
        if loader is not None:
            offload = getattr(loader, "_offload_strategy", None)

        policy = select_standard_generation_policy(
            physical_mb=diag.physical_total_mb or self._gpu_vram_mb(),
            free_mb=diag.free_mb,
            offload_strategy=offload,
        )

        self.config.height = int(policy.height)
        self.config.width = int(policy.width)
        self.config.num_inference_steps = int(policy.num_inference_steps)
        if self.config.guidance_scale < float(policy.guidance_scale):
            self.config.guidance_scale = float(policy.guidance_scale)

        if policy.enable_vae_tiling:
            os.environ.setdefault("FLUX_VAE_TILING", "true")

        self.logger.info(
            "[FLUX] Standard policy profile=%s %sx%s steps=%s guidance=%s "
            "prefer_offload=%s reason=%s gpu=%s",
            policy.profile,
            self.config.width,
            self.config.height,
            self.config.num_inference_steps,
            self.config.guidance_scale,
            policy.prefer_model_cpu_offload,
            policy.reason,
            diag.gpu_name,
        )
        try:
            self._vram_policy = {
                **policy.__dict__,
                "diagnostics": diag.as_dict(),
            }
        except Exception:
            self._vram_policy = {"profile": policy.profile, "reason": policy.reason}

    def _apply_production_vram_defaults(self, mode_key: str) -> None:
        """VRAM-aware Production policy: 512² / 12–16 on 6GB; never silent 768."""
        if mode_key != "production":
            return

        from src.features.custom_generator.inference.flux_vram_policy import (
            log_vram,
            select_production_generation_policy,
        )

        diag = log_vram("before_production_policy")
        offload = None
        loader = getattr(self, "model_loader", None) or getattr(
            getattr(self, "inference_engine", None), "model_loader", None
        )
        if loader is not None:
            offload = getattr(loader, "_offload_strategy", None)

        policy = select_production_generation_policy(
            physical_mb=diag.physical_total_mb or self._gpu_vram_mb(),
            free_mb=diag.free_mb,
            offload_strategy=offload,
            yaml_height=int(self.config.height),
            yaml_steps=int(self.config.num_inference_steps),
            yaml_guidance=float(self.config.guidance_scale),
        )

        self.config.height = int(policy.height)
        self.config.width = int(policy.width)
        self.config.num_inference_steps = int(policy.num_inference_steps)
        self.config.guidance_scale = float(policy.guidance_scale)

        if policy.enable_vae_tiling:
            os.environ.setdefault("FLUX_VAE_TILING", "true")

        self.logger.info(
            "[FLUX] Production policy profile=%s %sx%s steps=%s guidance=%s "
            "prefer_offload=%s reason=%s gpu=%s",
            policy.profile,
            self.config.width,
            self.config.height,
            self.config.num_inference_steps,
            self.config.guidance_scale,
            policy.prefer_model_cpu_offload,
            policy.reason,
            diag.gpu_name,
        )
        try:
            self._vram_policy = {
                **policy.__dict__,
                "diagnostics": diag.as_dict(),
            }
        except Exception:
            self._vram_policy = {"profile": policy.profile, "reason": policy.reason}

    def run(
        self,
        fabric_metadata: Dict[str, Any],
        user_customization: Dict[str, Any],
        output_filename: Optional[str] = None,
        reference_image: Optional[Any] = None,
        progress_callback: ProgressCallback = None,
    ) -> Dict[str, Any]:
        """Execute Kontext prompt build, fabric conditioning, inference, validation."""
        t_total = time.perf_counter()
        timings: Dict[str, float] = {}

        def _progress(step: str, pct: int) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(step, pct)
                except Exception:
                    pass

        def _mark(label: str, t_seg: float, *, end: bool = False) -> float:
            now = time.perf_counter()
            if end:
                msg = (
                    f"[GENERATION] {label} END t={now:.2f} "
                    f"elapsed={now - t_seg:.2f}s"
                )
            else:
                msg = f"[GENERATION] {label} START t={now:.2f}"
            self.logger.info(msg)
            print(msg, flush=True)
            return now

        if reference_image is None:
            raise RuntimeError(
                "Fabric reference image is required for FLUX.1-Kontext garment generation."
            )

        # Re-evaluate Standard settings with live free VRAM after model residency.
        try:
            mode_key = normalize_generation_mode(self.config.generation_mode)
            if mode_key == "standard":
                self._apply_high_vram_standard_defaults("standard")
        except Exception as pol_exc:
            self.logger.warning("Runtime VRAM policy refresh skipped: %s", pol_exc)

        # Keep fabric in memory — avoid re-reading from disk during conditioning.
        fabric_image = reference_image
        if hasattr(fabric_image, "convert"):
            fabric_image = fabric_image.convert("RGB")

        _progress("Preparing fabric", 22)
        t0 = _mark("FABRIC PREPROCESS", time.perf_counter())
        color_trace_ui = normalize_color_key(
            str(user_customization.get("color") or fabric_metadata.get("color") or "")
        ) or "match_fabric"
        try:
            from src.features.custom_generator.inference.fabric_appearance import (
                describe_fabric_appearance,
            )

            appearance = describe_fabric_appearance(fabric_image)
            # Color modes:
            # - Match Fabric: pixel palette is authoritative (preserve upload colors).
            # - Explicit UI color: target color is authoritative; pixels describe
            #   texture/pattern only (never silently ignore a real Color selection).
            ui_selected = (
                user_customization.get("color")
                or fabric_metadata.get("color")
                or ""
            )
            color_trace_ui = normalize_color_key(str(ui_selected)) or "match_fabric"
            explicit_color = not is_match_fabric_color(str(ui_selected))
            # Harden: explicit Color field implies recolor even if force_recolor was omitted.
            force_recolor = bool(user_customization.get("force_recolor")) or explicit_color
            source_palette = appearance.get("dominant_color_names") or []
            fractions = appearance.get("color_fractions") or {}
            if fractions:
                base_name = max(fractions.items(), key=lambda kv: kv[1])[0]
            else:
                base_name = source_palette[0] if source_palette else None
            motif_colors = [c for c in source_palette if c != base_name]
            if force_recolor and explicit_color:
                target = normalize_color_key(str(ui_selected))
                user_customization = {
                    **user_customization,
                    "color": target,
                    "force_recolor": True,
                }
                pattern_hint = appearance.get("pattern_hint") or fabric_metadata.get(
                    "pattern"
                )
                fabric_metadata = {
                    **fabric_metadata,
                    "pattern": pattern_hint,
                    "dominant_colors": [target],
                    "color_source": "ui_recolor",
                    # Pattern/texture only — do not embed source palette as "preserve colors".
                    "fabric_appearance": (
                        f"{pattern_hint} textile texture"
                        if pattern_hint
                        else appearance.get("appearance_summary")
                    ),
                    "source_palette": source_palette,
                    "motif_colors": motif_colors,
                    "base_color_name": base_name,
                }
                self.logger.info(
                    "[FLUX COLOR] source=ui_recolor palette=%s motifs=%s",
                    fabric_metadata.get("dominant_colors"),
                    motif_colors,
                )
            else:
                fabric_metadata = {
                    **fabric_metadata,
                    "pattern": appearance.get("pattern_hint")
                    or fabric_metadata.get("pattern"),
                    "dominant_colors": appearance.get("dominant_color_names")
                    or fabric_metadata.get("dominant_colors"),
                    "color_source": "fabric_pixels",
                    "fabric_appearance": appearance.get("appearance_summary"),
                    "source_palette": source_palette,
                    "motif_colors": motif_colors,
                    "base_color_name": base_name,
                }
                user_customization = {
                    k: v for k, v in user_customization.items() if k != "color"
                }
                user_customization["force_recolor"] = False
                self.logger.info(
                    "[FLUX COLOR] source=fabric_pixels palette=%s (Match Fabric)",
                    fabric_metadata.get("dominant_colors"),
                )
            color_mode = (
                "explicit"
                if fabric_metadata.get("color_source") == "ui_recolor"
                else "match_fabric"
            )
            target_color = (
                fabric_metadata.get("dominant_colors", [None])[0]
                if color_mode == "explicit"
                else "match_fabric"
            )
            for line in (
                "[FLUX COLOR DEBUG]",
                f"ui_selected_color={normalize_color_key(str(ui_selected)) or 'match_fabric'}",
                f"color_mode={color_mode}",
                f"source_palette={source_palette}",
                f"target_color={target_color}",
                f"recolor_enabled={color_mode == 'explicit'}",
            ):
                self.logger.info(line)
                print(line, flush=True)
            self.logger.info("Fabric appearance cues: %s", appearance)
        except Exception as exc:
            self.logger.warning("Fabric appearance enrichment skipped: %s", exc)
            color_mode = "match_fabric"
            target_color = None
        timings["fabric_appearance_s"] = round(time.perf_counter() - t0, 3)
        _mark("FABRIC PREPROCESS", t0, end=True)

        # Prompt text build (token encode happens later inside FLUX inference).
        _progress("Preparing garment conditioning", 32)
        t0 = _mark("CONDITIONING", time.perf_counter())
        garment_type = str(user_customization.get("garment_type") or "shirt")
        sleeve = str(user_customization.get("sleeve") or "")
        conditioning_target = (
            str(user_customization.get("color"))
            if bool(user_customization.get("force_recolor"))
            and not is_match_fabric_color(str(user_customization.get("color") or ""))
            else None
        )
        conditioning_image = build_garment_conditioning_image(
            fabric_image=fabric_image,
            garment_type=garment_type,
            width=self.config.width,
            height=self.config.height,
            sleeve=sleeve,
            target_color=conditioning_target,
        )
        conditioning_recolored = conditioning_target is not None
        recolor_audit = getattr(
            build_garment_conditioning_image, "last_recolor_audit", None
        )
        self.logger.info(
            "[FLUX COLOR DEBUG] conditioning_recolored=%s",
            conditioning_recolored,
        )
        print(
            f"[FLUX COLOR DEBUG] conditioning_recolored={conditioning_recolored}",
            flush=True,
        )
        timings["fabric_conditioning_s"] = round(time.perf_counter() - t0, 3)
        _mark("CONDITIONING", t0, end=True)

        _progress("Encoding prompt", 42)
        t0 = _mark("PROMPT BUILD", time.perf_counter())
        positive_prompt, negative_prompt = self.prompt_builder.build_kontext_prompt(
            fabric_metadata=fabric_metadata,
            user_customization=user_customization,
        )
        timings["prompt_building_s"] = round(time.perf_counter() - t0, 3)
        prompt_stats = getattr(self.prompt_builder, "last_prompt_stats", {}) or {}
        self.logger.info("=== FINAL POSITIVE PROMPT ===\n%s", positive_prompt)
        self.logger.info("=== FINAL NEGATIVE PROMPT ===\n%s", negative_prompt)
        _mark("PROMPT BUILD", t0, end=True)

        # Persist stage images for A/B audits (original already in uploads; save cond)
        debug_dir = Path(self.config.output_root) / "audit_stages"
        debug_dir.mkdir(parents=True, exist_ok=True)
        garment_id = output_filename or f"garment_{uuid.uuid4().hex[:8]}"
        stage_id = garment_id
        try:
            fabric_image.save(debug_dir / f"{stage_id}_A_fabric.png")
            conditioning_image.save(debug_dir / f"{stage_id}_B_conditioning.png")
            if recolor_audit is not None and conditioning_target:
                color_key = normalize_color_key(conditioning_target)
                color_audit_dir = debug_dir / f"{stage_id}_color_{color_key}"
                paths = save_fabric_recolor_audit(
                    recolor_audit, color_audit_dir, color_key
                )
                final_name = f"{color_key}_final_conditioning.png"
                final_path = color_audit_dir / final_name
                conditioning_image.save(final_path)
                paths[f"{color_key}_final_conditioning"] = str(final_path)
                self.logger.info(
                    "[FLUX COLOR AUDIT] saved=%s",
                    paths,
                )
                print(
                    f"[FLUX COLOR AUDIT] {color_key}_final_conditioning={final_path}",
                    flush=True,
                )
            self.logger.info(
                "[FLUX STAGES] fabric=%sx%s conditioning=%sx%s → %s",
                fabric_image.size[0],
                fabric_image.size[1],
                conditioning_image.size[0],
                conditioning_image.size[1],
                debug_dir,
            )
        except Exception as stage_exc:
            self.logger.warning("Stage image save skipped: %s", stage_exc)
        self.logger.info(
            "Kontext conditioning: garment silhouette filled with uploaded fabric "
            "(mode=%s %sx%s steps=%s guidance=%s)",
            self.config.generation_mode,
            self.config.width,
            self.config.height,
            self.config.num_inference_steps,
            self.config.guidance_scale,
        )

        raw_dir = Path(self.config.output_root) / "raw"
        raw_path = raw_dir / f"{garment_id}_raw.png"
        flux_input_path = debug_dir / f"{garment_id}_flux_input.png"

        image = self.inference_engine.generate(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            reference_image=conditioning_image,
            height=self.config.height,
            width=self.config.width,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            seed=self.config.seed,
            progress_callback=progress_callback,
            save_raw_path=str(raw_path),
            flux_input_audit_path=str(flux_input_path),
            color_trace={
                "selected_ui_color": color_trace_ui,
                "color_mode": "explicit" if conditioning_recolored else "match_fabric",
                "force_recolor": bool(user_customization.get("force_recolor")),
                "target_color": conditioning_target or "match_fabric",
                "conditioning_recolored": conditioning_recolored,
            },
        )

        # Apply contour-guided detail refiner (sharpens neckline, sleeves, seams; preserves fabric identity)
        try:
            from src.features.custom_generator.inference.garment_detail_refiner import (
                GarmentDetailRefiner,
            )

            refiner = GarmentDetailRefiner()
            image = refiner.refine(image, mask_fabric_interior=True, enabled=True)
        except Exception as ref_exc:
            self.logger.warning("Garment detail refiner skipped: %s", ref_exc)

        # Deterministic UI color: FLUX often preserves original base hues even when
        # the conditioning image is correctly recolored. Apply the same base-only
        # motif-preserving recolor to the generated garment (studio bg protected).
        if conditioning_target:
            try:
                post = recolor_fabric_base_preserving_motifs(
                    image,
                    conditioning_target,
                    protect_studio_background=True,
                    strength=1.0,
                )
                image = post.image
                post_dir = debug_dir / f"{garment_id}_post_{normalize_color_key(conditioning_target)}"
                save_fabric_recolor_audit(
                    post, post_dir, normalize_color_key(conditioning_target)
                )
                image.save(post_dir / f"{normalize_color_key(conditioning_target)}_post_final.png")
                self.logger.info(
                    "[FLUX COLOR] post-generation base recolor applied target=%s "
                    "coverage=%.3f",
                    conditioning_target,
                    post.base_coverage,
                )
                print(
                    f"[FLUX COLOR] post_generation_recolor=true target={conditioning_target}",
                    flush=True,
                )
            except Exception as post_exc:
                self.logger.warning("Post-generation base recolor skipped: %s", post_exc)

        color_val = fabric_metadata.get("dominant_colors") or user_customization.get("color")
        val_result = self.validator.validate(
            image,
            target_garment=garment_type,
            target_color=str(color_val) if color_val else None,
        )

        _progress("Saving result", 94)
        t0 = _mark("IMAGE SAVE", time.perf_counter())
        output_base = Path(self.config.output_root)
        images_dir = output_base / "images"
        metadata_dir = output_base / "metadata"
        images_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        image_path = images_dir / f"{garment_id}.png"
        # Visually lossless PNG: moderate compress_level is faster without quality loss.
        # Do NOT resize — save exact model output resolution for frontend/download.
        if hasattr(image, "size"):
            self.logger.info(
                "[FLUX] Final save size=%sx%s (must match generation; no downscale)",
                image.size[0],
                image.size[1],
            )
        try:
            output_stats = persist_and_verify_garment_png(
                image,
                image_path,
                expected_size=(self.config.width, self.config.height),
                compress_level=int(self.config.png_compress_level),
            )
        except Exception as save_exc:
            self.logger.error("Final garment image save/verify failed: %s", save_exc)
            print(f"[GARMENT OUTPUT] SAVE FAILED: {save_exc}", flush=True)
            raise RuntimeError(
                f"Failed to persist final garment image to {image_path}: {save_exc}"
            ) from save_exc

        if not Path(image_path).exists():
            raise RuntimeError(f"Final garment image missing after save: {image_path}")

        raw_exists = Path(raw_path).exists()
        self.logger.info(
            "[GARMENT OUTPUT] final=%s raw_exists=%s raw=%s",
            image_path.resolve(),
            raw_exists,
            raw_path,
        )
        print(
            f"[GARMENT OUTPUT] FINAL PATH={image_path.resolve()} "
            f"RAW PATH={raw_path} raw_exists={raw_exists}",
            flush=True,
        )
        timings["image_saving_s"] = round(time.perf_counter() - t0, 3)
        timings["total_pipeline_s"] = round(time.perf_counter() - t_total, 3)
        _mark("IMAGE SAVE", t0, end=True)

        if self.config.profile:
            self.logger.info(
                "\n[FLUX PIPELINE PROFILE]\n"
                "Fabric appearance: %.3f sec\n"
                "Prompt building: %.3f sec\n"
                "Fabric conditioning: %.3f sec\n"
                "Image saving: %.3f sec\n"
                "TOTAL pipeline: %.3f sec\n",
                timings["fabric_appearance_s"],
                timings["prompt_building_s"],
                timings["fabric_conditioning_s"],
                timings["image_saving_s"],
                timings["total_pipeline_s"],
            )

        t_meta = _mark("METADATA", time.perf_counter())
        metadata = {
            "garment_id": garment_id,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "prompt_stats": prompt_stats,
            "fabric_metadata": fabric_metadata,
            "user_customization": user_customization,
            "validation": val_result,
            "generation_mode": self.config.generation_mode,
            "mode_key": self.config.mode_key,
            "model": "FLUX.1-Kontext",
            "height": self.config.height,
            "width": self.config.width,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "image_path": str(image_path),
            "raw_image_path": str(raw_path),
            "pipeline_timings": timings,
            "vram_policy": getattr(self, "_vram_policy", None),
            "output_stats": output_stats,
        }
        serialize_json(metadata, metadata_dir / f"{garment_id}.json")

        exp_root = Path(self.config.experiments_root)
        flux_exp_dir = exp_root / "generation_results"
        flux_exp_dir.mkdir(parents=True, exist_ok=True)
        serialize_json(
            {
                **metadata,
                "stats": getattr(self.inference_engine, "last_execution_stats", {}),
            },
            flux_exp_dir / f"{garment_id}_exp.json",
        )

        stats = getattr(self.inference_engine, "last_execution_stats", None)
        if stats is not None:
            stats["generation_mode"] = self.config.generation_mode
            stats["mode_key"] = self.config.mode_key
            stats["pipeline_timings"] = timings
            stats["prompt_stats"] = prompt_stats
        _mark("METADATA", t_meta, end=True)

        _progress("Completed", 100)

        return {
            "image_path": str(image_path),
            "output_path": str(image_path),
            "metadata": metadata,
            "validation": val_result,
        }
