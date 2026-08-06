from __future__ import annotations

import gc
import logging
import random
import time
from typing import Any, Optional, Dict
from PIL import Image, ImageDraw

try:
    import torch
except ImportError:
    torch = None


class FLUXInferenceEngine:
    """Run diffusion generation with FLUX Kontext model loader and VRAM tracking."""

    def __init__(self, model_loader: Any, allow_fallback: bool = True) -> None:
        self.model_loader = model_loader
        self.allow_fallback = allow_fallback
        self.logger = logging.getLogger("fabricvision.garment_generation.inference")
        self.last_execution_stats: Dict[str, Any] = {}

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        reference_image: Optional[Image.Image] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 3.5,
        seed: Optional[int] = 42,
    ) -> Image.Image:
        """Generate a garment image given prompt parameters and optional reference image conditioning."""
        t_start = time.time()
        pipeline = getattr(self.model_loader, "pipeline", None)
        t_model_load_start = time.time()
        if pipeline is None and hasattr(self.model_loader, "load"):
            pipeline = self.model_loader.load()
        t_model_load_end = time.time()
        model_load_time = round(t_model_load_end - t_model_load_start, 2) if pipeline is not None else 0.0

        if seed is not None:
            random.seed(seed)
            if torch is not None and torch.cuda.is_available():
                torch.manual_seed(seed)

        vram_before = torch.cuda.memory_allocated() / (1024 ** 2) if (torch and torch.cuda.is_available()) else 0.0

        if pipeline is None:
            msg = "FLUX pipeline not initialized or weights missing."
            self.logger.warning(msg)
            if not self.allow_fallback:
                raise RuntimeError(f"Real FLUX model execution required but failed: {msg}")
            
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

        # Convert to FluxKontextPipeline in-memory via from_pipe when reference_image is provided
        if reference_image is not None:
            try:
                from diffusers import FluxKontextPipeline
                from_pipe_fn = getattr(FluxKontextPipeline, "from_pipe", None)
                if not isinstance(pipeline, FluxKontextPipeline) and callable(from_pipe_fn):
                    self.logger.info("Converting resident FLUX pipeline to FluxKontextPipeline via from_pipe (0.0s latency)...")
                    pipeline = from_pipe_fn(pipeline)
                    if hasattr(self.model_loader, "_pipeline"):
                        self.model_loader._pipeline = pipeline
            except Exception as kontext_exc:
                self.logger.warning("FluxKontextPipeline conversion notice (%s); proceeding with active pipeline", kontext_exc)

        self.logger.info("Inference started (Reference Image Provided: %s)", reference_image is not None)
        self.logger.info("Before inference VRAM: %.2f MB", vram_before)
        
        t_pipe_start = time.time()
        step_times: Dict[str, float] = {}

        try:
            generator = torch.Generator().manual_seed(seed) if (seed is not None and torch and torch.cuda.is_available()) else None
            
            # Step callback logging & sub-stage timing tracking
            def step_callback(pipe: Any, step_index: int, timestep: Any, callback_kwargs: Any) -> Any:
                now = time.time()
                if step_index == 0:
                    step_times["prompt_encoding_end"] = now
                    step_times["step_start"] = now
                elif step_index == num_inference_steps - 1:
                    step_times["step_end"] = now
                self.logger.info("Step %d/%d", step_index + 1, num_inference_steps)
                return callback_kwargs

            # Construct instruction-guided prompt when reference image is supplied
            final_prompt = prompt
            if reference_image is not None:
                final_prompt = (
                    f"{prompt}. Preserve the exact pattern, texture, colors, and visual identity of the reference image. "
                    "Synthesize a high-fashion standalone garment output using the exact fabric pattern, colors, print, weave texture, and design identity from the reference image."
                )

            kwargs: Dict[str, Any] = {
                "prompt": final_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "callback_on_step_end": step_callback,
            }
            if reference_image is not None:
                if hasattr(reference_image, "resize"):
                    ref_img_input = reference_image.resize((width, height), Image.Resampling.LANCZOS)
                else:
                    ref_img_input = reference_image
                kwargs["image"] = ref_img_input

            if generator is not None:
                kwargs["generator"] = generator

            output = pipeline(**kwargs)
            image = output.images[0]
            
            t_pipe_end = time.time()
            total_time = round(t_pipe_end - t_start, 2)
            
            prompt_enc_time = round(step_times.get("prompt_encoding_end", t_pipe_start) - t_pipe_start, 2)
            infer_time = round(step_times.get("step_end", t_pipe_end) - step_times.get("step_start", t_pipe_start), 2)
            decode_time = round(t_pipe_end - step_times.get("step_end", t_pipe_end), 2)
            if infer_time < 0.0:
                infer_time = round(t_pipe_end - t_pipe_start, 2)

            vram_after = torch.cuda.memory_allocated() / (1024 ** 2) if (torch and torch.cuda.is_available()) else 0.0
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if (torch and torch.cuda.is_available()) else 0.0

            self.logger.info("After inference VRAM: %.2f MB", vram_after)
            self.logger.info("Peak VRAM: %.2f MB", peak_vram)
            self.logger.info(
                "\n=== FLUX Performance Report ===\n"
                "Model Load: %.2f sec\n"
                "Prompt Encoding: %.2f sec\n"
                "Inference: %.2f sec\n"
                "Decode: %.2f sec\n"
                "Total: %.2f sec\n"
                "===============================",
                model_load_time,
                prompt_enc_time,
                infer_time,
                decode_time,
                total_time,
            )

            self.last_execution_stats = {
                "was_fallback_used": False,
                "was_real_flux_used": True,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after,
                "peak_vram_mb": peak_vram,
                "generation_time_s": total_time,
                "model_load_time_s": model_load_time,
                "prompt_encoding_time_s": prompt_enc_time,
                "inference_time_s": infer_time,
                "vae_decode_time_s": decode_time,
                "num_inference_steps": num_inference_steps,
            }

            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

            return image
        except Exception as exc:
            self.logger.exception("FLUX inference error: %s", exc)
            if not self.allow_fallback:
                raise RuntimeError(f"Real FLUX model inference failed: {exc}") from exc
            
            self.last_execution_stats = {
                "was_fallback_used": True,
                "was_real_flux_used": False,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_before,
                "peak_vram_mb": vram_before,
                "generation_time_s": round(time.time() - t_start, 2),
            }
            return self._generate_synthetic_preview(width, height, prompt)

    def _generate_synthetic_preview(self, width: int, height: int, prompt: str) -> Image.Image:
        """Generate a neutral studio garment preview canvas for dry-run testing."""
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        margin_x, margin_y = int(width * 0.25), int(height * 0.2)
        garment_box = [margin_x, margin_y, width - margin_x, height - margin_y]
        
        draw.rectangle(garment_box, fill=(240, 240, 245), outline=(200, 200, 210), width=3)
        draw.text((margin_x + 20, margin_y + 40), "FabricVision-AI", fill=(80, 80, 90))
        draw.text((margin_x + 20, margin_y + 80), "FLUX Garment Output", fill=(100, 100, 110))
        
        return img
