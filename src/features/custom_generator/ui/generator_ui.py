import json
import logging
from pathlib import Path
from typing import Any, Tuple
from PIL import Image
import gradio as gr

from src.features.custom_generator.pipeline.garment_generation_pipeline import GarmentGenerationConfig, GarmentGenerationPipeline
from src.features.virtual_tryon.models import GarmentConditioningInput, PersonConditioningInput
from src.features.virtual_tryon.tryon_pipeline import TryOnConfig, VirtualTryOnPipeline
from src.common.models.model_manager import ModelManager
from src.common.utils.taxonomy import load_fashion_taxonomy

logger = logging.getLogger("fabricvision.ui.generator")
workspace_root = Path(__file__).resolve().parents[4]

def generate_flux_garment(
    model_mgr: ModelManager,
    fabric_image: Any,
    gender: str,
    garment_type: str,
    color_palette: str,
    fabric: str,
    neckline: str,
    sleeve_length: str,
    style: str,
    occasion: str,
    fit: str,
    generation_mode: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[Any, str, str]:
    try:
        progress(0.1, desc="Stage 1: Loading FLUX Kontext Model")
        model_mgr.switch_to("flux")

        progress(0.25, desc="Stage 2: Processing Reference Image")
        ref_img_obj = None
        if fabric_image is not None:
            if isinstance(fabric_image, Image.Image):
                ref_img_obj = fabric_image.convert("RGB")
            else:
                ref_img_obj = Image.fromarray(fabric_image).convert("RGB")

        fabric_metadata = {
            "material": fabric.lower(),
            "dominant_colors": [color_palette.lower().replace(" ", "_")],
            "texture": "smooth",
            "style": style.lower(),
            "occasion": occasion.lower(),
            "fit": fit.lower(),
        }

        progress(0.55, desc="Stage 4: Building Fashion Instruction Prompt")
        user_customization = {
            "gender": "women" if gender.lower() == "female" else "men",
            "garment_type": garment_type.lower().replace(" ", "_"),
            "material": fabric.lower().replace(" ", "_"),
            "color": color_palette.lower().replace(" ", "_"),
            "neckline": neckline.lower().replace(" ", "_"),
            "sleeve": sleeve_length.lower().replace(" ", "_"),
            "style": style.lower().replace(" ", "_"),
            "occasion": occasion.lower().replace(" ", "_"),
            "fit": fit.lower().replace(" ", "_"),
        }

        progress(0.7, desc="Stage 5: Running FLUX.1-Kontext Generation")
        flux_config = GarmentGenerationConfig(
            config_dir=str(workspace_root / "configs"),
            config_path=str(workspace_root / "configs" / "custom_generator" / "flux_config.yaml"),
            output_root=str(workspace_root / "outputs" / "garment_generation"),
            experiments_root=str(workspace_root / "experiments"),
            generation_mode=generation_mode,
            allow_fallback=True,
        )

        pipeline = GarmentGenerationPipeline(
            config=flux_config,
            model_loader=model_mgr.flux_manager.loader,
        )

        result = pipeline.run(
            fabric_metadata=fabric_metadata,
            user_customization=user_customization,
            output_filename="gradio_flux_garment",
            reference_image=ref_img_obj,
        )

        progress(0.95, desc="Stage 6: Saving Output")
        gen_img = Image.open(result["image_path"])
        meta_str = json.dumps(result["metadata"], indent=2)

        stats = getattr(pipeline.inference_engine, "last_execution_stats", {})
        gen_time = stats.get("generation_time_s", 0.0)

        execution_card = (
            f"Generation Mode: {generation_mode}\n"
            f"Generation Time: {gen_time:.2f} seconds\n"
            "Status: Success"
        )
        return gen_img, execution_card, meta_str
    except Exception as exc:
        logger.exception("FLUX garment generation error: %s", exc)
        dummy = Image.new("RGB", (512, 512), color=(255, 255, 255))
        return dummy, f"Error occurred:\n{exc}", "{}"

def run_virtual_tryon(
    model_mgr: ModelManager,
    person_image: Any,
    generated_garment_image: Any,
    garment_type: str,
) -> Tuple[Any, str]:
    if person_image is None or generated_garment_image is None:
        return None, "Please complete garment generation and upload a person image first."

    try:
        temp_person_path = workspace_root / "outputs" / "temp" / "temp_person.png"
        temp_garment_path = workspace_root / "outputs" / "temp" / "temp_generated_garment.png"
        temp_person_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(person_image, Image.Image):
            person_img_obj = person_image.convert("RGB")
        else:
            person_img_obj = Image.fromarray(person_image).convert("RGB")
        person_img_obj.save(temp_person_path, format="PNG")

        if isinstance(generated_garment_image, Image.Image):
            garment_img_obj = generated_garment_image.convert("RGB")
        else:
            garment_img_obj = Image.fromarray(generated_garment_image).convert("RGB")
        garment_img_obj.save(temp_garment_path, format="PNG")

        model_mgr.switch_to("catvton")

        tryon_config = TryOnConfig(
            config_dir=str(workspace_root / "configs"),
            output_root=str(workspace_root / "outputs" / "virtual_tryon"),
            experiments_root=str(workspace_root / "experiments"),
            height=512,
            width=512,
            allow_fallback=True,
        )
        tryon_pipeline = VirtualTryOnPipeline(
            config=tryon_config,
            model_loader=model_mgr.catvton_manager.loader,
        )

        person_in = PersonConditioningInput(person_image=person_img_obj)
        garment_in = GarmentConditioningInput(
            garment_image=garment_img_obj,
            garment_type=garment_type.lower().replace(" ", "_") if garment_type else "kurti",
        )

        res = tryon_pipeline.run(
            person_input=person_in,
            garment_input=garment_in,
            output_filename="gradio_final_tryon",
        )

        tryon_img = Image.open(res.image_path)
        model_mgr.clear_vram()
        return tryon_img, "Virtual Try-On Completed Successfully"
    except Exception as exc:
        logger.exception("Virtual try-on error: %s", exc)
        dummy = Image.new("RGB", (512, 512), color=(240, 230, 220))
        return dummy, f"Error: {exc}"


def create_generator_ui(model_mgr: ModelManager):
    taxonomy = load_fashion_taxonomy()

    with gr.Blocks() as ui:
        
        with gr.Column(elem_classes=["gradio-box"]):
            gr.Markdown("## Step 1: Upload Fabric Design")
            with gr.Row():
                fabric_upload = gr.Image(label="Fabric / Texture Image", type="pil", sources=["upload"], height=280, elem_classes=["image-preview"])
        
        with gr.Column(elem_classes=["gradio-box"]):
            gr.Markdown("## Step 2: Customize Garment")
            with gr.Row():
                gender_dd = gr.Dropdown(label="Gender", choices=taxonomy.get("genders", ["Female", "Male", "Unisex"]), value="Female")
                garment_type_dd = gr.Dropdown(label="Garment Type", choices=taxonomy.get("garment_types", []), value="Kurta")
                color_palette_dd = gr.Dropdown(label="Color Palette (Optional)", choices=taxonomy.get("color_palettes", []), value="Navy Blue")
                fabric_dd = gr.Dropdown(label="Fabric Material", choices=taxonomy.get("fabrics", []), value="Cotton")

            with gr.Row():
                neckline_dd = gr.Dropdown(label="Neckline", choices=taxonomy.get("necklines", []), value="Round Neck")
                sleeve_length_dd = gr.Dropdown(label="Sleeve Length", choices=taxonomy.get("sleeve_lengths", []), value="Three Quarter Sleeve")
                style_dd = gr.Dropdown(label="Style", choices=taxonomy.get("styles", []), value="Casual")
                occasion_dd = gr.Dropdown(label="Occasion", choices=taxonomy.get("occasions", []), value="Casual")
                fit_dd = gr.Dropdown(label="Fit", choices=taxonomy.get("fits", []), value="Regular Fit")

        with gr.Column(elem_classes=["gradio-box"]):
            gr.Markdown("## Step 3: Generate Garment")
            with gr.Row():
                generation_mode_dd = gr.Dropdown(
                    label="Generation Quality",
                    choices=["Preview", "Standard", "Production"],
                    value="Standard",
                )
                generate_flux_btn = gr.Button("Generate Garment", variant="primary", elem_classes=["primary"])
                
            with gr.Row():
                generated_garment_preview = gr.Image(label="Generated Garment", type="pil", height=400, interactive=False, elem_classes=["image-preview"])
            
            with gr.Row():
                execution_card_tb = gr.Textbox(label="Status", interactive=False)
                metadata_str = gr.Textbox(visible=False)

        with gr.Column(elem_classes=["gradio-box"]):
            gr.Markdown("## Step 4: Virtual Try-On")
            with gr.Row():
                person_upload = gr.Image(label="Upload Target Person", type="pil", sources=["upload"], height=320, elem_classes=["image-preview"])
                
            with gr.Row():
                tryon_btn = gr.Button("Try It On", variant="primary", elem_classes=["primary"])
                
            with gr.Row():
                tryon_result_preview = gr.Image(label="Final Result", type="pil", height=400, interactive=False, elem_classes=["image-preview"])
            
            with gr.Row():
                tryon_status_tb = gr.Textbox(label="Status", interactive=False)

        generate_flux_btn.click(
            fn=lambda *args: generate_flux_garment(model_mgr, *args),
            inputs=[
                fabric_upload, gender_dd, garment_type_dd, color_palette_dd,
                fabric_dd, neckline_dd, sleeve_length_dd,
                style_dd, occasion_dd, fit_dd, generation_mode_dd,
            ],
            outputs=[generated_garment_preview, execution_card_tb, metadata_str],
        )

        tryon_btn.click(
            fn=lambda person, garment, gtype: run_virtual_tryon(model_mgr, person, garment, gtype),
            inputs=[person_upload, generated_garment_preview, garment_type_dd],
            outputs=[tryon_result_preview, tryon_status_tb],
        )

    return ui
