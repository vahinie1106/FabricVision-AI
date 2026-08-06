import logging
from pathlib import Path
from typing import Any, Tuple
from PIL import Image
import gradio as gr

from src.features.virtual_tryon.models import GarmentConditioningInput, PersonConditioningInput
from src.features.virtual_tryon.tryon_pipeline import TryOnConfig, VirtualTryOnPipeline
from src.common.models.model_manager import ModelManager

logger = logging.getLogger("fabricvision.ui.tryon")
workspace_root = Path(__file__).resolve().parents[4]

def run_standalone_tryon(
    model_mgr: ModelManager,
    person_image: Any,
    garment_image: Any,
) -> Tuple[Any, str]:
    if person_image is None or garment_image is None:
        return None, "Upload both person image and garment image first."

    try:
        temp_person_path = workspace_root / "outputs" / "temp" / "temp_person_standalone.png"
        temp_garment_path = workspace_root / "outputs" / "temp" / "temp_garment_standalone.png"
        temp_person_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(person_image, Image.Image):
            person_img_obj = person_image.convert("RGB")
        else:
            person_img_obj = Image.fromarray(person_image).convert("RGB")
        person_img_obj.save(temp_person_path, format="PNG")

        if isinstance(garment_image, Image.Image):
            garment_img_obj = garment_image.convert("RGB")
        else:
            garment_img_obj = Image.fromarray(garment_image).convert("RGB")
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
        # Defaulting garment_type to a safe default since UI doesn't provide it anymore
        garment_in = GarmentConditioningInput(
            garment_image=garment_img_obj,
            garment_type="kurti", 
        )

        res = tryon_pipeline.run(
            person_input=person_in,
            garment_input=garment_in,
            output_filename="gradio_standalone_tryon",
        )

        tryon_img = Image.open(res.image_path)
        model_mgr.clear_vram()
        return tryon_img, "Virtual try-on completed successfully."
    except Exception as exc:
        logger.exception("Virtual try-on error: %s", exc)
        dummy = Image.new("RGB", (512, 512), color=(240, 230, 220))
        return dummy, f"Error: {exc}"


def create_tryon_ui(model_mgr: ModelManager):
    with gr.Blocks() as ui:
        with gr.Column(elem_classes=["gradio-box"]):
            gr.Markdown("## Step 1: Upload Images")
            with gr.Row():
                garment_upload = gr.Image(label="Original Garment", type="pil", sources=["upload"], height=320, elem_classes=["image-preview"])
                person_upload = gr.Image(label="Target Person", type="pil", sources=["upload"], height=320, elem_classes=["image-preview"])
            
            with gr.Row():
                tryon_btn = gr.Button("Execute Virtual Try-On", variant="primary", elem_classes=["primary"])
                
        with gr.Column(elem_classes=["gradio-box"]):
            gr.Markdown("## Step 2: Final Result")
            with gr.Row():
                tryon_result_preview = gr.Image(label="Virtual Try-On Result", type="pil", height=400, interactive=False, elem_classes=["image-preview"])
            
            tryon_status_tb = gr.Textbox(label="Status", interactive=False)

        tryon_btn.click(
            fn=lambda person, garment: run_standalone_tryon(model_mgr, person, garment),
            inputs=[person_upload, garment_upload],
            outputs=[tryon_result_preview, tryon_status_tb],
        )

    return ui
