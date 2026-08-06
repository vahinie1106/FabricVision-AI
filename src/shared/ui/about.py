import gradio as gr

def create_about_ui():
    with gr.Blocks(elem_classes=["gradio-box"]) as ui:
        gr.Markdown("# About FabricVision-AI")
        gr.Markdown("## A Professional AI Fashion Design Platform")
        gr.Markdown(
            "FabricVision-AI is an advanced AI fashion platform designed for **Fabriplay**. "
            "It empowers designers and consumers to create, visualize, and analyze fashion using state-of-the-art AI models."
        )
        
        gr.Markdown("### Platform Features")
        gr.Markdown(
            "- **Custom Garment Generation**: Create photorealistic garments from basic fabric textures.\n"
            "- **Virtual Try-On**: See how garments fit on real people with high-fidelity virtual try-on technology.\n"
            "- **Semantic Analysis**: (Coming Soon) Automatically understand and categorize fashion imagery."
        )
        
        gr.Markdown("### AI Technologies")
        gr.Markdown(
            "- **FLUX Kontext**: Used for high-quality, image-conditioned garment synthesis.\n"
            "- **CatVTON**: Used for accurate, persona-based virtual try-on.\n"
            "- **Qwen2.5-VL**: Used for deep semantic understanding and metadata extraction."
        )
        
        gr.Markdown("### Future Roadmap")
        gr.Markdown(
            "- Full Semantic Analysis Integration\n"
            "- Pattern configuration support\n"
            "- Batch processing and export tools\n"
            "- Enhanced customization options"
        )
        
    return ui
