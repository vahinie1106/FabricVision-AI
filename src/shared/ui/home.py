import gradio as gr

def create_home_ui():
    with gr.Blocks() as ui:
        with gr.Column(elem_classes=["hero-section"]):
            gr.Markdown("# FabricVision-AI")
            gr.Markdown("## Professional AI Fashion Design Platform")
            gr.Markdown("Transform fabrics into professional garment designs and visualize them with AI-powered virtual try-on technology.")
            
        with gr.Row():
            with gr.Column(elem_classes=["gradio-box"]):
                gr.Markdown("### 🎨 Create Custom Garment")
                gr.Markdown("Generate a completely new garment by uploading a fabric design and customizing the style. Powered by FLUX Kontext.")
                gr.Markdown("*Click the 'Create Custom Garment' tab above to start.*")
                
            with gr.Column(elem_classes=["gradio-box"]):
                gr.Markdown("### 👕 Virtual Try-On")
                gr.Markdown("Upload an existing garment and visualize how it looks on a person. Powered by CatVTON.")
                gr.Markdown("*Click the 'Virtual Try-On' tab above to start.*")
                
            with gr.Column(elem_classes=["gradio-box"]):
                gr.Markdown("### 🧠 Semantic Analysis")
                gr.Markdown("Extract detailed semantic metadata from any garment image. Powered by Qwen2.5-VL.")
                gr.Markdown("*Coming Soon.*")
                
    return ui
