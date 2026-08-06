"""FabricVision-AI Main Entry point.
Assembles the modular features into a premium tabbed interface.
"""

from __future__ import annotations

import gradio as gr

from src.common.models.model_manager import ModelManager
from src.features.custom_generator.ui.generator_ui import create_generator_ui
from src.features.virtual_tryon.ui.tryon_ui import create_tryon_ui
from src.features.semantic_analysis.ui.semantic_ui import create_semantic_ui
from src.shared.ui.home import create_home_ui
from src.shared.ui.about import create_about_ui
from src.shared.ui.theme import get_fabriplay_theme

# Global shared model manager
_model_manager = ModelManager()

def create_ui() -> gr.Blocks:
    """Create and return the master FabricVision-AI Gradio Blocks interface."""
    
    css_path = "src/shared/assets/style.css"
    
    with gr.Blocks(title="Fabriplay | FabricVision-AI", theme=get_fabriplay_theme(), css=css_path) as ui:
        
        with gr.Column(elem_id="app-container"):
            
            with gr.Tabs(elem_classes=["tab-nav"]):
                
                with gr.TabItem("Home"):
                    create_home_ui()
                    
                with gr.TabItem("Create Custom Garment"):
                    create_generator_ui(_model_manager)
                    
                with gr.TabItem("Virtual Try-On"):
                    create_tryon_ui(_model_manager)
                    
                with gr.TabItem("Semantic Analysis"):
                    create_semantic_ui()
                    
                with gr.TabItem("About"):
                    create_about_ui()
                    
    return ui

if __name__ == "__main__":
    app = create_ui()
    app.launch()
