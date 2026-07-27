import gradio as gr

from src.ui.main_ui import create_ui


def test_create_ui_returns_gradio_blocks():
    ui = create_ui()

    assert isinstance(ui, gr.Blocks)
    assert ui is not None
