"""Entry point for launching the FabricVision-AI Gradio UI."""

from src.ui.main_ui import create_ui


def main() -> None:
    """Create and launch the FabricVision-AI interface."""
    demo = create_ui()
    demo.launch()


if __name__ == "__main__":
    main()

