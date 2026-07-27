"""FabricVision-AI UI foundation built with Gradio Blocks.

This module provides the first user interface for the project without
integrating any AI inference, datasets, or image generation logic.
"""

from __future__ import annotations

from typing import Any

import gradio as gr


def _build_gender_options() -> list[str]:
    """Return gender choices for the form."""
    return ["Male", "Female"]


def _build_garment_options(gender: str | None) -> list[str]:
    """Return garment choices based on the selected gender."""
    if gender == "Female":
        return ["Top", "Shirt", "Kurti", "T-Shirt", "Dress"]
    return ["T-Shirt", "Formal Shirt", "Polo Shirt", "Hoodie"]


def _build_material_options() -> list[str]:
    """Return fabric material choices."""
    return [
        "Acrylic",
        "Artificial_fur",
        "Artificial_leather",
        "Blended",
        "Chenille",
        "Corduroy",
        "Cotton",
        "Crepe",
        "Denim",
        "Felt",
        "Fleece",
        "Leather",
        "Linen",
        "Lut",
        "Nylon",
        "Polyester",
        "Satin",
        "Silk",
        "Suede",
        "Terrycloth",
        "Unclassified",
        "Utilities",
        "Velvet",
        "Viscose",
        "Wool",
    ]


def _build_pattern_options() -> list[str]:
    """Return fabric pattern choices."""
    return [
        "Argyle",
        "Batik",
        "Camouflage",
        "Checkered",
        "Dotted",
        "Floral",
        "Leopard",
        "Solid",
        "Striped",
        "Zebra",
        "Zigzag",
    ]


def _build_size_options() -> list[str]:
    """Return dress size choices."""
    return ["XS", "S", "M", "L", "XL", "XXL"]


def _build_color_options() -> list[str]:
    """Return garment color choices."""
    return [
        "Black",
        "White",
        "Blue",
        "Red",
        "Green",
        "Yellow",
        "Brown",
        "Grey",
        "Pink",
        "Orange",
        "Purple",
        "Beige",
        "Navy",
        "Maroon",
        "Olive",
    ]


def generate_try_on(
    person_image: Any,
    fabric_image: Any,
    gender: str | None,
    garment_type: str | None,
    fabric_material: str | None,
    fabric_pattern: str | None,
    dress_size: str | None,
    garment_color: str | None,
) -> tuple[Any, str]:
    """Placeholder function for the future AI integration phase.

    It intentionally does not perform inference. Instead, it returns the
    original placeholder image and a message explaining the next phase.
    """

    placeholder_image = None
    message = "AI integration will be added in the next phase."
    return placeholder_image, message


def create_ui() -> gr.Blocks:
    """Create and return the FabricVision-AI Gradio Blocks UI."""
    with gr.Blocks(title="FabricVision-AI") as ui:
        gr.Markdown("# FabricVision-AI")
        gr.Markdown("### AI-Powered Virtual Try-On System")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.Markdown("## Configuration")

                person_image = gr.Image(
                    label="Upload Person Image",
                    type="numpy",
                    sources=["upload"],
                    height=280,
                )

                fabric_image = gr.Image(
                    label="Upload Fabric Design Image",
                    type="numpy",
                    sources=["upload"],
                    height=280,
                )

                gender_dropdown = gr.Dropdown(
                    label="Gender",
                    choices=_build_gender_options(),
                    value="Male",
                )

                garment_dropdown = gr.Dropdown(
                    label="Garment Type",
                    choices=_build_garment_options("Male"),
                    value="T-Shirt",
                )

                material_dropdown = gr.Dropdown(
                    label="Fabric Material",
                    choices=_build_material_options(),
                    value="Cotton",
                )

                pattern_dropdown = gr.Dropdown(
                    label="Fabric Pattern",
                    choices=_build_pattern_options(),
                    value="Solid",
                )

                size_dropdown = gr.Dropdown(
                    label="Dress Size",
                    choices=_build_size_options(),
                    value="M",
                )

                color_dropdown = gr.Dropdown(
                    label="Garment Color",
                    choices=_build_color_options(),
                    value="Blue",
                )

                generate_button = gr.Button("Generate Try-On", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("## Preview")
                result_image = gr.Image(
                    label="Generated Try-On Result",
                    type="numpy",
                    height=560,
                    interactive=False,
                    show_label=True,
                )
                result_message = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False,
                )

        def update_garment_options(gender: str | None) -> gr.Dropdown:
            """Update garment choices when the gender selection changes."""
            return gr.Dropdown(
                choices=_build_garment_options(gender),
                value=_build_garment_options(gender)[0],
            )

        gender_dropdown.change(
            fn=lambda gender: gr.update(
                choices=_build_garment_options(gender),
                value=_build_garment_options(gender)[0],
            ),
            inputs=[gender_dropdown],
            outputs=[garment_dropdown],
        )

        generate_button.click(
            fn=generate_try_on,
            inputs=[
                person_image,
                fabric_image,
                gender_dropdown,
                garment_dropdown,
                material_dropdown,
                pattern_dropdown,
                size_dropdown,
                color_dropdown,
            ],
            outputs=[result_image, result_message],
        )

    return ui
