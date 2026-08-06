import gradio as gr

def get_fabriplay_theme() -> gr.Theme:
    """Returns a premium, pastel-based Gradio theme for Fabriplay."""
    return gr.themes.Default(
        primary_hue="rose",
        secondary_hue="stone",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Fira Code"), "ui-monospace", "Consolas", "monospace"],
    ).set(
        # Colors - Pastel / Soft
        body_background_fill="*neutral_50",
        body_background_fill_dark="*neutral_950",
        body_text_color="*neutral_800",
        body_text_color_subdued="*neutral_500",
        
        # Primary Accent (Muted Rose / Pastel Pink)
        color_accent="*primary_500",
        color_accent_soft="*primary_50",
        
        # Borders and Backgrounds
        background_fill_primary="white",
        background_fill_secondary="*neutral_50",
        border_color_primary="*neutral_200",
        
        # Components
        block_background_fill="white",
        block_border_width="1px",
        block_border_color="*neutral_200",
        block_radius="*radius_xl",
        
        # Inputs
        input_background_fill="*neutral_50",
        input_border_color="*neutral_200",
        input_radius="*radius_md",
        
        # Buttons
        button_primary_background_fill="*primary_500",
        button_primary_text_color="white",
        button_secondary_background_fill="white",
        button_secondary_border_color="*neutral_200",
        button_secondary_text_color="*neutral_700",
    )
