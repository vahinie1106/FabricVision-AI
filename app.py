print("FabricVision AI Started Successfully!")

import cv2
import numpy as np

print("OpenCV Version:", cv2.__version__)
print("NumPy Version:", np.__version__)

import gradio as gr

def hello(name):
    return f"Hello {name}!"

demo = gr.Interface(
    fn=hello,
    inputs="text",
    outputs="text"
)

demo.launch()

