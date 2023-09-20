import sys
import modules.scripts as scripts
import gradio as gr

from apply_block_weight import do


class Script(scripts.Script):
    def title(self):
        return "Apply LBW"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        with gr.Tab("Settings"):
            input_ = gr.Textbox(label="Path to input safetensors")
            output = gr.Textbox(label="Path to output safetensors")
            ratios = gr.Textbox(label="The LBW numeric array")
            apply = gr.Button(value="Apply")
            args = [input_, output, ratios]
            apply.click(do, inputs=args)
            return args

    def run(self, p, input_, output, ratios):
        return
