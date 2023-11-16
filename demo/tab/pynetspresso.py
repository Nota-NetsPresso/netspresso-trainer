import os
import gradio as gr
from func.pynetspresso import NetsPressoSession, login_with_session, compress_with_session

# TODO: directly import from netspresso_trainer.models
SUPPORTING_TASK_LIST = ['classification', 'segmentation']

def tab_pynetspresso(args):

    session = gr.State(NetsPressoSession())

    gr.Markdown(
        "\n\n### <center>NOTE: This feature needs an internet connection</center>\n\n")

    with gr.Row(equal_height=True):
        with gr.Column():
            gr.Markdown(
                "If you have not signed up at NetsPresso, please sign up frist: [netspresso.ai](https://netspresso.ai/signup)")
            email_input = gr.Textbox(
                label="Email", type="email"
            )
            password_input = gr.Textbox(
                label="Password", type="password"
            )
            with gr.Row(equal_height=True):
                gr.ClearButton([email_input, password_input])
                login_button = gr.Button(
                    value="Login", variant='primary'
                )

        with gr.Column():
            with gr.Group():
                with gr.Row():
                    model_name = gr.Textbox(label="Model name")
                    model_task = gr.Dropdown(label="Task", value='classification', multiselect=False,
                                                choices=SUPPORTING_TASK_LIST)
                model_path = gr.Textbox(label="Model path")
                with gr.Row():
                    compress_input_batch_size = gr.Number(label="Batch size", value=1, minimum=1, maximum=1)
                    compress_input_channels = gr.Number(label="Channels", value=3, minimum=1)
                    compress_input_height = gr.Number(label="Height", value=256, minimum=32, maximum=512)
                    compress_input_width = gr.Number(label="Width", value=256, minimum=32, maximum=512)
                compression_ratio = gr.Slider(
                    minimum=0, maximum=1, value=0.5, step=0.1,
                    info="The removal ratio of the filters (e.g. 0.2 removes 20% of the filters in the model)"
                )
                compressed_model_path = gr.Textbox(label="Output model path")

            compress_button = gr.Button(
                value="Compress", variant='primary'
            )
            result_compressed_model_path = gr.Textbox(label="Result")
            
    return session, email_input, password_input, model_name, model_task, model_path, \
        compress_input_batch_size, compress_input_channels, compress_input_height, compress_input_width, \
        compression_ratio, compressed_model_path, result_compressed_model_path, \
        login_button, compress_button