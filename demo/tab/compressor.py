import os
from pathlib import Path

import gradio as gr
from func.pynetspresso import NetsPressoSession, compress_with_session, login_with_session

# TODO: directly import from netspresso_trainer.models
SUPPORTING_TASK_LIST = ['classification', 'segmentation']

PATH_PYNETSPRESSO_DOCS = os.getenv(
    "PATH_PYNETSPRESSO_DOCS", default="docs/description_pynetspresso.md")


def tab_compressor(args):

    session = gr.State(NetsPressoSession())

    gr.Markdown(Path(PATH_PYNETSPRESSO_DOCS).read_text())

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
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

        with gr.Column(scale=2):
            with gr.Group():
                with gr.Row():
                    model_name = gr.Textbox(
                        label="Model name",
                        info="Leave empty to use same as the model filename."
                    )
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
                compressed_model_dir = gr.Textbox(
                    label="Output model directory",
                    info="Leave empty to use the same model directory. The compressed model and its parent directory are named with the postfix (_compressed)."
                )

            compress_button = gr.Button(
                value="Compress", variant='primary'
            )
            result_compressed_model_path = gr.Textbox(
                label="Result",
                show_copy_button=True,
                interactive=False
            )

    login_button.click(
        login_with_session, inputs=[session, email_input, password_input], outputs=[session]
    )
    password_input.submit(
        login_with_session, inputs=[session, email_input, password_input], outputs=[session]
    )

    compress_button.click(
        compress_with_session,
        inputs=[session, model_name, model_path,
                compress_input_batch_size, compress_input_channels, compress_input_height, compress_input_width,
                compression_ratio, compressed_model_dir],
        outputs=[session, result_compressed_model_path])

    return session, email_input, password_input, model_name, model_path, \
        compress_input_batch_size, compress_input_channels, compress_input_height, compress_input_width, \
        compression_ratio, compressed_model_dir, result_compressed_model_path, \
        login_button, compress_button
