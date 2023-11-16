import argparse
import os
from pathlib import Path

import netspresso
import netspresso_trainer
import gradio as gr

from netspresso_trainer.models import SUPPORTING_MODEL_LIST, SUPPORTING_TASK_LIST

from func.augmentation import summary_transform, get_augmented_images
from func.scheduler import get_lr_dataframe_from_config
from func.pynetspresso import NetsPressoSession, login_with_session, compress_with_session

__version__netspresso_trainer = netspresso_trainer.__version__
__version__netspresso = netspresso.__version__


# TODO: directly import from netspresso_trainer.models
SUPPORTING_TASK_LIST = ['classification', 'segmentation']

PATH_AUG_DOCS = os.getenv(
    "PATH_AUG_DOCS", default="docs/description_augmentation.md")
PATH_AUG_EXAMPLE_CONFIG = os.getenv(
    "PATH_AUG_EXAMPLE_CONFIG", default="config/augmentation_template_common.yaml")
PATH_SCHEDULER_DOCS = os.getenv(
    "PATH_SCHEDULER_DOCS", default="docs/description_lr_scheduler.md")
PATH_SCHEDULER_EXAMPLE_CONFIG = os.getenv(
    "PATH_SCHEDULER_EXAMPLE_CONFIG", default="config/training_template_common.yaml")
PATH_EXAMPLE_IMAGE = os.getenv(
    "PATH_EXAMPLE_IMAGE", default="assets/kyunghwan_cat.jpg")


def parse_args():

    parser = argparse.ArgumentParser(
        description="GUI for NetsPresso Trainer")

    # -------- User arguments ----------------------------------------

    parser.add_argument(
        '--image', type=Path, default=PATH_EXAMPLE_IMAGE,
        help="Path for an example image")

    parser.add_argument(
        '--local', action='store_true',
        help="Whether to run in local environment or not")

    parser.add_argument(
        '--port', type=int, default=50003,
        help="Service port (only applicable when running on local server)")

    args, _ = parser.parse_known_args()

    return args



def launch_gradio(args):
    with gr.Blocks(theme='nota-ai/theme', title="NetsPresso Trainer") as demo:
        gr.Markdown("\n\n# <center>Welcome to NetsPresso Trainer!</center>\n\n")
        gr.Markdown(
            "<center>Package version: "
            f"<code>netspresso-trainer=={__version__netspresso_trainer}</code> "
            f"<code>netspresso=={__version__netspresso}</code></center>"
        )

        with gr.Tab("Train"):
            gr.Markdown("\n\n### <center>TBD</center>\n\n")

        with gr.Tab("Augmentation"):
            gr.Markdown(Path(PATH_AUG_DOCS).read_text())

            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    task_choices = gr.Radio(
                        label="Task: ", value='classification', choices=SUPPORTING_TASK_LIST)
                with gr.Column(scale=1):
                    phase_choices = gr.Radio(
                        label="Phase: ", value='train', choices=['train', 'valid'])
            model_choices = gr.Radio(
                label="Model: ", value='resnet50', choices=SUPPORTING_MODEL_LIST)
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    config_input = gr.Code(label="Augmentation configuration",
                                           value=Path(PATH_AUG_EXAMPLE_CONFIG).read_text(),
                                           language='yaml', lines=30)
                with gr.Column(scale=2):
                    transform_repr_output = gr.Code(
                        label="Data transform", lines=10)
                    transform_button = gr.Button(
                        value="Compose transform", variant='primary')
                    test_image = gr.Image(
                        value=str(args.image), label="Test image", type='pil')
            run_button = gr.Button(
                value="Get augmented samples", variant='primary')
            augmented_images = gr.Gallery(label="Results", columns=5)

            transform_compose_inputs = [phase_choices,
                                        task_choices, model_choices, config_input]
            run_inputs = transform_compose_inputs + [test_image]
            transform_button.click(
                summary_transform, inputs=transform_compose_inputs, outputs=transform_repr_output)
            run_button.click(get_augmented_images,
                             inputs=run_inputs, outputs=augmented_images)

        with gr.Tab("Scheduler"):
            gr.Markdown(Path(PATH_SCHEDULER_DOCS).read_text())
            with gr.Row(equal_height=True):
                with gr.Column():
                    config_input = gr.Code(label="Training configuration",
                                           value=Path(PATH_SCHEDULER_EXAMPLE_CONFIG).read_text(),
                                           language='yaml', lines=30)
                    run_button = gr.Button(value="Run", variant='primary')
                with gr.Column():
                    plot_output = gr.LinePlot()

            run_button.click(get_lr_dataframe_from_config,
                             inputs=config_input, outputs=plot_output)

        with gr.Tab("PyNetsPresso"):

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

            login_button.click(
                login_with_session, inputs=[session, email_input, password_input], outputs=[session]
            )
            password_input.submit(
                login_with_session, inputs=[session, email_input, password_input], outputs=[session]
            )

            compress_button.click(
                compress_with_session,
                inputs=[session, model_name, model_task, model_path,
                        compress_input_batch_size, compress_input_channels, compress_input_height, compress_input_width,
                        compression_ratio, compressed_model_path],
                outputs=[session, result_compressed_model_path])

    demo.queue()

    if args.local:
        demo.launch(server_name="0.0.0.0", server_port=args.port)
    else:
        demo.launch()


if __name__ == "__main__":
    args = parse_args()
    launch_gradio(args)
