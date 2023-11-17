import argparse
import os
from pathlib import Path

import netspresso
import netspresso_trainer
import gradio as gr

from tab.augmentation import tab_augmentation
from tab.scheduler import tab_scheduler
from tab.experiments import tab_experiments
from tab.pynetspresso import tab_pynetspresso

__version__netspresso_trainer = netspresso_trainer.__version__
__version__netspresso = netspresso.__version__


PATH_EXAMPLE_IMAGE = os.getenv("PATH_EXAMPLE_IMAGE", default="assets/kyunghwan_cat.jpg")


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

        with gr.Tab("Experiments"):
            tab_experiments(args)

        with gr.Tab("Home"):
            with gr.Tab("Train"):
                gr.Markdown("\n\n### <center>TBD</center>\n\n")

            with gr.Tab("Augmentation"):
                task_choices, phase_choices, model_choices, config_input, \
                    transform_repr_output, test_image, augmented_images, transform_button, run_button = \
                    tab_augmentation(args)

            with gr.Tab("Scheduler"):
                config_input, plot_output, run_button = tab_scheduler(args)

        with gr.Tab("PyNetsPresso"):
            session, email_input, password_input, model_name, model_task, model_path, \
                compress_input_batch_size, compress_input_channels, compress_input_height, compress_input_width, \
                compression_ratio, compressed_model_path, result_compressed_model_path, \
                login_button, compress_button = \
                tab_pynetspresso(args)

    demo.queue()

    if args.local:
        demo.launch(server_name="0.0.0.0", server_port=args.port)
    else:
        demo.launch()


if __name__ == "__main__":
    args = parse_args()
    launch_gradio(args)
