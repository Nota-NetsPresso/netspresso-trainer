import argparse
import os
from pathlib import Path

import netspresso
import netspresso_trainer
import gradio as gr

from tab.augmentation import tab_augmentation
from tab.scheduler import tab_scheduler
from tab.pynetspresso import tab_pynetspresso

from func.augmentation import summary_transform, get_augmented_images
from func.scheduler import get_lr_dataframe_from_config
from func.pynetspresso import login_with_session, compress_with_session

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

        with gr.Tab("Train"):
            gr.Markdown("\n\n### <center>TBD</center>\n\n")

        with gr.Tab("Augmentation"):
            task_choices, phase_choices, model_choices, config_input, transform_repr_output, test_image, augmented_images, transform_button, run_button = tab_augmentation(args)
            
            
            transform_compose_inputs = [phase_choices,
                                        task_choices, model_choices, config_input]
            run_inputs = transform_compose_inputs + [test_image]
            transform_button.click(
                summary_transform, inputs=transform_compose_inputs, outputs=transform_repr_output)
            run_button.click(get_augmented_images,
                             inputs=run_inputs, outputs=augmented_images)

        with gr.Tab("Scheduler"):
            config_input, plot_output, run_button = tab_scheduler(args)

            run_button.click(get_lr_dataframe_from_config,
                             inputs=config_input, outputs=plot_output)

        with gr.Tab("PyNetsPresso"):
            session, email_input, password_input, model_name, model_task, model_path, \
            compress_input_batch_size, compress_input_channels, compress_input_height, compress_input_width, \
            compression_ratio, compressed_model_path, result_compressed_model_path, \
            login_button, compress_button = \
                tab_pynetspresso(args)

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
