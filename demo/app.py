import argparse
import os
from pathlib import Path
from urllib import request

import gradio as gr
import netspresso
import netspresso_trainer
from tab.experiments import tab_experiments
from tab.home.main import tab_home
from tab.compressor import tab_compressor

__version__netspresso_trainer = netspresso_trainer.__version__
__version__netspresso = netspresso.__version__


PATH_EXAMPLE_IMAGE = os.getenv("PATH_EXAMPLE_IMAGE", default="assets/kyunghwan_cat.jpg")


def change_tab_to_pynetspresso():
    return gr.Tabs.update(selected='compressor')


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


def get_snippet_from_url(url: str) -> str:
    response = request.urlopen(url)
    data = response.read().decode()
    return data


def launch_gradio(args):
    with gr.Blocks(
        theme='nota-ai/theme-v4',
        title="NetsPresso Trainer",
        css=get_snippet_from_url("https://huggingface.co/spaces/nota-ai/theme-v4/raw/main/main.css")
    ) as demo:
        gr.Markdown("\n\n# <center>Welcome to NetsPresso Trainer!</center>\n\n")
        gr.Markdown(
            "<center>Package version: "
            f"<code>netspresso-trainer=={__version__netspresso_trainer}</code> "
            f"<code>netspresso=={__version__netspresso}</code></center>"
        )

        with gr.Tabs() as tabs:

            with gr.Tab("Home", id='home'):
                tab_home(args)

            with gr.Tab("Experiments", id='experiments'):
                experiment_df, experiment_selected, experiment_button_launcher, experiment_button_compressor = tab_experiments(
                    args)

            with gr.Tab("Compressor", id='compressor'):
                session, email_input, password_input, model_name, model_path, \
                    compress_input_batch_size, compress_input_channels, compress_input_height, compress_input_width, \
                    compression_ratio, compressed_model_dir, result_compressed_model_path, \
                    login_button, compress_button = \
                    tab_compressor(args)

            experiment_button_compressor.click(
                fn=experiment_df.find_compression_info_with_id, inputs=[experiment_selected],
                outputs=[model_name, model_path, compress_input_batch_size, compress_input_channels,
                         compress_input_height, compress_input_width]).success(
                fn=change_tab_to_pynetspresso, inputs=None, outputs=[tabs])

    demo.queue()

    if args.local:
        demo.launch(server_name="0.0.0.0", server_port=args.port)
    else:
        demo.launch()


if __name__ == "__main__":
    args = parse_args()
    launch_gradio(args)
