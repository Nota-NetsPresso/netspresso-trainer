import os
from pathlib import Path

import gradio as gr
from func.experiments import ExperimentDataFrame


def tab_experiments(args):
    experiment_df = ExperimentDataFrame(experiment_dir="./outputs")
    with gr.Row():
        with gr.Column():
            experiment_summary_plot = gr.ScatterPlot()
        with gr.Column():
            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    with gr.Group():
                        with gr.Row():
                            experiment_select_task = gr.Dropdown(label="Task")
                            experiment_select_model = gr.Dropdown(label="Model")
                            experiment_select_data = gr.Dropdown(label="Data")
                        experiment_threshold_macs = gr.Slider(label="MACs (smaller than)")
                        experiment_threshold_params = gr.Slider(label="Params # (smaller than)")
            with gr.Row():
                experiment_select_clear = gr.Button(value="Clear")
                experiment_select_search = gr.Button(value="Search", variant='primary')
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            experiment_selected_experiment = gr.Textbox(label="Selected checkpoint")
        with gr.Column(scale=1):
            experiment_goto_launcher = gr.Button(value="Benchmark", variant='primary')
            experiment_goto_compressor = gr.Button(value="Compress", variant='primary')
    with gr.Row(equal_height=True):
        experiment_table = gr.Dataframe(
            value=experiment_df.filtered_with_headers([
                "is_fx_retrain", "id", "model", "task", "data", "performance", "macs", "params", "primary_metric"
            ]),
            interactive=False,
            height=600,
            overflow_row_behaviour='paginate',
            column_widths=[f"{x}%" for x in [3, 20]]
        )

    return experiment_table
