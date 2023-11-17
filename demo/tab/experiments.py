import os
from pathlib import Path

import gradio as gr
from func.experiments import ExperimentDataFrame, COLUMN_NAME_AS


def tab_experiments(args):
    experiment_df = ExperimentDataFrame(experiment_dir="./outputs")

    experiment_select_task_choices = sorted(list(set(experiment_df.dataframe['task'])))
    experiment_select_data_choices = sorted(list(set(experiment_df.dataframe['data'])))
    experiment_select_model_choices = sorted(list(set(experiment_df.dataframe['model'])))

    with gr.Row():
        with gr.Column():
            experiment_summary_plot = gr.ScatterPlot(
                value=None,
                x=COLUMN_NAME_AS["macs"],
                y=COLUMN_NAME_AS["performance"]
            )
        with gr.Column():
            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    with gr.Group():
                        with gr.Row():
                            experiment_select_task = gr.Dropdown(
                                label="Task",
                                choices=experiment_select_task_choices,
                            )
                            experiment_select_data = gr.Dropdown(
                                label="Dataset",
                                choices=experiment_select_data_choices,
                            )
                            experiment_select_model = gr.Dropdown(
                                label="Model",
                                choices=experiment_select_model_choices,
                            )
                        experiment_threshold_macs = gr.Slider(label="MACs (smaller than)")
                        experiment_threshold_params = gr.Slider(label="# Params (smaller than)")
                        experiment_select_compressed_only = gr.Checkbox(
                            value=True, label="Show compressed models", interactive=True
                        )
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
                "is_fx_retrain", "id", "model", "task", "data",
                "primary_metric", "performance", "macs", "params",
            ]),
            interactive=False,
            height=600,
            overflow_row_behaviour='paginate',
            column_widths=[f"{x}%" for x in [5, 20]]
        )

    experiment_table.change(
        lambda x: x, inputs=[experiment_table], outputs=[experiment_summary_plot]
    )

    return experiment_table
