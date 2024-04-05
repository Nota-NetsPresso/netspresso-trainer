import os
from pathlib import Path

import gradio as gr
import pandas as pd
from func.experiments import COLUMN_NAME_AS, ExperimentDataFrame


def tab_experiments(args):
    experiment_df = ExperimentDataFrame(
        headers=[
            "is_fx_retrain", "id", "model", "task", "data",
            "primary_metric", "performance", "macs", "params",
        ],
        experiment_dir="./outputs"
    )

    experiment_select_task_choices = sorted(set(experiment_df.default_no_render['task']))
    experiment_select_data_choices = sorted(set(experiment_df.default_no_render['data']))
    experiment_select_model_choices = sorted(set(experiment_df.default_no_render['model']))

    def id_from_dataframe_select(df: pd.DataFrame, evt: gr.SelectData):
        return df.at[evt.index[0], COLUMN_NAME_AS["id"]]
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row(equal_height=True), gr.Column(scale=4), gr.Group():
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
                experiment_threshold_macs = gr.Slider(
                    label="MACs (G, smaller than or equal to)",
                    minimum=0, maximum=10, step=0.1, value=-1
                )
                experiment_threshold_params = gr.Slider(
                    label="# Params (M, smaller than or equal to)",
                    minimum=0, maximum=1000, step=1, value=-1
                )
                experiment_select_compressed_ignore = gr.Checkbox(
                    value=False, label="Ignore compressed models", interactive=True
                )
            with gr.Row():
                gr.ClearButton([
                    experiment_select_task,
                    experiment_select_data,
                    experiment_select_model,
                    experiment_threshold_macs,
                    experiment_threshold_params,
                    experiment_select_compressed_ignore
                ])
                experiment_button_search = gr.Button(value="Search", variant='primary')
        with gr.Column(scale=2):
            with gr.Row():
                experiment_selected = gr.Textbox(label="Selected checkpoint")
            with gr.Row():
                experiment_button_launcher = gr.Button(value="Benchmark")
                experiment_button_compressor = gr.Button(value="Compress", variant='primary')

    with gr.Accordion("📊 See scatter plot", open=False):
        gr.Markdown("Scatter plot is rendered when the filter 'task' is selected.")
        experiment_summary_plot = gr.ScatterPlot(
            value=None,
            x=COLUMN_NAME_AS['macs'],
            y=COLUMN_NAME_AS['performance'],
            tooltip=COLUMN_NAME_AS['id'],
        )

    with gr.Row(equal_height=True):
        experiment_table = gr.Dataframe(
            value=experiment_df.default,
            interactive=False,
            height=600,
            overflow_row_behaviour='paginate',
            column_widths=[f"{x}%" for x in [5, 20]]
        )

    experiment_button_search.click(
        fn=experiment_df.filter_with,
        inputs=[
            experiment_select_task,
            experiment_select_data,
            experiment_select_model,
            experiment_threshold_macs,
            experiment_threshold_params,
            experiment_select_compressed_ignore
        ],
        outputs=[experiment_table])
    experiment_table.change(
        lambda x: x, inputs=[experiment_table], outputs=[experiment_summary_plot]
    )
    experiment_table.select(
        fn=id_from_dataframe_select,
        inputs=[experiment_table],
        outputs=[experiment_selected],
        show_progress="hidden"
    )

    return experiment_df, experiment_selected, experiment_button_launcher, experiment_button_compressor
