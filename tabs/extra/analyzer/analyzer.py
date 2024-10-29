import os, sys
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_audio_analyzer_script
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def analyzer_tab():
    with gr.Column():
        index_input = gr.File(type="filepath")
        first_audio_input = gr.Audio(type="filepath")
        second_audio_input = gr.Audio(type="filepath")
        first_audio_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )
        second_audio_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )
        get_info_button = gr.Button(
            value=i18n("Get information about the audio"), variant="primary"
        )
        first_plot_path = gr.Image(type="filepath", interactive=False)
        second_plot_path = gr.Image(type="filepath", interactive=False)
        comparison_plot_path = gr.Image(type="filepath", interactive=False)

    get_info_button.click(
        fn=run_audio_analyzer_script,
        inputs=[first_audio_input, second_audio_input, index_input],
        outputs=[first_audio_info, second_audio_info, first_plot_path, second_plot_path, comparison_plot_path],
    )