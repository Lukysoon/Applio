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

        # Create audio file upload component
        audio_files = gr.File(
            file_count="multiple",
            label=i18n("Audio Files"),
            type="filepath"
        )

        # Create exactly 3 textboxes for audio information
        audio_info_1 = gr.Textbox(
            label=i18n("Output Information 1"),
            info=i18n("The output information for the first audio file will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )

        audio_info_2 = gr.Textbox(
            label=i18n("Output Information 2"),
            info=i18n("The output information for the second audio file will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )

        audio_info_3 = gr.Textbox(
            label=i18n("Output Information 3"),
            info=i18n("The output information for the third audio file will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )

        # Create button
        get_info_button = gr.Button(
            value=i18n("Get information about the audio"),
            variant="primary"
        )

        # Create exactly 3 plot components
        plot_1 = gr.Image(type="filepath", interactive=False)
        plot_2 = gr.Image(type="filepath", interactive=False)
        plot_3 = gr.Image(type="filepath", interactive=False)

        comparison_plot_path = gr.Image(type="filepath", interactive=False)

        # Set up the click event with individual components
        get_info_button.click(
            fn=run_audio_analyzer_script,
            inputs=[audio_files, index_input],
            outputs=[
                audio_info_1,
                audio_info_2,
                audio_info_3,
                plot_1,
                plot_2,
                plot_3,
                comparison_plot_path
            ]
        )