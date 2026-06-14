import os

import gradio as gr
from dotenv import load_dotenv

from .transcribe import transcribe_audio
from .minutes import generate_with_gpt, generate_with_llama

load_dotenv()

_MODELS = {
    "GPT-4.1 Mini (OpenAI — closed-source)": "gpt",
    "Llama 3.1 8B (Meta — open-source, requires GPU)": "llama",
}


def process(audio_file: str | None, model_label: str, progress=gr.Progress()) -> tuple[str, str]:
    if audio_file is None:
        return "Please upload an audio file.", ""

    progress(0.1, desc="Transcribing audio with Whisper...")
    transcription = transcribe_audio(audio_file)

    progress(0.5, desc="Generating meeting minutes...")
    if _MODELS[model_label] == "gpt":
        minutes = generate_with_gpt(transcription)
    else:
        minutes = generate_with_llama(transcription)

    progress(1.0, desc="Done!")
    return transcription, minutes


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Meeting Scribe") as demo:
        gr.Markdown(
            "# Meeting Scribe\n"
            "Upload a meeting recording to generate structured minutes with action items."
        )
        with gr.Row():
            audio_input = gr.Audio(label="Meeting Audio", type="filepath")
            model_dropdown = gr.Dropdown(
                choices=list(_MODELS.keys()),
                value=list(_MODELS.keys())[0],
                label="Model",
            )
        btn = gr.Button("Generate Minutes", variant="primary")
        with gr.Row():
            transcript_box = gr.Textbox(label="Transcript", lines=10)
            minutes_box = gr.Markdown(label="Meeting Minutes", min_height=200)

        btn.click(
            process,
            inputs=[audio_input, model_dropdown],
            outputs=[transcript_box, minutes_box],
        )
    return demo


def main() -> None:
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
