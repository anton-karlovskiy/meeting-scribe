# Meeting Scribe

Transcribes meeting audio and generates structured minutes with action items. Uses OpenAI Whisper for transcription and your choice of model for the minutes:

- **GPT-4.1 Mini** (OpenAI, closed-source) — fast, no GPU required
- **Llama 3.1 8B** (Meta, open-source) — runs locally, requires a CUDA GPU

## Setup

**1. Install dependencies**

```bash
uv sync
```

For the open-source Llama path, also install the `llm` extras:

```bash
uv sync --extra llm
```

**2. Configure environment**

```bash
cp .env.example .env
```

Then edit `.env`:

```
OPENAI_API_KEY=sk-...       # required for Whisper transcription and GPT minutes
HF_TOKEN=hf_...             # required only when using the Llama model
```

To use Llama 3.1 8B you also need to accept the license on [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).

> **Note:** You do not need to download the model manually. On the first run with Llama selected, the weights (~16 GB) are downloaded automatically to your HuggingFace cache (`~/.cache/huggingface/`). Subsequent runs use the cached copy. A CUDA-capable NVIDIA GPU is required.

**3. Run**

```bash
uv run meeting-scribe
```

Then open the local URL printed to the terminal (default: `http://127.0.0.1:7860`).

## Usage

1. Upload an MP3 or WAV recording of your meeting
2. Select a model (GPT-4.1 Mini or Llama 3.1 8B)
3. Click **Generate Minutes**

The app returns:
- Full transcript of the audio
- Markdown meeting minutes with summary, key discussion points, takeaways, and action items with owners

## Project structure

```
src/meeting_scribe/
├── transcribe.py   # Whisper transcription via OpenAI API
├── minutes.py      # Minutes generation (GPT and Llama backends)
└── app.py          # Gradio UI and entry point
```

## Sample audio

The original course used a Denver City Council meeting extract. You can download it [here](https://drive.google.com/file/d/1N_kpSojRR5RYzupz6nqM8hMSoEF_R7pU/view?usp=sharing) or find the full dataset on [HuggingFace](https://huggingface.co/datasets/huuuyeah/MeetingBank_Audio/tree/main).
