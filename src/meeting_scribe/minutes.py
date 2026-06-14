from typing import Any

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are an assistant that produces minutes of meetings from transcripts, "
    "with summary, key discussion points, takeaways and action items with owners, "
    "in markdown."
)

USER_PROMPT = (
    "Below is an extract transcript of a meeting. Please write minutes in markdown, "
    "including a summary with attendees, location and date; discussion points; "
    "takeaways; and action items with owners.\n{transcription}"
)

_llama_model = None
_llama_tokenizer = None


def generate_with_gpt(transcription: str, model: str = "gpt-4.1-mini") -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(transcription=transcription)},
        ],
    )
    return response.choices[0].message.content or ""


def _load_llama(model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct") -> tuple[Any, Any]:
    global _llama_model, _llama_tokenizer
    if _llama_model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        _llama_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _llama_tokenizer.pad_token = _llama_tokenizer.eos_token
        _llama_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quant_config,
        )
    return _llama_model, _llama_tokenizer


def generate_with_llama(
    transcription: str,
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
) -> str:
    import torch

    model, tokenizer = _load_llama(model_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(transcription=transcription)},
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=2000)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Keep only the assistant's reply after the last assistant header
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    idx = decoded.rfind(marker)
    if idx != -1:
        decoded = decoded[idx + len(marker):].strip()
    # Strip trailing EOS tokens
    decoded = decoded.replace("<|eot_id|>", "").strip()
    return decoded
