
import os
import time
from pathlib import Path
import torch
import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(
    os.environ.get("NLP_WORKBENCH_ROOT", str(DEFAULT_PROJECT_ROOT))
)

DEFAULT_SENTIMENT_PATH = (
    PROJECT_ROOT / "models" / "yelp_distilbert_sentiment"
)

# For cloud deployment, this can be a Hugging Face model repository ID.
# If unset, load the fine-tuned model from the project directory.
SENTIMENT_MODEL_ID = (
    os.environ.get("SENTIMENT_MODEL_ID")
    or str(DEFAULT_SENTIMENT_PATH)
)

QA_MODEL_ID = "distilbert/distilbert-base-cased-distilled-squad"
GEN_QA_MODEL_ID = "google/flan-t5-small"
GPT2_MODEL_ID = "openai-community/gpt2"

print("Loading sentiment model...")
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_ID)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    SENTIMENT_MODEL_ID
).to(DEVICE).eval()

print("Loading extractive QA model...")
qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_ID)
qa_model = AutoModelForQuestionAnswering.from_pretrained(
    QA_MODEL_ID
).to(DEVICE).eval()

print("Loading generative QA model...")
genqa_tokenizer = AutoTokenizer.from_pretrained(GEN_QA_MODEL_ID)
genqa_model = AutoModelForSeq2SeqLM.from_pretrained(
    GEN_QA_MODEL_ID
).to(DEVICE).eval()

print("Loading GPT-2...")
gpt_tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_ID)
gpt_model = AutoModelForCausalLM.from_pretrained(
    GPT2_MODEL_ID
).to(DEVICE).eval()

# GPT-2 has no dedicated padding token by default.
if gpt_tokenizer.pad_token is None:
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_model.config.pad_token_id = gpt_tokenizer.pad_token_id


def analyze_sentiment(review):
    if not review or not review.strip():
        return "Please enter a review.", {}

    inputs = sentiment_tokenizer(
        review.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(DEVICE)

    with torch.inference_mode():
        logits = sentiment_model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    label = sentiment_model.config.id2label[predicted_id].upper()

    # Display probabilities using the model's actual label mapping.
    details = {
        sentiment_model.config.id2label[i].upper():
            round(float(probabilities[i].item()), 4)
        for i in range(len(probabilities))
    }
    return label, details


def answer_question(context, question):
    if not context or not context.strip():
        return "Please enter a context.", "", 0.0
    if not question or not question.strip():
        return "Please enter a question.", "", 0.0

    context = context.strip()
    question = question.strip()

    # Extractive QA: locate an answer span in the context.
    inputs = qa_tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=384
    ).to(DEVICE)

    with torch.inference_mode():
        outputs = qa_model(**inputs)

    start = int(torch.argmax(outputs.start_logits, dim=-1).item())
    end = int(torch.argmax(outputs.end_logits, dim=-1).item())

    if end < start:
        extractive_answer = ""
    else:
        answer_ids = inputs["input_ids"][0][start:end + 1]
        extractive_answer = qa_tokenizer.decode(
            answer_ids, skip_special_tokens=True
        ).strip()

    # Generative QA: generate an answer using the same context.
    prompt = (
        "Answer the question using only the context. "
        "If the answer is not stated, say that it is not stated.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer:"
    )

    gen_inputs = genqa_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    start_time = time.perf_counter()
    with torch.inference_mode():
        generated_ids = genqa_model.generate(
            **gen_inputs,
            max_new_tokens=64,
            do_sample=False
        )
    elapsed = time.perf_counter() - start_time

    generated_answer = genqa_tokenizer.decode(
        generated_ids[0], skip_special_tokens=True
    ).strip()

    return extractive_answer, generated_answer, round(elapsed, 3)


def generate_text(
    prompt,
    mode,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    seed
):
    if not prompt or not prompt.strip():
        return "Please enter a prompt.", {}

    if mode not in ("Greedy", "Sampling"):
        return "Invalid decoding mode.", {}

    if mode == "Sampling" and temperature <= 0:
        return "Temperature must be greater than zero.", {}

    inputs = gpt_tokenizer(
        prompt.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    generation_args = {
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": gpt_tokenizer.pad_token_id,
        "eos_token_id": gpt_tokenizer.eos_token_id,
    }

    if mode == "Greedy":
        generation_args["do_sample"] = False
    else:
        generation_args.update({
            "do_sample": True,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
        })
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

    start_time = time.perf_counter()
    with torch.inference_mode():
        output_ids = gpt_model.generate(
            **inputs,
            **generation_args
        )
    elapsed = time.perf_counter() - start_time

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    continuation = gpt_tokenizer.decode(
        new_tokens, skip_special_tokens=True
    ).strip()

    details = {
        "mode": mode.lower(),
        "max_new_tokens": int(max_new_tokens),
        "temperature": (
            float(temperature) if mode == "Sampling" else None
        ),
        "top_k": int(top_k) if mode == "Sampling" else None,
        "top_p": float(top_p) if mode == "Sampling" else None,
        "seed": int(seed) if mode == "Sampling" and seed is not None else None,
        "generated_tokens": int(len(new_tokens)),
        "response_time_seconds": round(elapsed, 3),
        "device": str(DEVICE),
    }
    return continuation, details


with gr.Blocks(title="NLP Workbench") as demo:
    gr.Markdown("# NLP Workbench")
    gr.Markdown(
        "Sentiment Analysis · Question Answering · Text Generation"
    )
    gr.Markdown(
        "A Transformer-based NLP application combining a Yelp-trained "
        "sentiment classifier, extractive and generative QA, and GPT-2."
    )

    with gr.Tab("Sentiment Analysis"):
        review = gr.Textbox(
            label="Review text",
            placeholder="Enter a restaurant or service review...",
            lines=4
        )
        sentiment_button = gr.Button("Analyze Sentiment")
        sentiment_output = gr.Textbox(label="Predicted sentiment")
        sentiment_details = gr.JSON(label="Class probabilities")
        sentiment_button.click(
            analyze_sentiment,
            inputs=[review],
            outputs=[sentiment_output, sentiment_details]
        )
        gr.Examples(
            examples=[
                ["The food was delicious and the staff were very friendly."],
                ["The service was terrible and the food was disappointing."]
            ],
            inputs=review
        )

    with gr.Tab("Question Answering"):
        context = gr.Textbox(
            label="Context / passage", lines=5,
            placeholder="Enter the passage containing relevant information..."
        )
        question = gr.Textbox(
            label="Question",
            placeholder="Ask a question about the passage..."
        )
        qa_button = gr.Button("Answer Question")
        extractive_output = gr.Textbox(label="Extractive QA answer")
        generative_output = gr.Textbox(label="Generative QA answer")
        qa_time = gr.Number(label="Generative response time (seconds)")
        qa_button.click(
            answer_question,
            inputs=[context, question],
            outputs=[extractive_output, generative_output, qa_time]
        )

    with gr.Tab("Text Generation"):
        prompt = gr.Textbox(
            label="Prompt", lines=4,
            placeholder="Enter a prompt..."
        )
        mode = gr.Radio(
            choices=["Greedy", "Sampling"],
            value="Greedy",
            label="Decoding strategy"
        )
        max_tokens = gr.Slider(
            1, 150, value=40, step=1, label="Maximum new tokens"
        )
        temperature = gr.Slider(
            0.1, 1.5, value=0.8, step=0.1, label="Temperature"
        )
        top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
        top_p = gr.Slider(
            0.1, 1.0, value=0.95, step=0.05, label="Top-P"
        )
        seed = gr.Number(
            value=44, precision=0, label="Random seed for sampling"
        )
        generate_button = gr.Button("Generate Text")
        generated_output = gr.Textbox(
            label="Generated continuation", lines=8
        )
        generation_details = gr.JSON(label="Generation configuration and timing")
        generate_button.click(
            generate_text,
            inputs=[
                prompt, mode, max_tokens, temperature, top_k, top_p, seed
            ],
            outputs=[generated_output, generation_details]
        )

    gr.Markdown("---")
    gr.Markdown(
        "**Models:** Yelp-trained DistilBERT · DistilBERT extractive QA · "
        "FLAN-T5-small · GPT-2"
    )
    gr.Markdown(
        "**Note:** Model outputs can be incomplete or incorrect. "
        "Evaluate answers rather than treating them as guaranteed facts."
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        show_error=True
    )
