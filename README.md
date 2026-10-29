# NLP Workbench

Interactive NLP application with three capabilities:

1. Sentiment analysis using a DistilBERT model fine-tuned on Yelp reviews.
2. Extractive question answering using a DistilBERT SQuAD model and generative QA using FLAN-T5-small.
3. Controlled text generation using GPT-2, with greedy and sampling options.

## Runtime requirements

- Python 3.10 or later compatible with the selected PyTorch and Transformers releases
- Internet access on first launch to download the pretrained models
- The fine-tuned sentiment model saved at `models/yelp_distilbert_sentiment`
- GPU recommended for responsive inference; CPU may be slower

## Run from Colab

The application file expects the project root to be available at:

`/content/drive/MyDrive/NLP_Workbench`

Mount Google Drive before launching, then run:

```bash
python deployment/app.py
```

For a Windows or other local installation, set `NLP_WORKBENCH_ROOT` to the folder containing the `models` directory before launching.

## Important

The Gradio share URL created in Colab is temporary. It is not a permanent deployment URL. Keep the model files and project results backed up.
