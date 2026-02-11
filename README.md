# QuestionGenerator

Prototype scripts for generating questions from text in two ways:

1. **Model-based**: fine-tune a Hugging Face **T5** model to generate questions from short passages (e.g., movie plots) using the JSON datasets in `datasets/`.
2. **OCR + NLP**: convert PDF pages to images, extract text with **Tesseract**, and run **spaCy** NER to build simple template questions from detected entities.

## Project layout

- `question_class.py` — fine-tunes `t5-small` for question generation and saves:
  - `trained_t5_model/` (model + tokenizer)
  - `training_results.pt` (losses + model state)
  - `loss_graph.png` (via `graph_plot.py`)
- `test.py` — loads a trained model and generates questions for sample plots from `datasets/ParaphraseRC_dev.json`.
- `image_converter.py` — converts a PDF to PNGs (uses `pdf2image`).
- `main.py` — OCR + spaCy analysis pipeline (currently uses hard-coded Windows paths).
- `datasets/` — JSON datasets (`ParaphraseRC_dev.json`, `ParaphraseRC_test.json`, etc.).
- `pdf_images/` — sample extracted images.

## Requirements

This project is a collection of scripts (no packaged dependencies). You’ll need Python and a few libraries.

### Python packages

Install the core dependencies:

```bash
python3 -m pip install torch transformers matplotlib
```

For the OCR/NLP utilities:

```bash
python3 -m pip install pillow pytesseract spacy pdf2image
python3 -m spacy download en_core_web_sm
```

### System dependencies (OCR/PDF)

- **Tesseract** (required for `pytesseract`)
  - macOS: `brew install tesseract`
- **Poppler** (required for `pdf2image`)
  - macOS: `brew install poppler`

## Fine-tune T5 (question generation)

1. From the `QuestionGenerator/` directory:

```bash
python3 question_class.py
```

2. After training, the model is saved to `trained_t5_model/`.

Notes:
- Training uses `datasets/ParaphraseRC_test.json` and creates a 70/30 train/validation split.
- If CUDA is available, the script will use it; otherwise it runs on CPU (slow).

## Generate questions with a trained model

`test.py` loads a model directory and generates a question for each sample plot:

```bash
python3 test.py
```

If you trained locally with `question_class.py`, update `model_directory` in `test.py` to point at:

- `trained_t5_model`

## PDF → images → OCR (optional utilities)

1. Convert a PDF to images by editing paths in `image_converter.py`, then run:

```bash
python3 image_converter.py
```

2. Run OCR + spaCy analysis by editing the hard-coded paths in `main.py`, then run:

```bash
python3 main.py
```

## Notes
- Several scripts currently contain **hard-coded Windows paths** (e.g., `C:\\Users\\...`). Update those paths for your machine before running.
- `question_class.py` saves the trained model to `trained_t5_model/`, while `test.py` is currently configured to load from `outputs/1/trained_t5_model`.

