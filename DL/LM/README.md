# LM

Language-model experiments for detecting AI-generated press-release text.

## Installation

The notebooks in this folder use standard Python NLP packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install transformers datasets torch scikit-learn evaluate pandas tqdm jupyter
```

A GPU is strongly recommended for the transformer fine-tuning notebooks.

## Quick Start

Open the RoBERTa fine-tuning notebook:

```bash
jupyter notebook roberta-finetuning.ipynb
```

The notebook loads `ai_press_releases.csv`, splits the human and AI columns into sentence-level samples, fine-tunes `roberta-base`, and reports validation accuracy and F1.

If you want the paragraph-level attention experiment instead, open:

```bash
jupyter notebook multi-attention-layer-training.ipynb
```

## What Is In This Folder?

This project studies whether official-style press-release text was written by humans or by a text generator.

- `roberta-finetuning.ipynb` is the main sentence-level RoBERTa fine-tuning workflow.
- `multi-attention-layer-training.ipynb` extends the setup with a paragraph-level multi-attention model.
- `ai_press_releases.csv` is the working dataset.
- `Final project.Rmd` and `Final-project.pdf` are the written report.
- `images/` contains plots and figures used in the report.

## Why This Project Exists

The central research question in the report is simple: how much AI-written text appears in official U.S. Senate press-release material?

The folder approaches that question in two steps:

- a sentence-level detector based on `roberta-base`
- a longer-context experiment that weights the most informative sentences in a paragraph

That makes the folder useful both as a modeling experiment and as a course project archive.

## Main Workflows

### `roberta-finetuning.ipynb`

This notebook:

1. loads `ai_press_releases.csv`
2. drops missing rows
3. splits both source columns into sentence-level samples
4. builds train, validation, and test splits with `stratify`
5. fine-tunes `roberta-base`
6. evaluates with accuracy and weighted F1

Core model setup:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2,
)
```

### `multi-attention-layer-training.ipynb`

This notebook shifts from sentence classification to paragraph classification. It splits paragraphs into sentences, encodes them with a RoBERTa-based model, then applies a multi-attention layer to focus on the most informative parts of the paragraph.

The notebook references a previously trained checkpoint path when loading the sentence encoder, so you may need to edit local paths before re-running it.

### `Final project.Rmd`

The report ties the project together:

- research question and motivation
- background on perceptrons, transformers, and RoBERTa
- integrated gradients for interpretation
- the paragraph-level attention architecture
- training notes and supporting figures

## Data and Artifacts

| Path | Purpose |
| --- | --- |
| `ai_press_releases.csv` | Source dataset with human and AI press-release text |
| `images/` | ROC, confusion matrix, training curves, t-SNE, and report figures |
| `data/prob.csv` | Saved probability-related artifact |
| `interpretation.zip` | Extra interpretation material |
| `Final-project.pdf` | Rendered project report |

## Examples

Open the main notebook in Jupyter:

```bash
jupyter notebook roberta-finetuning.ipynb
```

Open the paragraph-level experiment:

```bash
jupyter notebook multi-attention-layer-training.ipynb
```

Render the report if you have R Markdown tooling available:

```bash
Rscript -e "rmarkdown::render('Final project.Rmd')"
```

## Notes

- The notebooks were originally run in Kaggle-style environments, so some file paths may need to be changed for local use.
- If you hit CUDA memory issues, reduce `batch_size` or sequence length first.
- Check for text overlap between splits if you change the sentence or paragraph preprocessing.

## License

No project-specific license file is included in this folder.
