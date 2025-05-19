# Transformer for Plasma Etching

This project provides a minimal implementation of an attention-based model to learn how plasma etching recipes influence profile metrics. It relies on PyTorch and expects tuning knob values as inputs. Basic explainability utilities such as positional encoding and attention heatmaps are included so you can inspect which steps matter most.

## Files

- `attention_model.py` – attention model and utilities for loading an Excel workbook where each sheet represents a recipe structure.
- `etch_rate_model.py` – extended example with film scheme and layout embeddings.
- `requirements.txt` – Python dependencies (includes `openpyxl` for Excel support).

## Usage

Prepare an Excel workbook where each sheet defines a recipe. Columns starting with `X_` represent sequential step inputs and those beginning with `Y_` are targets. Install the dependencies and run training:

```bash
pip install -r requirements.txt
python attention_model.py /path/to/recipes.xlsx --epochs 20
```

If no workbook path is given, the script looks for `recipes.xlsx` in the current directory.

This repository does not include any dataset; you must supply your own workbook to train the model.
