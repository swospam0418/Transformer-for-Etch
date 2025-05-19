# Transformer for Plasma Etching

This project provides a minimal implementation of an attention-based model
to learn the relationship between a plasma etching recipe and the resulting
profile metrics. It is intended as a starting point for researchers who want
to experiment with Transformer-like architectures on process control data.

The code uses PyTorch and expects recipe steps and tuning knob values as input
features. The model outputs predicted profile properties such as bamboo width
or etch depth. The example also demonstrates basic explainability utilities
like positional encoding and attention heatmaps so you can inspect how each
step influences the prediction or etch depth.


## Files

- `attention_model.py` – basic attention model and utilities. Includes an
  example for loading recipe data from an Excel workbook where each sheet
  corresponds to a different recipe structure.
- `etch_rate_model.py` – extended example with film scheme and layout embeddings.
- `requirements.txt` – Python dependencies (including `openpyxl` for Excel
  support).

## Usage

Install the dependencies and adapt the dataset loader in `attention_model.py`
to your CSV, database, or Excel workbook. The script includes a simple
training loop example that demonstrates loading multiple sheets from an Excel
file.

```bash
pip install -r requirements.txt
python attention_model.py
```


Running the script will train on a small synthetic dataset and then display a
heatmap of the positional encodings as well as the attention weights for an
example recipe, giving a quick look at how steps interact.


This repository does not include a dataset. You must provide your own
data in the expected format to train the model.
