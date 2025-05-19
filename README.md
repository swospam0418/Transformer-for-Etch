# Transformer for Plasma Etching

This project provides a minimal implementation of an attention-based model to learn how plasma etching recipes influence profile metrics. It relies on PyTorch and expects tuning knob values as inputs. Basic explainability utilities such as positional encoding and attention heatmaps are included so you can inspect which steps matter most.

## Files

- `attention_model.py` – attention model and utilities for loading an Excel workbook where each sheet represents a recipe structure.
- `etch_rate_model.py` – extended example with film scheme and layout embeddings.
- `requirements.txt` – Python dependencies (includes `openpyxl` for Excel support).

## Usage

Prepare an Excel workbook where each sheet defines a recipe.

### Column naming

Step columns must be prefixed with `X_<stepType>` describing the step type
(for example `X_ME_Power`). Target columns should begin with `Y_`.
An excerpt of a minimal worksheet might look like:

```
| X_ME_Power | X_ME_Time | Y_EtchRate |
|------------|-----------|-----------|
| 300        | 60        | 42        |
```

Install the dependencies and run training:

```bash
pip install -r requirements.txt
python attention_model.py /path/to/recipes.xlsx --epochs 20


- `attention_model.py` – basic attention model and utilities. Includes an
  example for loading recipe data from an Excel workbook where each sheet
  corresponds to a different recipe structure.
- `etch_rate_model.py` – extended example with film scheme and layout embeddings.
- `requirements.txt` – Python dependencies (including `openpyxl` for Excel
  support).

## Usage

Install the dependencies and adapt the dataset loader in `attention_model.py`
to your Excel workbook. The script includes a simple training loop that
demonstrates loading multiple sheets and reports training/test R² scores.

```bash
pip install -r requirements.txt
python attention_model.py recipes.xlsx --epochs 20
```

When running the script from an IDE such as VS Code that uses an IPython
kernel, make sure the workbook path is supplied on the command line. For
example:

```bash
python attention_model.py my_recipes.xlsx
```

You can also invoke the training routine directly from a Python shell:

```python
from attention_model import train_multisheet_excel
train_multisheet_excel("my_recipes.xlsx")
```

If no workbook path is given, the script looks for `recipes.xlsx` in the current directory.

This repository does not include any dataset; you must supply your own workbook to train the model.


Running the script expects an Excel workbook in the format described in
`attention_model.py`. It will display a heatmap of the positional encodings and
attention weights for a sample recipe so you can inspect how steps interact.


This repository does not include a dataset. You must provide your own
data in the expected format to train the model.

