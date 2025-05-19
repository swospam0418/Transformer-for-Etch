import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        length = x.size(1)
        return x + self.pe[:length]


def plot_positional_encoding(pe: torch.Tensor) -> None:
    """Display a heatmap of positional encodings."""
    plt.figure(figsize=(6, 3))
    plt.imshow(pe.cpu().numpy(), aspect="auto", cmap="viridis")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Step")
    plt.title("Positional Encoding")
    plt.colorbar(label="value")
    plt.tight_layout()
    plt.show()



class RecipeDataset(Dataset):
    """Example dataset for plasma etching recipes."""

    def __init__(self, csv_path: str, seq_len: int, num_step_types: int):
        self.data = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.num_step_types = num_step_types

        # Example columns: step_type_0 ... step_type_{L-1}, knob_0 ... knob_{L-1}, targets ...
        self.step_cols = [f"step_type_{i}" for i in range(seq_len)]
        self.knob_cols = [f"knob_{i}" for i in range(seq_len)]
        self.target_cols = [c for c in self.data.columns if c.startswith("target_")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        step_types = row[self.step_cols].to_numpy(dtype=np.int64)
        knobs = row[self.knob_cols].to_numpy(dtype=np.float32)
        targets = row[self.target_cols].to_numpy(dtype=np.float32)
        return torch.from_numpy(step_types), torch.from_numpy(knobs), torch.from_numpy(targets)

      
class ExcelRecipeDataset(Dataset):
    """Load recipe data from an Excel workbook with multiple sheets."""

    def __init__(self, excel_path: str):
        sheets = pd.read_excel(excel_path, sheet_name=None)
        frames = []
        max_len = 0
        for df in sheets.values():
            step_cols = [c for c in df.columns if c.startswith("step_type_")]
            knob_cols = [c for c in df.columns if c.startswith("knob_")]
            max_len = max(max_len, len(step_cols))
            frames.append(df)

        all_step_cols = [f"step_type_{i}" for i in range(max_len)]
        all_knob_cols = [f"knob_{i}" for i in range(max_len)]
        for i, df in enumerate(frames):
            for col in all_step_cols:
                if col not in df.columns:
                    df[col] = 0
            for col in all_knob_cols:
                if col not in df.columns:
                    df[col] = 0.0
            df = df.reindex(columns=all_step_cols + all_knob_cols, fill_value=0)
            frames[i] = df

        self.data = pd.concat(frames, ignore_index=True)
        self.seq_len = max_len
        self.step_cols = all_step_cols
        self.knob_cols = all_knob_cols
        self.target_cols = [c for c in self.data.columns if c.startswith("target_")]
        self.num_step_types = int(self.data[self.step_cols].max().max()) + 1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        step_types = row[self.step_cols].to_numpy(dtype=np.int64)
        knobs = row[self.knob_cols].to_numpy(dtype=np.float32)
        targets = row[self.target_cols].to_numpy(dtype=np.float32)
        return (
            torch.from_numpy(step_types),
            torch.from_numpy(knobs),
            torch.from_numpy(targets),
        )


class MultiSheetRecipeDataset(Dataset):
    """Dataset for workbooks where each sheet contains a recipe structure.

    Columns beginning with ``X_`` define the sequential steps in that sheet. The
    portion between the first and second underscore denotes the step type (e.g.
    ``X_ME_Power``). Columns beginning with ``Y_`` are treated as targets. The
    loader flattens each sheet into ``step_type_i`` and ``knob_i`` columns so
    that all sheets can be concatenated into a single table.
    """

    def __init__(self, excel_path: str) -> None:
        sheets = pd.read_excel(excel_path, sheet_name=None)

        step_names: set[str] = set()
        max_len = 0
        processed: list[pd.DataFrame] = []

        # First pass to gather step types and maximum sequence length
        temp_frames = []
        for df in sheets.values():
            step_cols = [c for c in df.columns if c.startswith("X_")]
            target_cols = [c for c in df.columns if c.startswith("Y_")]
            order = step_cols
            max_len = max(max_len, len(order))
            step_names.update(col.split("_")[1] for col in order)
            temp_frames.append((df, order, target_cols))

        self.step_map = {name: idx for idx, name in enumerate(sorted(step_names))}

        all_step_cols = [f"step_type_{i}" for i in range(max_len)]
        all_knob_cols = [f"knob_{i}" for i in range(max_len)]

        # Convert each sheet to unified layout
        for df, order, tcols in temp_frames:
            new_df = pd.DataFrame()
            for i in range(max_len):
                if i < len(order):
                    orig = order[i]
                    step_name = orig.split("_")[1]
                    new_df[f"step_type_{i}"] = self.step_map[step_name]
                    new_df[f"knob_{i}"] = df[orig].astype(float)
                else:
                    new_df[f"step_type_{i}"] = 0
                    new_df[f"knob_{i}"] = 0.0
            for col in tcols:
                new_df[col] = df[col].astype(float)
            processed.append(new_df)

        # Ensure all frames share the same target columns
        target_cols = sorted({c for _, _, tc in temp_frames for c in tc})
        for df in processed:
            for c in target_cols:
                if c not in df.columns:
                    df[c] = 0.0
            df.sort_index(axis=1, inplace=True)

        self.data = pd.concat(processed, ignore_index=True)
        self.seq_len = max_len
        self.step_cols = all_step_cols
        self.knob_cols = all_knob_cols
        self.target_cols = target_cols
        self.num_step_types = len(self.step_map)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        step_types = row[self.step_cols].to_numpy(dtype=np.int64)
        knobs = row[self.knob_cols].to_numpy(dtype=np.float32)
        targets = row[self.target_cols].to_numpy(dtype=np.float32)
        return (
            torch.from_numpy(step_types),
            torch.from_numpy(knobs),
            torch.from_numpy(targets),
        )




class AttentionModel(nn.Module):
    """Transformer-like model with optional attention weight extraction."""

    def __init__(self, num_step_types: int, d_model: int, nhead: int, num_targets: int, seq_len: int):
        super().__init__()
        self.embedding = nn.Embedding(num_step_types, d_model)
        self.knob_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.fc = nn.Linear(d_model, num_targets)

    def forward(self, step_types: torch.Tensor, knobs: torch.Tensor, return_attn: bool = False):
        """Forward pass.

        Args:
            step_types: (B, L) tensor of step category ids.
            knobs: (B, L) tensor of knob values.
            return_attn: if True, also return attention weights.
        """
        x = self.embedding(step_types) + self.knob_proj(knobs.unsqueeze(-1))
        x = self.pos_encoder(x)
        attn_output, attn_weights = self.attn(x, x, x, need_weights=return_attn)
        x = self.ff(attn_output)
        out = self.fc(x.mean(dim=1))
        if return_attn:
            return out, attn_weights
        return out

    def attention_heatmap(self, step_types: torch.Tensor, knobs: torch.Tensor) -> torch.Tensor:
        """Return attention weights for plotting."""
        self.eval()
        with torch.no_grad():
            _, weights = self.forward(step_types.unsqueeze(0), knobs.unsqueeze(0), return_attn=True)
        return weights.squeeze(0)


def plot_attention_heatmap(weights: torch.Tensor) -> None:
    """Plot an attention weight heatmap."""
    if weights.dim() == 3:
        weights = weights.mean(0)
    plt.figure(figsize=(4, 4))
    plt.imshow(weights.cpu().numpy(), aspect="auto", cmap="viridis")
    plt.xlabel("Step")
    plt.ylabel("Step")
    plt.title("Attention Weights")
    plt.colorbar(label="weight")
    plt.tight_layout()
    plt.show()


def step_importance(weights: torch.Tensor) -> torch.Tensor:
    """Return average attention each step receives."""
    if weights.dim() == 3:
        weights = weights.mean(0)
    return weights.mean(0)



def train_multisheet_excel(excel_path: str, epochs: int = 10) -> None:
    """Train on a workbook containing multiple recipe structures."""
    dataset = MultiSheetRecipeDataset(excel_path)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = AttentionModel(
        num_step_types=dataset.num_step_types,
        d_model=64,
        nhead=4,
        num_targets=len(dataset.target_cols),
        seq_len=dataset.seq_len,
    )

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for step_types, knobs, targets in train_loader:
            optim.zero_grad()
            preds = model(step_types, knobs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optim.step()

        # evaluate R2 on train and test sets
        model.eval()
        with torch.no_grad():
            train_preds, train_tgts = [], []
            for st, kb, tg in train_loader:
                p = model(st, kb)
                train_preds.append(p)
                train_tgts.append(tg)
            test_preds, test_tgts = [], []
            for st, kb, tg in test_loader:
                p = model(st, kb)
                test_preds.append(p)
                test_tgts.append(tg)
        train_preds = torch.cat(train_preds).cpu().numpy()
        train_tgts = torch.cat(train_tgts).cpu().numpy()
        test_preds = torch.cat(test_preds).cpu().numpy()
        test_tgts = torch.cat(test_tgts).cpu().numpy()

        train_r2 = r2_score(train_tgts, train_preds, multioutput="variance_weighted")
        test_r2 = r2_score(test_tgts, test_preds, multioutput="variance_weighted")
        print(
            f"Epoch {epoch+1}/{epochs} Loss {loss.item():.4f} Train R2 {train_r2:.3f} Test R2 {test_r2:.3f}"
        )

    # explainability utilities
    plot_positional_encoding(model.pos_encoder.pe[: dataset.seq_len])
    sample_steps, sample_knobs, sample_targets = next(iter(test_loader))
    attn = model.attention_heatmap(sample_steps[0], sample_knobs[0])
    plot_attention_heatmap(attn)

    # show real vs predicted for a few recipes
    with torch.no_grad():
        preds = model(sample_steps, sample_knobs)
    for i in range(min(3, len(sample_steps))):
        print(
            f"Recipe {i}: real={sample_targets[i].tolist()} pred={preds[i].tolist()}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the attention model on a recipe workbook"
    )
    parser.add_argument(
        "workbook",
        nargs="?",
        default="recipes.xlsx",
        help="Path to the Excel workbook with recipe sheets",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    args, _ = parser.parse_known_args()

    train_multisheet_excel(args.workbook, epochs=args.epochs)

