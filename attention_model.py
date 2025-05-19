import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
        for df in frames:
            for col in all_step_cols:
                if col not in df.columns:
                    df[col] = 0
            for col in all_knob_cols:
                if col not in df.columns:
                    df[col] = 0.0
            df.reindex(columns=all_step_cols + all_knob_cols, fill_value=0)

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


def _generate_example_csv(path: str, num_samples=100, seq_len=4, num_targets=2, num_step_types=5):
    rng = np.random.default_rng(0)
    data = {}
    for i in range(seq_len):
        data[f"step_type_{i}"] = rng.integers(0, num_step_types, size=num_samples)
        data[f"knob_{i}"] = rng.random(size=num_samples)
    for t in range(num_targets):
        data[f"target_{t}"] = rng.random(size=num_samples)
    pd.DataFrame(data).to_csv(path, index=False)


def _generate_example_excel(path: str) -> None:
    """Create a toy Excel workbook with two recipe structures."""
    rng = np.random.default_rng(0)
    # first sheet has 4 steps
    data_a = {}
    for i in range(4):
        data_a[f"step_type_{i}"] = rng.integers(0, 5, size=50)
        data_a[f"knob_{i}"] = rng.random(size=50)
    data_a["target_0"] = rng.random(size=50)
    df_a = pd.DataFrame(data_a)

    # second sheet has 3 steps
    data_b = {}
    for i in range(3):
        data_b[f"step_type_{i}"] = rng.integers(0, 5, size=50)
        data_b[f"knob_{i}"] = rng.random(size=50)
    data_b["target_0"] = rng.random(size=50)
    df_b = pd.DataFrame(data_b)

    with pd.ExcelWriter(path) as writer:
        df_a.to_excel(writer, sheet_name="ME_SL1_SL2_DF", index=False)
        df_b.to_excel(writer, sheet_name="ME_SL1_DF", index=False)


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


def train_example():
    csv_path = "example.csv"
    seq_len = 4
    num_step_types = 5
    num_targets = 2

    _generate_example_csv(csv_path, num_samples=200, seq_len=seq_len, num_targets=num_targets, num_step_types=num_step_types)
    dataset = RecipeDataset(csv_path, seq_len=seq_len, num_step_types=num_step_types)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)


    model = AttentionModel(num_step_types=num_step_types, d_model=32, nhead=4, num_targets=num_targets, seq_len=seq_len)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(5):
        for step_types, knobs, targets in loader:
            optim.zero_grad()
            preds = model(step_types, knobs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optim.step()
        print(f"Epoch {epoch+1} loss: {loss.item():.4f}")


    # visualize positional encodings
    plot_positional_encoding(model.pos_encoder.pe[:seq_len])

    # show attention weights for first batch
    step_types, knobs, _ = next(iter(loader))
    attn = model.attention_heatmap(step_types[0], knobs[0])
    plot_attention_heatmap(attn)
    print("Step importance:", step_importance(attn))


def train_excel_example():
    excel_path = "recipes.xlsx"
    _generate_example_excel(excel_path)
    dataset = ExcelRecipeDataset(excel_path)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = AttentionModel(
        num_step_types=dataset.num_step_types,
        d_model=32,
        nhead=4,
        num_targets=len(dataset.target_cols),
        seq_len=dataset.seq_len,
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(5):
        for step_types, knobs, targets in loader:
            optim.zero_grad()
            preds = model(step_types, knobs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optim.step()
        print(f"Epoch {epoch+1} loss: {loss.item():.4f}")

    plot_positional_encoding(model.pos_encoder.pe[:dataset.seq_len])
    step_types, knobs, _ = next(iter(loader))
    attn = model.attention_heatmap(step_types[0], knobs[0])
    plot_attention_heatmap(attn)
    print("Step importance:", step_importance(attn))


if __name__ == "__main__":
    train_excel_example()

