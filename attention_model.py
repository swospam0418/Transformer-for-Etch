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


def _generate_example_csv(path: str, num_samples=100, seq_len=4, num_targets=2, num_step_types=5):
    rng = np.random.default_rng(0)
    data = {}
    for i in range(seq_len):
        data[f"step_type_{i}"] = rng.integers(0, num_step_types, size=num_samples)
        data[f"knob_{i}"] = rng.random(size=num_samples)
    for t in range(num_targets):
        data[f"target_{t}"] = rng.random(size=num_samples)
    pd.DataFrame(data).to_csv(path, index=False)


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


if __name__ == "__main__":
    train_example()
