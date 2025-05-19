import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


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
    def __init__(self, num_step_types: int, d_model: int, nhead: int, num_targets: int):
        super().__init__()
        self.embedding = nn.Embedding(num_step_types, d_model)
        self.knob_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, num_targets)

    def forward(self, step_types: torch.Tensor, knobs: torch.Tensor):
        # step_types: (B, L)
        # knobs: (B, L)
        emb = self.embedding(step_types) + self.knob_proj(knobs.unsqueeze(-1))
        emb = emb.transpose(0, 1)  # (L, B, D)
        enc = self.encoder(emb)
        enc = enc.mean(dim=0)  # (B, D)
        return self.fc(enc)


def train_example():
    csv_path = "example.csv"
    seq_len = 4
    num_step_types = 5
    num_targets = 2

    _generate_example_csv(csv_path, num_samples=200, seq_len=seq_len, num_targets=num_targets, num_step_types=num_step_types)
    dataset = RecipeDataset(csv_path, seq_len=seq_len, num_step_types=num_step_types)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = AttentionModel(num_step_types=num_step_types, d_model=32, nhead=4, num_targets=num_targets)
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


if __name__ == "__main__":
    train_example()
