import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from attention_model import PositionalEncoding


class EtchRateDataset(Dataset):
    """Dataset including film scheme/layout info and etch rate targets."""

    def __init__(self, csv_path: str, seq_len: int):
        self.data = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.step_cols = [f"step_type_{i}" for i in range(seq_len)]
        self.knob_cols = [f"knob_{i}" for i in range(seq_len)]
        self.scheme_col = "scheme_id"
        self.layout_col = "layout_id"
        self.target_cols = [c for c in self.data.columns if c.startswith("target_")]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        step_types = row[self.step_cols].to_numpy(dtype=np.int64)
        knobs = row[self.knob_cols].to_numpy(dtype=np.float32)
        scheme = np.int64(row[self.scheme_col])
        layout = np.int64(row[self.layout_col])
        targets = row[self.target_cols].to_numpy(dtype=np.float32)
        return (
            torch.from_numpy(step_types),
            torch.from_numpy(knobs),
            torch.tensor(scheme),
            torch.tensor(layout),
            torch.from_numpy(targets),
        )






class EtchRateTransformer(nn.Module):
    """Transformer model for etch rate prediction with scheme/layout embeddings."""

    def __init__(
        self,
        num_step_types: int,
        num_schemes: int,
        num_layouts: int,
        d_model: int,
        nhead: int,
        num_targets: int,
        seq_len: int,
    ):
        super().__init__()
        self.step_embed = nn.Embedding(num_step_types, d_model)
        self.knob_proj = nn.Linear(1, d_model)
        self.scheme_embed = nn.Embedding(num_schemes, d_model)
        self.layout_embed = nn.Embedding(num_layouts, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, num_targets)

    def forward(
        self,
        step_types: torch.Tensor,
        knobs: torch.Tensor,
        scheme: torch.Tensor,
        layout: torch.Tensor,
    ) -> torch.Tensor:
        step_emb = self.step_embed(step_types) + self.knob_proj(knobs.unsqueeze(-1))
        scheme_emb = self.scheme_embed(scheme).unsqueeze(1)
        layout_emb = self.layout_embed(layout).unsqueeze(1)
        x = step_emb + scheme_emb + layout_emb
        x = self.pos_encoder(x)
        enc = self.encoder(x)
        return self.fc(enc.mean(dim=1))


if __name__ == "__main__":
    print("etch_rate_model.py provides the EtchRateTransformer class for advanced use cases.")
