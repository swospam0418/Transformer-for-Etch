# coding: utf-8
"""Utilities for training transformer models on etch recipes.

This module groups functions by purpose so that the workflow is easier to
understand. It is mostly a refactoring of the original script into logical
sections: data processing, model definition, training utilities and analysis
helpers.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import List, Dict, Tuple, Sequence, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader
import seaborn as sns

__all__ = [
    "read_merge_sheets",
    "EtchDataProcessor",
    "EtchDataset",
    "collate_fn",
    "SingleEtchDataset",
    "collate_fn_single",
    "PositionalEncoding",
    "EtchTransformer",
    "ModelTrainer",
    "single_recipe_predict",
    "evaluate_predictions",
    "check_architectures",
    "inspect_step_and_position_encoding",
    "visualize_step_type_attention",
    "move_step_to_positions_and_plot",
    "change_knob_and_plot",
]

# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def read_merge_sheets(file_path: str) -> pd.DataFrame:
    """Read all sheets in an Excel workbook and concatenate them."""
    xls = pd.ExcelFile(file_path)
    merged = [
        pd.read_excel(file_path, sheet_name=s).assign(SHEET_NAME=s)
        for s in xls.sheet_names
    ]
    return pd.concat(merged, ignore_index=True)


class EtchDataProcessor:
    """Parse column names, standardise values and build sequences."""

    def __init__(self) -> None:
        self.step_types: Dict[str, int] = {}
        self.tuning_knobs: Dict[str, List[str]] = {}
        self.step_param_indices: Dict[str, List[int]] = {}
        self.all_tuning_cols: List[str] = []
        self.all_profile_cols: List[str] = []
        self.x_col_stats: Dict[str, Tuple[float, float]] = {}
        self.y_col_stats: Dict[str, Tuple[float, float]] = {}

    def parse_columns(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            if col.startswith("X_"):
                parts = col.split("_")
                step_type = parts[1] if len(parts) > 1 else "UNKNOWN"
                knob = "_".join(parts[2:]) if len(parts) > 2 else "default"
                if step_type not in self.step_types:
                    self.step_types[step_type] = len(self.step_types)
                    self.tuning_knobs[step_type] = []
                if knob not in self.tuning_knobs[step_type]:
                    self.tuning_knobs[step_type].append(knob)
                if col not in self.all_tuning_cols:
                    self.all_tuning_cols.append(col)
            elif col.startswith("Y_") and col not in self.all_profile_cols:
                self.all_profile_cols.append(col)

    def build_indices(self) -> None:
        col_idx_map = {c: i for i, c in enumerate(self.all_tuning_cols)}
        for st, knobs in self.tuning_knobs.items():
            idxs = []
            for knob in knobs:
                name = f"X_{st}" if knob == "default" else f"X_{st}_{knob}"
                if name in col_idx_map:
                    idxs.append(col_idx_map[name])
            self.step_param_indices[st] = idxs

    def fit_statistics(self, df: pd.DataFrame) -> None:
        eps = 1e-8
        for col in self.all_tuning_cols:
            mean_val, std_val = (df[col].dropna().mean(), df[col].dropna().std()) if col in df.columns else (0.0, 1.0)
            self.x_col_stats[col] = (mean_val, std_val if std_val >= eps else 1.0)
        for col in self.all_profile_cols:
            mean_val, std_val = (df[col].dropna().mean(), df[col].dropna().std()) if col in df.columns else (0.0, 1.0)
            self.y_col_stats[col] = (mean_val, std_val if std_val >= eps else 1.0)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()
        for col in self.all_tuning_cols:
            mean, std = self.x_col_stats[col]
            df_new[col] = ((df_new.get(col, 0.0)).fillna(0.0) - mean) / std
        for col in self.all_profile_cols:
            mean, std = self.y_col_stats[col]
            df_new[col] = ((df_new.get(col, 0.0)).fillna(0.0) - mean) / std
        return df_new

    def build_sequences_and_profiles(self, df: pd.DataFrame) -> Tuple[List[List[Tuple[int, np.ndarray]]], np.ndarray]:
        profiles = df[self.all_profile_cols].values
        X_matrix = df[self.all_tuning_cols].values
        param_dim = len(self.all_tuning_cols)
        sequences: List[List[Tuple[int, np.ndarray]]] = []
        for row in X_matrix:
            recipe_seq = []
            for st, st_idx in self.step_types.items():
                idxs = self.step_param_indices[st]
                if not idxs:
                    continue
                param_vec = np.zeros(param_dim, dtype=float)
                param_vec[idxs] = row[idxs]
                if not np.all(param_vec == 0):
                    recipe_seq.append((st_idx, param_vec))
            recipe_seq.sort(key=lambda x: x[0])
            sequences.append(recipe_seq)
        return sequences, profiles

    def unscale_profile(self, profile_vec: np.ndarray) -> np.ndarray:
        unscaled = np.zeros_like(profile_vec)
        for d, col in enumerate(self.all_profile_cols):
            mean, std = self.y_col_stats[col]
            unscaled[d] = profile_vec[d] * std + mean
        return unscaled


class EtchDataset(Dataset):
    """Dataset for a list of sequences and profile vectors."""

    def __init__(self, sequences: Sequence, profiles: np.ndarray):
        self.sequences = list(sequences)
        self.profiles = profiles

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.profiles[idx]


def collate_fn(batch: List[Any]):
    sequences, profiles = zip(*batch)
    batch_size = len(sequences)
    max_len = max(len(s) for s in sequences) or 1
    param_dim = len(sequences[0][0][1]) if sequences[0] else 1

    step_seq = torch.zeros((batch_size, max_len), dtype=torch.long)
    param_seq = torch.zeros((batch_size, max_len, param_dim), dtype=torch.float)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, seq in enumerate(sequences):
        for j, (stype_idx, param_vec) in enumerate(seq):
            step_seq[i, j] = stype_idx
            param_seq[i, j] = torch.from_numpy(param_vec)
        mask[i, : len(seq)] = True

    profiles_t = torch.tensor(profiles, dtype=torch.float)
    return {"step_seq": step_seq, "param_seq": param_seq, "mask": mask, "profile": profiles_t}


class SingleEtchDataset(Dataset):
    def __init__(self, seq: List[Tuple[int, np.ndarray]], profile: np.ndarray):
        self.seq, self.profile = seq, profile

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.seq, self.profile


def collate_fn_single(batch):
    seq, profile = batch[0]
    L = max(len(seq), 1)
    param_dim = len(seq[0][1]) if seq else 1

    step_seq = torch.zeros((L,), dtype=torch.long)
    param_seq = torch.zeros((L, param_dim), dtype=torch.float)
    mask = torch.zeros((L,), dtype=torch.bool)

    for j, (stype_idx, param_vec) in enumerate(seq):
        step_seq[j] = stype_idx
        param_seq[j] = torch.from_numpy(param_vec)
        mask[j] = True

    profile_t = torch.tensor(profile[None, :], dtype=torch.float)
    return {"step_seq": step_seq.unsqueeze(0), "param_seq": param_seq.unsqueeze(0), "mask": mask.unsqueeze(0), "profile": profile_t}


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class EtchTransformer(nn.Module):
    """Transformer encoder with a trainable CLS token."""

    def __init__(
        self,
        num_step_types: int,
        param_dim: int,
        profile_dim: int,
        d_model: int = 256,
        n_head: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.step_embedding = nn.Embedding(num_step_types, d_model)
        self.param_projection = nn.Linear(param_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.xavier_uniform_(self.cls_token)
        self.pre_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=512, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, profile_dim),
        )

    def forward(self, step_seq: torch.Tensor, param_seq: torch.Tensor, mask: torch.Tensor | None = None, inspect: bool = False) -> torch.Tensor | Tuple:
        B, L, _ = param_seq.shape
        x = self.step_embedding(step_seq) + self.param_projection(param_seq)
        x = x / math.sqrt(2)
        cls_tok = self.cls_token.expand(B, 1, self.d_model)
        x = torch.cat([cls_tok, x], dim=1)
        if mask is not None:
            cls_mask = torch.ones((B, 1), dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
        x = self.pre_norm(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=~mask if mask is not None else None)
        pooled = x[:, 0]
        out = self.output_layer(pooled)
        if inspect:
            return out, x
        return out


def patch_mha_record_weights(mha: nn.MultiheadAttention):
    """Record attention weights on the module instance for later inspection."""

    if getattr(mha, "_record_patch_done", False):
        return

    orig_forward = mha.forward

    def forward_and_record(query, key, value, *args, **kwargs):
        kwargs.update(need_weights=True, average_attn_weights=False)
        attn_output, attn_weights = orig_forward(query, key, value, *args, **kwargs)
        mha.last_attn_weights = attn_weights.detach()
        return attn_output

    mha.forward = forward_and_record
    mha._record_patch_done = True


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

class ModelTrainer:
    def __init__(self, model: nn.Module, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self._init_history()

    def _init_history(self) -> None:
        self.train_losses, self.val_losses = [], []
        self.train_r2s, self.val_r2s = [], []

    def train_and_validate(self, train_ds: Dataset, val_ds: Dataset, batch_size: int = 32, epochs: int = 100, lr: float = 1e-4) -> None:
        train_loader = DataLoader(train_ds, batch_size, True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size, False, collate_fn=collate_fn)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=5)
        criterion = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            self.model.train()
            tr_loss, preds, trues = 0.0, [], []
            for batch in train_loader:
                optim.zero_grad()
                out = self.model(batch["step_seq"].to(self.device), batch["param_seq"].to(self.device), batch["mask"].to(self.device))
                loss = criterion(out, batch["profile"].to(self.device))
                loss.backward()
                optim.step()
                scheduler.step()
                tr_loss += loss.item()
                preds.append(out.detach().cpu().numpy())
                trues.append(batch["profile"].numpy())

            tr_loss /= len(train_loader)
            tr_r2 = r2_score(np.concatenate(trues), np.concatenate(preds))

            self.model.eval()
            vl_loss, vpreds, vtrues = 0.0, [], []
            with torch.no_grad():
                for batch in val_loader:
                    out = self.model(batch["step_seq"].to(self.device), batch["param_seq"].to(self.device), batch["mask"].to(self.device))
                    loss = criterion(out, batch["profile"].to(self.device))
                    vl_loss += loss.item()
                    vpreds.append(out.cpu().numpy())
                    vtrues.append(batch["profile"].numpy())

            vl_loss /= len(val_loader)
            vl_r2 = r2_score(np.concatenate(vtrues), np.concatenate(vpreds))

            self.train_losses.append(tr_loss)
            self.val_losses.append(vl_loss)
            self.train_r2s.append(tr_r2)
            self.val_r2s.append(vl_r2)

            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"[Epoch {epoch:3d}/{epochs}] "
                    f"Train Loss={tr_loss:.4f} R²={tr_r2:.4f} | "
                    f"Val Loss={vl_loss:.4f} R²={vl_r2:.4f}"
                )

    def plot_training_history(self) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(self.train_losses, label="Train")
        ax1.plot(self.val_losses, label="Val")
        ax1.set_title("MSE Loss")
        ax1.legend()
        ax2.plot(self.train_r2s, label="Train R²")
        ax2.plot(self.val_r2s, label="Val R²")
        ax2.set_title("R² Score")
        ax2.legend()
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------


def single_recipe_predict(model: nn.Module, seq: List[Tuple[int, np.ndarray]], processor: EtchDataProcessor, device: str) -> np.ndarray:
    if not seq:
        return np.zeros((len(processor.all_profile_cols),), dtype=float)
    L, param_dim = len(seq), len(seq[0][1])
    step_seq = torch.tensor([s[0] for s in seq], dtype=torch.long, device=device).unsqueeze(0)
    param_seq = torch.tensor([s[1] for s in seq], dtype=torch.float, device=device).unsqueeze(0)
    mask = torch.ones((1, L), dtype=torch.bool, device=device)
    with torch.no_grad():
        out = model(step_seq, param_seq, mask)
    return out.squeeze(0).cpu().numpy()


def evaluate_predictions(model: nn.Module, loader: DataLoader, processor: EtchDataProcessor, device: str, max_samples: int = 15):
    model.eval()
    preds, acts = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["step_seq"].to(device), batch["param_seq"].to(device), batch["mask"].to(device))
            preds.append(out.cpu().numpy())
            acts.append(batch["profile"].numpy())
    preds, acts = np.concatenate(preds), np.concatenate(acts)
    unscale = lambda arr: np.stack([arr[:, d] * processor.y_col_stats[c][1] + processor.y_col_stats[c][0] for d, c in enumerate(processor.all_profile_cols)], axis=1)
    preds_u, acts_u = unscale(preds), unscale(acts)
    for i in range(min(max_samples, len(preds_u))):
        r2 = r2_score(acts_u[i], preds_u[i])
        print(f"\nSample {i+1}: R2={r2:.4f}")
        plt.figure(figsize=(12, 4))
        plt.plot(acts_u[i], label="Actual", color="blue")
        plt.plot(preds_u[i], label="Predicted", color="red", linestyle="--")
        plt.title(f"Sample {i+1} Prediction vs Actual")
        plt.legend()
        plt.show()
    return preds_u, acts_u


# ---------------------------------------------------------------------------
# Additional interpretability helpers
# ---------------------------------------------------------------------------


def check_architectures(df: pd.DataFrame, arch_col: str, sequences, profiles, train_idx, test_idx, processor: EtchDataProcessor, model: nn.Module, device: str, max_samples: int = 5):
    uniques = df[arch_col].unique()
    for arch in uniques:
        print("\n", "-" * 20, f"Architecture = {arch}", "-" * 20)
        mask = df[arch_col] == arch
        arch_idx = np.where(mask)[0]
        tr = list(set(arch_idx) & set(train_idx))[:max_samples]
        te = list(set(arch_idx) & set(test_idx))[:max_samples]
        if tr:
            print("[Train]")
            _evaluate_and_plot_samples(tr, sequences, profiles, processor, model, device)
        if te:
            print("[Test]")
            _evaluate_and_plot_samples(te, sequences, profiles, processor, model, device)


def _evaluate_and_plot_samples(idxs, seqs, profs, proc, mdl, device):
    for i_idx in idxs:
        ds = SingleEtchDataset(seqs[i_idx], profs[i_idx])
        dl = DataLoader(ds, 1, collate_fn=collate_fn_single)
        with torch.no_grad():
            for b in dl:
                out = mdl(b["step_seq"].to(device), b["param_seq"].to(device), b["mask"].to(device)).cpu().numpy()
        pred_u, act_u = _unscale_y(out, b["profile"].numpy(), proc)
        plt.figure(figsize=(10, 4))
        plt.plot(act_u.flatten(), label="Actual", color="blue")
        plt.plot(pred_u.flatten(), label="Predicted", color="red", linestyle="--")
        plt.title(f"Sample idx={i_idx}  - Profile Comparison")
        plt.legend()
        plt.show()


def _unscale_y(pred, true, proc):
    d = pred.shape[1]
    un_p, un_t = np.zeros_like(pred), np.zeros_like(true)
    for k, col in enumerate(proc.all_profile_cols[:d]):
        m, s = proc.y_col_stats[col]
        un_p[:, k], un_t[:, k] = pred[:, k] * s + m, true[:, k] * s + m
    return un_p, un_t


def inspect_step_and_position_encoding(model, proc, sequences, idxs=[0], device="cpu"):
    print("\n=== (1) Step / Position Encoding 檢查 ===")
    for idx in idxs:
        seq = sequences[idx]
        names = [list(proc.step_types.keys())[s[0]] for s in seq]
        print(f"Recipe {idx}:", list(zip(range(1, len(seq) + 1), names)))
        step_t = torch.tensor([s[0] for s in seq], dtype=torch.long).unsqueeze(0)
        param_t = torch.tensor([s[1] for s in seq], dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            step_e = model.step_embedding(step_t)
            pos_e = model.pos_encoder(step_e + model.param_projection(param_t))
        for i, n in enumerate(names):
            print(f"  {i+1:2d}. {n:8s} step_emb[:5] {step_e[0,i,:5].numpy().round(3)}")
            print(f"       pos_out[:5] {pos_e[0,i,:5].numpy().round(3)}")


def extract_attention_weights(model, step_seq, param_seq, mask, device):
    model.eval()
    with torch.no_grad():
        model(step_seq.to(device), param_seq.to(device), mask.to(device))
    weights = []
    for lyr in model.transformer.layers:
        if not hasattr(lyr.self_attn, "last_attn_weights"):
            raise RuntimeError("未找到 last_attn_weights；請確認已 patch_mha_record_weights()")
        weights.append(lyr.self_attn.last_attn_weights.cpu())
    return weights


def visualize_step_type_attention(model, proc, seqs, sample_index: int = 0, layer_idx: int = 0, head_idx: int = 0, device: str = "cpu"):
    print("\n=== Step Type Attention ===")
    if sample_index >= len(seqs):
        print("sample_index is out of range")
        return
    seq = seqs[sample_index]
    step_t = torch.tensor([s[0] for s in seq], dtype=torch.long).unsqueeze(0)
    param_t = torch.tensor([s[1] for s in seq], dtype=torch.float).unsqueeze(0)
    mask_t = torch.ones_like(step_t, dtype=torch.bool)
    attn_layers = extract_attention_weights(model, step_t, param_t, mask_t, device)
    if layer_idx >= len(attn_layers):
        print(f"layer_idx={layer_idx} 超出層數範圍 (共有 {len(attn_layers)})")
        return
    A_full = attn_layers[layer_idx][0, head_idx].numpy()
    A = A_full[1:, 1:]
    labels = [list(proc.step_types.keys())[s[0]] for s in seq]
    sns.heatmap(A, cmap="Blues", xticklabels=labels, yticklabels=labels, square=True, cbar_kws={"label": "Attention"})
    plt.title(f"Layer{layer_idx}-Head{head_idx} Sample {sample_index}")
    plt.xlabel("Key positions (step type)")
    plt.ylabel("Query positions (step type)")
    plt.show()


def move_step_to_positions_and_plot(model, proc, seq, device, step_idx_to_move=0, positions_to_try=[0, 1, 2], actual_profile=None):
    curves = {"Baseline": proc.unscale_profile(single_recipe_predict(model, seq, proc, device))}
    if actual_profile is not None:
        curves["Actual"] = actual_profile
    for pos in positions_to_try:
        modified = deepcopy(seq)
        step = modified.pop(step_idx_to_move)
        modified.insert(pos, step)
        curves[f"Pos {pos+1}"] = proc.unscale_profile(single_recipe_predict(model, modified, proc, device))
    plt.figure(figsize=(8, 4))
    for k, v in curves.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.title("Move Step What-If")
    plt.show()


def change_knob_and_plot(model, proc, seq, device, step_idx=0, knob_idx=0, knob_values=[-200, 0, +200], baseline_offset=1200, actual_profile=None):
    curves = {"Baseline": proc.unscale_profile(single_recipe_predict(model, seq, proc, device))}
    if actual_profile is not None:
        curves["Actual"] = actual_profile
    col_name = proc.all_tuning_cols[knob_idx]
    mean, std = proc.x_col_stats[col_name]
    for delta in knob_values:
        mod = deepcopy(seq)
        vec = mod[step_idx][1].copy()
        vec[knob_idx] = ((baseline_offset + delta) - mean) / std
        mod[step_idx] = (mod[step_idx][0], vec)
        curves[f"{baseline_offset+delta}"] = proc.unscale_profile(single_recipe_predict(model, mod, proc, device))
    plt.figure(figsize=(8, 4))
    for k, v in curves.items():
        plt.plot(v, label=k)
    plt.title("Knob Sweep What-if")
    plt.legend()
    plt.show()


def swap_steps_and_plot(model, proc, seq, device, idx_a: int, idx_b: int, actual_profile: np.ndarray | None = None, title: str | None = None):
    curves = {"Baseline": proc.unscale_profile(single_recipe_predict(model, seq, proc, device))}
    if actual_profile is not None:
        curves["Actual"] = actual_profile
    swapped = deepcopy(seq)
    swapped[idx_a], swapped[idx_b] = swapped[idx_b], swapped[idx_a]
    curves[f"Swap {idx_a+1}↔{idx_b+1}"] = proc.unscale_profile(single_recipe_predict(model, swapped, proc, device))
    plt.figure(figsize=(10, 4))
    for k, v in curves.items():
        plt.plot(v, label=k)
    plt.title(title or "Swap-Step What-if")
    plt.legend()
    plt.show()


def integrated_gradients_attribution(model: nn.Module, seq: List[Tuple[int, np.ndarray]], processor: EtchDataProcessor, device: str = "cpu", baseline: str = "zeros", steps: int = 50):
    if not seq:
        raise ValueError("Sequence empty – nothing to attribute.")
    model.eval()
    L, param_dim = len(seq), len(seq[0][1])
    step_t = torch.tensor([s[0] for s in seq], dtype=torch.long, device=device)
    param_t = torch.tensor([s[1] for s in seq], dtype=torch.float, device=device)
    if baseline == "zeros":
        base_param = torch.zeros_like(param_t)
    elif baseline == "mean":
        base_param = torch.zeros_like(param_t)
        for d in range(param_dim):
            col = processor.all_tuning_cols[d]
            mu, sig = processor.x_col_stats[col]
            base_param[:, d] = (0.0 - mu) / sig
    else:
        raise ValueError("baseline must be 'zeros' or 'mean'")
    alphas = torch.linspace(0, 1, steps + 1, device=device).view(-1, 1, 1)
    param_interp = (base_param.unsqueeze(0) + alphas * (param_t.unsqueeze(0) - base_param.unsqueeze(0))).clone().detach().requires_grad_(True)
    mask = torch.ones((1, L), dtype=torch.bool, device=device)
    accumulated = torch.zeros_like(param_t)
    for i in range(steps + 1):
        out = model(step_t.unsqueeze(0), param_interp[i].unsqueeze(0), mask)
        scalar = out.norm()
        scalar.backward(retain_graph=True)
        accumulated += param_interp.grad[i]
        param_interp.grad.zero_()
    avg_grads = accumulated / (steps + 1)
    attr_param = (param_t - base_param) * avg_grads
    attr_step = attr_param.norm(dim=1)
    return attr_step.cpu().numpy(), attr_param.cpu().numpy()


def permutation_importance(model: nn.Module, dataset: Dataset, processor: EtchDataProcessor, device: str = "cpu", n_repeats: int = 5):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    def _score(dl):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for b in dl:
                out = model(b["step_seq"].to(device), b["param_seq"].to(device), b["mask"].to(device))
                preds.append(out.cpu().numpy())
                trues.append(b["profile"].numpy())
        return r2_score(np.concatenate(trues), np.concatenate(preds))

    baseline_r2 = _score(loader)
    print(f"Baseline R² = {baseline_r2:.4f}")
    results = []
    for col_idx, col_name in enumerate(processor.all_tuning_cols):
        drops = []
        for _ in range(n_repeats):
            perm_seqs = []
            values = [vec[col_idx] for seq in dataset.sequences for _stype, vec in seq]
            np.random.shuffle(values)
            v_iter = iter(values)
            for seq in dataset.sequences:
                new_seq = []
                for stype, vec in seq:
                    new_vec = vec.copy()
                    new_vec[col_idx] = next(v_iter)
                    new_seq.append((stype, new_vec))
                perm_seqs.append(new_seq)
            perm_ds = EtchDataset(perm_seqs, dataset.profiles)
            perm_r2 = _score(DataLoader(perm_ds, 64, False, collate_fn=collate_fn))
            drops.append(baseline_r2 - perm_r2)
        results.append((col_name, np.mean(drops), np.std(drops)))
    df_imp = pd.DataFrame(results, columns=["knob", "mean_drop", "std"])
    df_imp.sort_values("mean_drop", ascending=False, inplace=True, ignore_index=True)
    print("\nPermutation Importance (Top‑20 knobs):")
    print(df_imp.head(20))
    return df_imp

