"""Reconstruction-only baseline models for Step 6."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """Flatten windows from (N, T, F) into (N, T*F)."""
    if X.ndim != 3:
        raise ValueError(f"Expected (N, T, F) windows, got shape {X.shape}")
    return X.reshape(X.shape[0], -1)


def unflatten_windows(X_flat: np.ndarray, window_len: int = 100, feature_dim: int = 40) -> np.ndarray:
    """Unflatten vectors from (N, T*F) into (N, T, F)."""
    if X_flat.ndim != 2:
        raise ValueError(f"Expected (N, T*F) matrix, got shape {X_flat.shape}")
    expected = window_len * feature_dim
    if X_flat.shape[1] != expected:
        raise ValueError(f"Expected second dimension {expected}, got {X_flat.shape[1]}")
    return X_flat.reshape(X_flat.shape[0], window_len, feature_dim)


def compute_compression_ratio(input_dim: int, latent_dim: Optional[int]) -> Optional[float]:
    """Compute input_dim / latent_dim when latent_dim is meaningful."""
    if latent_dim is None or latent_dim <= 0:
        return None
    return float(input_dim) / float(latent_dim)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters for a torch model."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class TrainMeanWindowReconstructor:
    """Reconstruct every sample with the train mean vector in scaled space."""

    def __init__(self):
        self.mean_vec_: Optional[np.ndarray] = None
        self.mean_vector_: Optional[np.ndarray] = None

    def fit(self, X_train_scaled_flat: np.ndarray, X_val_scaled_flat: Optional[np.ndarray] = None) -> None:
        del X_val_scaled_flat
        self.mean_vec_ = X_train_scaled_flat.mean(axis=0)
        self.mean_vector_ = self.mean_vec_

    def reconstruct(self, X_scaled_flat: np.ndarray) -> np.ndarray:
        if self.mean_vec_ is None:
            raise RuntimeError("TrainMeanWindowReconstructor is not fitted.")
        return np.tile(self.mean_vec_[None, :], (X_scaled_flat.shape[0], 1))

    def encode(self, X_scaled_flat: np.ndarray) -> None:
        del X_scaled_flat
        return None


class LastSnapshotRepeatReconstructor:
    """Reconstruct each window by repeating its last snapshot across all timesteps."""

    def __init__(self, window_len: int = 100):
        self.window_len = int(window_len)

    def fit(self, X_train_scaled_flat: np.ndarray, X_val_scaled_flat: Optional[np.ndarray] = None) -> None:
        del X_train_scaled_flat, X_val_scaled_flat
        return None

    def reconstruct(self, X_windows: np.ndarray) -> np.ndarray:
        if X_windows.ndim != 3:
            raise ValueError(f"Expected windows with shape (N, T, F), got {X_windows.shape}")
        last = X_windows[:, -1, :]
        return np.repeat(last[:, None, :], repeats=self.window_len, axis=1)

    def reconstruct_windows(self, X_windows: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for reconstruct()."""
        return self.reconstruct(X_windows)

    def encode(self, X_windows: np.ndarray) -> np.ndarray:
        if X_windows.ndim != 3:
            raise ValueError(f"Expected windows with shape (N, T, F), got {X_windows.shape}")
        return X_windows[:, -1, :]


class PCAReconstructor:
    """PCA reconstructor operating in scaled flattened space."""

    def __init__(self, latent_dim: int, random_state: int = 42):
        self.latent_dim = int(latent_dim)
        self.random_state = int(random_state)
        self.pca = PCA(n_components=self.latent_dim, random_state=self.random_state)
        self.model = self.pca

    def fit(self, X_train_scaled_flat: np.ndarray, X_val_scaled_flat: Optional[np.ndarray] = None) -> None:
        del X_val_scaled_flat
        self.pca.fit(X_train_scaled_flat)

    def reconstruct(self, X_scaled_flat: np.ndarray) -> np.ndarray:
        z = self.pca.transform(X_scaled_flat)
        return self.pca.inverse_transform(z)

    def encode(self, X_scaled_flat: np.ndarray) -> np.ndarray:
        return self.pca.transform(X_scaled_flat)


class _MLPAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


@dataclass
class MLPAEConfig:
    input_dim: int = 4000
    latent_dim: int = 16
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10
    random_state: int = 42
    device: str = "cpu"


class MLPAutoencoderReconstructor:
    """Small MLP autoencoder operating in scaled flattened space."""

    def __init__(self, config: MLPAEConfig):
        self.config = config
        self.model: Optional[_MLPAE] = None
        self.fitted_ = False
        self.train_seconds_: float = 0.0
        self.best_val_loss_: float = float("inf")

    def _set_seed(self) -> None:
        seed = int(self.config.random_state)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fit(self, X_train_scaled_flat: np.ndarray, X_val_scaled_flat: Optional[np.ndarray] = None) -> None:
        if X_val_scaled_flat is None:
            raise ValueError("MLPAutoencoderReconstructor requires validation data for early stopping.")

        self._set_seed()

        train_ds = TensorDataset(
            torch.from_numpy(X_train_scaled_flat.astype(np.float32)),
            torch.from_numpy(X_train_scaled_flat.astype(np.float32)),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val_scaled_flat.astype(np.float32)),
            torch.from_numpy(X_val_scaled_flat.astype(np.float32)),
        )

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

        device = torch.device(self.config.device)
        self.model = _MLPAE(
            input_dim=self.config.input_dim,
            latent_dim=self.config.latent_dim,
            dropout=self.config.dropout,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        best_state = None
        best_val = float("inf")
        stale = 0

        t0 = perf_counter()
        for _ in range(self.config.max_epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    out = self.model(xb)
                    loss = criterion(out, yb)
                    val_sum += float(loss.item()) * len(xb)
                    val_count += len(xb)

            val_loss = val_sum / max(val_count, 1)
            if val_loss < best_val - 1e-9:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.config.patience:
                    break

        self.train_seconds_ = float(perf_counter() - t0)
        self.best_val_loss_ = best_val

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.fitted_ = True

    def reconstruct(self, X_scaled_flat: np.ndarray) -> np.ndarray:
        if not self.fitted_ or self.model is None:
            raise RuntimeError("MLPAutoencoderReconstructor is not fitted.")

        device = torch.device(self.config.device)
        self.model.eval()

        ds = TensorDataset(torch.from_numpy(X_scaled_flat.astype(np.float32)))
        loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle=False)

        out_chunks = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device)
                out = self.model(xb)
                out_chunks.append(out.cpu().numpy())

        return np.vstack(out_chunks)

    def encode(self, X_scaled_flat: np.ndarray) -> np.ndarray:
        if not self.fitted_ or self.model is None:
            raise RuntimeError("MLPAutoencoderReconstructor is not fitted.")

        device = torch.device(self.config.device)
        self.model.eval()

        ds = TensorDataset(torch.from_numpy(X_scaled_flat.astype(np.float32)))
        loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle=False)

        z_chunks = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device)
                z = self.model.encoder(xb)
                z_chunks.append(z.cpu().numpy())

        return np.vstack(z_chunks)
