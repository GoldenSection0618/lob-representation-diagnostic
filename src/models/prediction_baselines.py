"""Prediction-only baseline models for Step 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CLASS_ORDER = [0, 1, 2]


def flatten_windows(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected 3D windows (N, T, F), got shape {X.shape}")
    return X.reshape(X.shape[0], -1)


def compute_class_weights(y_train: np.ndarray, class_order=None) -> np.ndarray:
    class_order = class_order or CLASS_ORDER
    counts = np.array([(y_train == c).sum() for c in class_order], dtype=np.float64)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (len(class_order) * counts)
    return weights.astype(np.float32)


class MajorityBaseline:
    """Always predicts the majority class from training labels."""

    def __init__(self, class_order=None):
        self.class_order = class_order or CLASS_ORDER
        self.majority_class_: Optional[int] = None
        self.class_prior_: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "MajorityBaseline":
        del X_train
        counts = np.array([(y_train == c).sum() for c in self.class_order], dtype=np.float64)
        total = counts.sum()
        if total <= 0:
            raise ValueError("Empty y_train in MajorityBaseline.fit")
        self.class_prior_ = counts / total
        self.majority_class_ = int(self.class_order[int(np.argmax(counts))])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.majority_class_ is None:
            raise RuntimeError("MajorityBaseline must be fitted before predict.")
        return np.full(shape=(len(X),), fill_value=self.majority_class_, dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.class_prior_ is None:
            raise RuntimeError("MajorityBaseline must be fitted before predict_proba.")
        return np.tile(self.class_prior_[None, :], (len(X), 1))


class LogisticRegressionBaseline:
    """Flattened-window logistic regression with train-only standardization."""

    def __init__(self, random_state: int = 42, max_iter: int = 2000):
        self.random_state = random_state
        self.max_iter = max_iter
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=self.max_iter,
                        class_weight="balanced",
                        random_state=self.random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegressionBaseline":
        self.pipeline.fit(flatten_windows(X_train), y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(flatten_windows(X)).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(flatten_windows(X)).astype(np.float64)


class _MLPNet(nn.Module):
    def __init__(self, input_dim: int = 4000, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MLPTrainConfig:
    random_state: int = 42
    max_epochs: int = 100
    batch_size: int = 256
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    device: str = "cpu"


class MLPBaseline:
    """Small MLP baseline with train-only scaling and early stopping."""

    def __init__(self, config: MLPTrainConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.model: Optional[_MLPNet] = None
        self._fitted = False

    def _set_seed(self) -> None:
        seed = self.config.random_state
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "MLPBaseline":
        self._set_seed()

        X_train_f = flatten_windows(X_train)
        X_val_f = flatten_windows(X_val)

        X_train_s = self.scaler.fit_transform(X_train_f).astype(np.float32)
        X_val_s = self.scaler.transform(X_val_f).astype(np.float32)

        train_ds = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train.astype(np.int64)))
        val_ds = TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val.astype(np.int64)))

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

        device = torch.device(self.config.device)
        self.model = _MLPNet(input_dim=X_train_s.shape[1], num_classes=3, dropout=self.config.dropout).to(device)

        class_weights = compute_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        best_val_loss = float("inf")
        best_state = None
        stale_epochs = 0

        for _epoch in range(self.config.max_epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_loss_total = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    val_loss_total += float(loss.item()) * len(xb)
                    val_count += len(xb)

            val_loss = val_loss_total / max(val_count, 1)
            if val_loss < best_val_loss - 1e-7:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._fitted = True
        return self

    def _prepare_input(self, X: np.ndarray) -> torch.Tensor:
        if not self._fitted or self.model is None:
            raise RuntimeError("MLPBaseline must be fitted before inference.")
        X_f = flatten_windows(X)
        X_s = self.scaler.transform(X_f).astype(np.float32)
        return torch.from_numpy(X_s)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        xt = self._prepare_input(X)
        device = torch.device(self.config.device)
        self.model.eval()

        probs = []
        with torch.no_grad():
            for i in range(0, len(xt), self.config.batch_size):
                xb = xt[i : i + self.config.batch_size].to(device)
                logits = self.model(xb)
                p = torch.softmax(logits, dim=1)
                probs.append(p.cpu().numpy())
        return np.vstack(probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1).astype(np.int64)
