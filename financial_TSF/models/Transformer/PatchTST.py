import copy
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PatchTSTConfig, PatchTSTForRegression

from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.base_model import BaseModel

# default hyper-parameters 

PATCHTST_PARAMS: dict[str, Any] = {
    "context_length": 128,       # lookback window intervals
    "patch_length": 4,          # length of each patch
    "d_model": 64,              # transformer hidden dimension
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "ffn_dim": 128,
    "dropout": 0.2,
    "head_dropout": 0.2,
    "batch_size": 64,
    "lr": 1e-4,
    "num_epochs": 100,
    "patience": 30,             # early-stopping patience (epochs)
}


# sliding-window dataset 

class _LazySlidingWindowDataset(Dataset):
    """
    Memory-efficient dataset that builds sliding windows **on-the-fly**
    in __getitem__ instead of pre-materialising all (N, ctx, F) arrays.

    Each ticker segment is stored as a 2-D array; a global index is
    mapped to (ticker_idx, local_row) at access time.
    """

    def __init__(
        self,
        segments: list[np.ndarray],
        targets: list[np.ndarray],
        context_length: int,
    ):
        """
        Parameters
        ----------
        segments : list of np.ndarray, each (T_i, F)
            Per-ticker feature arrays (already concatenated with any
            look-back prefix, so the window can always be read directly).
        targets : list of np.ndarray, each (N_i, num_targets)
            Per-ticker target arrays (length = number of *actual* rows,
            which may be shorter than T_i when a prefix was prepended).
        context_length : int
            Sliding-window size.
        """
        self.segments = segments
        self.targets = targets
        self.context_length = context_length

        # build an index map: global_idx -> (segment_idx, local_row)
        # local_row refers to the target row; the window is read from
        # the segment ending at (offset + local_row).
        self._offsets: list[int] = []          # prefix offset per segment
        self._lengths: list[int] = []          # number of target rows
        self._cum_lengths: list[int] = []      # cumulative lengths

        total = 0
        for seg, tgt in zip(segments, targets):
            n_targets = len(tgt)
            offset = len(seg) - n_targets       # prefix rows
            self._offsets.append(offset)
            self._lengths.append(n_targets)
            total += n_targets
            self._cum_lengths.append(total)

        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        # binary-search for the right segment
        seg_idx = self._find_segment(idx)
        local_row = idx - (self._cum_lengths[seg_idx] - self._lengths[seg_idx])

        seg = self.segments[seg_idx]
        offset = self._offsets[seg_idx]
        abs_idx = offset + local_row            # position inside segment

        start = max(0, abs_idx - self.context_length + 1)
        window = seg[start: abs_idx + 1]        # (<=ctx, F)

        # left-zero-pad when not enough history
        if len(window) < self.context_length:
            pad = np.zeros(
                (self.context_length - len(window), seg.shape[1]),
                dtype=np.float32,
            )
            window = np.concatenate([pad, window], axis=0)

        target = self.targets[seg_idx][local_row]
        return torch.from_numpy(window), torch.from_numpy(target)

    # -- helpers --------------------------------------------------------
    def _find_segment(self, idx: int) -> int:
        """Return segment index for a global row index (binary search)."""
        lo, hi = 0, len(self._cum_lengths) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cum_lengths[mid] <= idx:
                lo = mid + 1
            else:
                hi = mid
        return lo



class PatchTSTWrapper(BaseModel):
    """
    Pipeline-compatible wrapper around HuggingFace PatchTSTForRegression.
    """

    def __init__(self, config: AppConfig, model_config: dict | None = None):
        self.config = config
        mc = model_config or PATCHTST_PARAMS

        self.num_targets: int = len(config.settings.forecast_horizon)

        # hyper-parameters
        self.context_length: int = mc.get("context_length", 32)
        self.patch_length: int   = mc.get("patch_length", 8)
        self.d_model: int        = mc.get("d_model", 64)
        self.num_heads: int      = mc.get("num_attention_heads", 4)
        self.num_layers: int     = mc.get("num_hidden_layers", 2)
        self.ffn_dim: int        = mc.get("ffn_dim", 128)
        self.dropout: float      = mc.get("dropout", 0.2)
        self.head_dropout: float = mc.get("head_dropout", 0.2)
        self.batch_size: int     = mc.get("batch_size", 64)
        self.lr: float           = mc.get("lr", 1e-4)
        self.num_epochs: int     = mc.get("num_epochs", 100)
        self.patience: int       = mc.get("patience", 10)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: PatchTSTForRegression | None = None

        print(f"[PatchTST] device={self.device}  context_length={self.context_length}  "
              f"patch_length={self.patch_length}  num_targets={self.num_targets}")


    @staticmethod
    def _build_windows(X: np.ndarray, context_size: int) -> np.ndarray:
        """
        Build (N, context_size, F) windows from an (N, F) feature matrix
        The first rows that lack enough history are zero-padded from left
        """
        N, F = X.shape
        out = np.zeros((N, context_size, F), dtype=np.float32)
        for i in range(N):
            start = max(0, i - context_size + 1)
            w = X[start: i + 1]
            out[i, context_size - len(w):] = w
        return out

    @staticmethod
    def _build_windows_with_prefix(
        prefix: np.ndarray, X: np.ndarray, context_size: int,
    ) -> np.ndarray:
        """
        Build windows for X using the tail of prefix dataset as additional
        lookback so data prior splitpoint can be used and has not to be padded with zero
        """
        tail = prefix[-(context_size - 1):] if len(prefix) >= context_size - 1 else prefix
        combined = np.concatenate([tail, X], axis=0)
        offset = len(tail)  # X[0] lives at index offset in combined

        N, F = X.shape
        out = np.zeros((N, context_size, F), dtype=np.float32)
        for i in range(N):
            idx = offset + i
            start = max(0, idx - context_size + 1)
            w = combined[start: idx + 1]
            out[i, context_size - len(w):] = w
        return out

    def _make_dataset(
        self,
        split_data: list[dict],
        x_key: str,
        y_key: str,
        prefix_x_key: str | None = None,
    ) -> _LazySlidingWindowDataset:
        """
        Create a single **lazy** dataset from the per-ticker split data.
        Windows are computed on-the-fly in __getitem__ so that only the
        raw 2-D feature matrices need to stay in RAM.

        Parameters
        ----------
        split_data : per-ticker list from by the pipeline container each ticker dict.
        x_key / y_key : containing the keys for features and targets in the ticker dicts.
        prefix_x_key : for val and test dataset creation, use the prefix key
            for look-back windows to avoid zero-padding the earliest rows.

        Returns
        -------
        A _LazySlidingWindowDataset for the specified split.
        """
        segments: list[np.ndarray] = []
        targets:  list[np.ndarray] = []

        for split in split_data:
            X = split[x_key].values.astype(np.float32)
            y = split[y_key].values.astype(np.float32)

            if prefix_x_key is not None and prefix_x_key in split:
                prefix = split[prefix_x_key].values.astype(np.float32)
                # keep only the tail needed for look-back
                tail = prefix[-(self.context_length - 1):] \
                       if len(prefix) >= self.context_length - 1 else prefix
                segment = np.concatenate([tail, X], axis=0)
            else:
                segment = X

            segments.append(segment)
            targets.append(y)

        return _LazySlidingWindowDataset(
            segments, targets, self.context_length,
        )


    def fit(self, step_result: StepResult) -> Any:
        split_data = step_result.split_data

        # datasets
        train_ds = self._make_dataset(split_data, "X_train", "y_train")
        val_ds   = self._make_dataset(split_data, "X_val", "y_val",
                                      prefix_x_key="X_train")

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        # model initialisation
        num_features = split_data[0]["X_train"].shape[1]

        patchtst_cfg = PatchTSTConfig(
            num_input_channels=num_features,
            context_length=self.context_length,
            patch_length=self.patch_length,
            patch_stride=4,          # non-overlapping
            prediction_length=self.num_targets,
            d_model=self.d_model,
            num_attention_heads=self.num_heads,
            num_hidden_layers=self.num_layers,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
            scaling=False,                          # no internal scaling, already scaled in pipeline
            pre_norm=True,
            norm_type="batchnorm",
            num_targets=self.num_targets,
        )
        self.model = PatchTSTForRegression(patchtst_cfg)
        self.model.to(self.device)

        # ── optimiser & loss ────────────────────────────────────────────
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5,
        )
        loss_fn = nn.HuberLoss()

        # ── training loop with early stopping ───────────────────────────
        best_val_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        no_improve = 0

        for epoch in range(1, self.num_epochs + 1):
            # --- train ---
            self.model.train()
            train_loss_sum = 0.0
            for windows, targets in train_loader:
                windows = windows.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                preds = self.model(past_values=windows).regression_outputs
                loss = loss_fn(preds, targets)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()

            avg_train = train_loss_sum / len(train_loader)

            # --- validate ---
            self.model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for windows, targets in val_loader:
                    windows = windows.to(self.device)
                    targets = targets.to(self.device)
                    preds = self.model(past_values=windows).regression_outputs
                    val_loss_sum += loss_fn(preds, targets).item()

            avg_val = val_loss_sum / len(val_loader)

            print(
                f"[PatchTST] Epoch {epoch:>3}/{self.num_epochs}  "
                f"train_loss={avg_train:.6f}  val_loss={avg_val:.6f}"
            )

            # early stopping bookkeeping
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                print(f"[PatchTST] Early stopping at epoch {epoch}")
                break

        self.model.load_state_dict(best_state)
        print(f"[PatchTST] Training complete. Best val loss: {best_val_loss:.6f}")
        return best_val_loss


    def predict(self, step_result: StepResult) -> tuple:
        """
        Return ``(y_test, y_test_pred, y_train, y_train_pred)`` aligned
        with the concatenated global arrays in ``step_result``.
        """
        self.model.eval()

        # test: use tail of X_val as look-back prefix
        y_test_pred  = self._predict_splits(
            step_result.split_data, "X_test", prefix_key="X_val",
        )
        # train: zero-pad the earliest windows (no prior data available)
        y_train_pred = self._predict_splits(
            step_result.split_data, "X_train",
        )
        
        #validating that the ticker order in the split_data matches the global y_train / y_test arrays in step_result for correct alignment of predictions and target for evaluation step
        ticker_order_split_data = [split['ticker_name'] for split in step_result.split_data]
        ticker_order_y_train = step_result.y_train.index.get_level_values('ticker').unique().tolist()
        if ticker_order_split_data != ticker_order_y_train:
            raise ValueError(f"Ticker order mismatch between split_data and y_train! "
                             f"split_data order: {ticker_order_split_data}  y_train order: {ticker_order_y_train}")
        
        return step_result.y_test, y_test_pred, step_result.y_train, y_train_pred

    # ------------------------------------------------------------------ #

    def _predict_splits(
        self,
        split_data: list[dict],
        x_key: str,
        prefix_key: str | None = None,
    ) -> np.ndarray:
        """Run inference per ticker, concatenate results."""
        all_preds: list[np.ndarray] = []

        for split in split_data:
            X = split[x_key].values.astype(np.float32)

            if prefix_key is not None and prefix_key in split:
                prefix = split[prefix_key].values.astype(np.float32)
                tail = prefix[-(self.context_length - 1):] \
                       if len(prefix) >= self.context_length - 1 else prefix
                segment = np.concatenate([tail, X], axis=0)
            else:
                segment = X

            dummy_y = np.zeros((len(X), self.num_targets), dtype=np.float32)
            ds     = _LazySlidingWindowDataset([segment], [dummy_y], self.context_length)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

            ticker_preds: list[np.ndarray] = []
            with torch.no_grad():
                for batch_w, _ in loader:
                    batch_w = batch_w.to(self.device)
                    out = self.model(past_values=batch_w).regression_outputs
                    ticker_preds.append(out.cpu().numpy())

            all_preds.append(np.concatenate(ticker_preds, axis=0))

        return np.concatenate(all_preds, axis=0)

    def tune_hyperparameters(self, step_result: StepResult) -> dict:
        print("Starting hyperparameter tuning...")

        best_config = run_hyperparameter_tuning(self.config, step_result)

        return best_config


def run_hyperparameter_tuning(pipeline_config: AppConfig, step_result: StepResult) -> dict:
    # define search space
    param_grid = {
        "context_length": [64, 128, 256],
        "patch_length": [4, 8, 16],
        "d_model": [32, 64, 128],
        "num_attention_heads": [2, 4, 8],
        "num_hidden_layers": [2, 4, 8],
        "ffn_dim": [64, 128, 256],
        "batch_size": [32, 64, 128],
        "lr": [1e-4],
    }

    best_loss = float("inf")
    best_config: dict = {}

    # loop 15 times over random configurations
    for i in range(15):
        model_config = {
            "context_length": random.choice(param_grid["context_length"]),
            "patch_length": random.choice(param_grid["patch_length"]),
            "d_model": random.choice(param_grid["d_model"]),
            "num_attention_heads": random.choice(param_grid["num_attention_heads"]),
            "num_hidden_layers": random.choice(param_grid["num_hidden_layers"]),
            "ffn_dim": random.choice(param_grid["ffn_dim"]),
            "dropout": 0.2,
            "head_dropout": 0.2,
            "batch_size": random.choice(param_grid["batch_size"]),
            "lr": random.choice(param_grid["lr"]),
            "num_epochs": 100,
            "patience": 90,
        }

        print(f"Testing Config {i}: {model_config}")

        model = PatchTSTWrapper(pipeline_config, model_config=model_config)
        final_val_loss = model.fit(step_result)
        print(f"Validation Loss for Config {i}: {final_val_loss}")

        if final_val_loss < best_loss:
            best_loss = final_val_loss
            best_config = model_config

    print("Winner:", best_config, "with best loss:", best_loss)
    return best_config
