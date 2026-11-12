"""Train and evaluate the pose decoder MLP (woofdecoder.body.PoseClassifier), as in notebook 31.

    splits = split_by_group(X, y, kept["group"].to_numpy())
    history = train_pose_decoder(splits, lr=0.003, num_epochs=250, checkpoint_path=path)
    evaluate_pose_decoder(path, splits)

The checkpoint is a plain state_dict, which is what BodyBranch loads
(configs/pipeline.yaml: branches.body.head).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from woofdecoder.body.branch import DECODER_CLASSES, PoseClassifier, pick_device
from wooftrainer.body_trainer.pose_dataset import PoseSplits
from wooftrainer.settings import (
    POSE_DECODER_BATCH_SIZE,
    POSE_DECODER_EPOCHS,
    POSE_DECODER_LR,
    POSE_DECODER_LR_VALUES,
    POSE_DECODER_SEED,
)

NUM_CLASSES = len(DECODER_CLASSES)
PRINT_EVERY = 10


def class_weights(y_train: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights: total / (num_classes * count), so rare classes count more."""
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
    if (counts == 0).any():
        missing = [DECODER_CLASSES[i] for i in np.flatnonzero(counts == 0)]
        raise ValueError(f"No training examples for: {missing}")
    return counts.sum() / (NUM_CLASSES * counts)


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool = False) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64))
    )
    return DataLoader(dataset, batch_size=POSE_DECODER_BATCH_SIZE, shuffle=shuffle)


def predict_logits(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X.astype(np.float32)).to(device)).cpu().numpy()


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_macro_f1: list[float] = field(default_factory=list)
    best_f1: float = -1.0
    best_epoch: int = -1

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        plt.plot(self.train_loss, label="Train loss")
        plt.plot(self.val_loss, label="Val loss")
        plt.plot(self.val_macro_f1, label="Val macro-F1")
        plt.axvline(self.best_epoch, color="gray", linestyle=":", label="Best epoch")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()


def train_pose_decoder(
    splits: PoseSplits,
    checkpoint_path: str | Path,
    lr: float = POSE_DECODER_LR,
    num_epochs: int = POSE_DECODER_EPOCHS,
    seed: int = POSE_DECODER_SEED,
    device: torch.device | None = None,
    verbose: bool = True,
) -> TrainHistory:
    """Adam + class-weighted cross-entropy; saves the state_dict with the best val macro-F1."""
    if len(splits.y_train) == 0 or len(splits.y_val) == 0:
        raise ValueError("train and val splits must both be non-empty")
    device = device or pick_device()
    torch.manual_seed(seed)
    model = PoseClassifier(n_features=splits.X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weights = torch.tensor(class_weights(splits.y_train), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    train_loader = make_loader(splits.X_train, splits.y_train, shuffle=True)
    val_loader = make_loader(splits.X_val, splits.y_val)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = TrainHistory()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        history.train_loss.append(epoch_loss / len(train_loader))

        model.eval()
        val_total = 0.0
        val_preds = []
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features.to(device))
                val_total += criterion(outputs, labels.to(device)).item() * len(labels)
                val_preds.append(outputs.argmax(dim=1).cpu())
        history.val_loss.append(val_total / len(splits.y_val))
        # Same metric as evaluation: macro over all classes, absent classes score 0.
        val_f1 = score_predictions(splits.y_val, torch.cat(val_preds).numpy())["macro_f1"]
        history.val_macro_f1.append(val_f1)

        is_best = val_f1 > history.best_f1
        if is_best:
            history.best_f1 = val_f1
            history.best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
        if verbose and (epoch % PRINT_EVERY == 0 or is_best):
            print(
                f"Epoch [{epoch}/{num_epochs}] train loss {history.train_loss[-1]:.4f}, "
                f"val loss {history.val_loss[-1]:.4f}, val F1 {val_f1:.4f}{' *' if is_best else ''}"
            )

    if verbose:
        print(f"Best val macro-F1: {history.best_f1:.4f} at epoch {history.best_epoch}")
    return history


def sweep_learning_rates(
    splits: PoseSplits,
    checkpoint_dir: str | Path,
    lr_values: tuple[float, ...] = POSE_DECODER_LR_VALUES,
    num_epochs: int = POSE_DECODER_EPOCHS,
    verbose: bool = False,
) -> pd.DataFrame:
    """One run per learning rate (same seed); checkpoints saved as decoder_lr{lr}.pt."""
    rows = []
    for lr in lr_values:
        path = Path(checkpoint_dir) / f"decoder_lr{lr}.pt"
        history = train_pose_decoder(splits, path, lr=lr, num_epochs=num_epochs, verbose=verbose)
        rows.append(
            {
                "lr": lr,
                "best_val_macro_f1": history.best_f1,
                "best_epoch": history.best_epoch,
                "checkpoint": str(path),
            }
        )
        print(f"lr {lr}: best val macro-F1 {history.best_f1:.4f} (epoch {history.best_epoch})")
    return pd.DataFrame(rows).sort_values("best_val_macro_f1", ascending=False, ignore_index=True)


def load_pose_decoder(
    checkpoint_path: str | Path, device: torch.device | None = None
) -> PoseClassifier:
    """Input size is read from the checkpoint, so it matches whatever the model was trained on."""
    device = device or pick_device()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model = PoseClassifier(n_features=state_dict["fc1.weight"].shape[1]).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=range(NUM_CLASSES), average="macro", zero_division=0)
        ),
    }


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float | str]:
    """Always predict the most common training class."""
    most_common = int(np.bincount(y_train, minlength=NUM_CLASSES).argmax())
    scores = score_predictions(y_test, np.full(len(y_test), most_common))
    return {"predicts": DECODER_CLASSES[most_common], **scores}


def evaluate_pose_decoder(
    checkpoint_path: str | Path,
    splits: PoseSplits,
    split: str = "test",
    verbose: bool = True,
) -> dict:
    """Accuracy, macro-F1, per-class report and confusion matrix (rows = true, cols = predicted)."""
    X, y = getattr(splits, f"X_{split}"), getattr(splits, f"y_{split}")
    device = pick_device()
    model = load_pose_decoder(checkpoint_path, device)
    predictions = predict_logits(model, X, device).argmax(axis=1)

    scores = score_predictions(y, predictions)
    confusion = pd.DataFrame(
        confusion_matrix(y, predictions, labels=range(NUM_CLASSES)),
        index=pd.Index(DECODER_CLASSES, name="true"),
        columns=pd.Index(DECODER_CLASSES, name="predicted"),
    )
    report = classification_report(
        y, predictions, labels=range(NUM_CLASSES), target_names=DECODER_CLASSES, zero_division=0
    )
    baseline = majority_baseline(splits.y_train, y)

    if verbose:
        print(f"{split} accuracy: {scores['accuracy']:.4f}, macro-F1: {scores['macro_f1']:.4f}")
        print(
            f"baseline ({baseline['predicts']}) accuracy: {baseline['accuracy']:.4f}, macro-F1: {baseline['macro_f1']:.4f}"
        )
        print(report)
        print(confusion)
    return {**scores, "baseline": baseline, "report": report, "confusion": confusion}
