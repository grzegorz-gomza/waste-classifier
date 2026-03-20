from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@dataclass(frozen=True)
class PlotConfig:
    enabled: bool = True
    dpi: int = 300


class PlotArtifacts:
    def __init__(self, report_dir: Path, config: Optional[PlotConfig] = None):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or PlotConfig()

    def _should_plot(self) -> bool:
        return bool(self.config.enabled)

    def plot_dl_training_progress(
        self,
        *,
        model_name: str,
        train_losses: Sequence[float],
        train_accs: Sequence[float],
        val_losses: Sequence[float],
        val_accs: Sequence[float],
        run_name: str,
        output_name: Optional[str] = None,
    ) -> Optional[Path]:
        if not self._should_plot():
            return None

        epochs = np.arange(1, len(train_losses) + 1)

        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax1.plot(epochs, train_losses, label="train_loss", color="#1f77b4", linewidth=2)
        ax1.plot(epochs, val_losses, label="val_loss", color="#ff7f0e", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(epochs, train_accs, label="train_acc", color="#2ca02c", linewidth=2)
        ax2.plot(epochs, val_accs, label="val_acc", color="#d62728", linewidth=2)
        ax2.set_ylabel("Accuracy (%)")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.title(f"DL Training Progress ({model_name})")
        out = self.report_dir / (output_name or f"{run_name}_dl_training_progress.png")
        plt.tight_layout()
        plt.savefig(out, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_dl_training_progress_multi(
        self,
        *,
        run_name: str,
        curves: Dict[str, Dict[str, Sequence[float]]],
    ) -> Optional[Path]:
        """Plot multiple DL models' training curves on a single chart.

        curves format:
          {
            "resnet50": {"train_losses": [...], "val_losses": [...], "train_accs": [...], "val_accs": [...]},
            ...
          }
        """
        if not self._should_plot():
            return None
        if not curves:
            return None

        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax2 = ax1.twinx()

        for model_name, m in curves.items():
            train_losses = m.get("train_losses", [])
            val_losses = m.get("val_losses", [])
            train_accs = m.get("train_accs", [])
            val_accs = m.get("val_accs", [])

            epochs = np.arange(1, len(train_losses) + 1)
            if len(epochs) == 0:
                continue

            ax1.plot(epochs, train_losses, label=f"{model_name}_train_loss", linewidth=2)
            if len(val_losses) == len(train_losses):
                ax1.plot(epochs, val_losses, label=f"{model_name}_val_loss", linewidth=2, linestyle="--")

            if len(train_accs) == len(train_losses):
                ax2.plot(epochs, train_accs, label=f"{model_name}_train_acc", linewidth=2)
            if len(val_accs) == len(train_losses):
                ax2.plot(epochs, val_accs, label=f"{model_name}_val_acc", linewidth=2, linestyle="--")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy (%)")
        ax1.grid(True, alpha=0.3)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

        plt.title("DL Training Progress (All DL Models)")
        out = self.report_dir / f"{run_name}_dl_training_progress_all_models.png"
        plt.tight_layout()
        plt.savefig(out, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_ml_training_progress(
        self,
        *,
        run_name: str,
        evals_result: Dict[str, Dict[str, List[float]]],
        metric_key: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> Optional[Path]:
        if not self._should_plot():
            return None

        if not evals_result:
            return None

        dataset_name = next(iter(evals_result.keys()))
        metrics = evals_result[dataset_name]
        if not metrics:
            return None

        selected_metric = metric_key or next(iter(metrics.keys()))
        values = metrics.get(selected_metric)
        if not values:
            return None

        iters = np.arange(1, len(values) + 1)

        fig = plt.figure(figsize=(12, 7))
        plt.plot(iters, values, label=f"{dataset_name}:{selected_metric}", color="#1f77b4", linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel(selected_metric)
        plt.title("ML Training Progress (XGBoost)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")

        out = self.report_dir / (output_name or f"{run_name}_ml_training_progress.png")
        plt.tight_layout()
        plt.savefig(out, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_confusion_matrix(
        self,
        *,
        run_name: str,
        cm: np.ndarray,
        class_names: Sequence[str],
        title: str,
        file_suffix: str,
        normalize: bool = False,
        output_name: Optional[str] = None,
    ) -> Optional[Path]:
        if not self._should_plot():
            return None

        cm_arr = np.asarray(cm)
        if normalize:
            denom = cm_arr.sum(axis=1, keepdims=True)
            denom[denom == 0] = 1
            cm_arr = cm_arr.astype(float) / denom

        fig = plt.figure(figsize=(22, 18))
        fmt = ".2f" if normalize else "d"
        sns.heatmap(
            cm_arr,
            annot=False,
            fmt=fmt,
            cmap="Blues",
            xticklabels=list(class_names),
            yticklabels=list(class_names),
            cbar_kws={"label": "Proportion" if normalize else "Count"},
        )
        plt.title(title, fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("Predicted Label", fontsize=14, labelpad=10)
        plt.ylabel("True Label", fontsize=14, labelpad=10)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout(pad=3.0)

        out = self.report_dir / (output_name or f"{run_name}_confusion_matrix_{file_suffix}.png")
        plt.savefig(out, dpi=self.config.dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        return out

    def plot_model_comparison(
        self,
        *,
        run_name: str,
        results: Dict[str, Dict[str, float]],
        metric: str = "accuracy",
    ) -> Optional[Path]:
        if not self._should_plot():
            return None

        if not results:
            return None

        labels = list(results.keys())
        values = [float(results[k].get(metric, 0.0)) for k in labels]

        fig = plt.figure(figsize=(12, 7))
        bars = plt.bar(labels, values, color="#1f77b4")
        plt.title(f"Model Comparison ({metric})", fontsize=16, fontweight="bold")
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.grid(axis="y", alpha=0.3)

        for bar, v in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.3f}", ha="center")

        out = self.report_dir / f"{run_name}_model_comparison_{metric}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return out
