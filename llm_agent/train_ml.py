"""
Train Random Forest + XGBoost on the CIC IoT tabular dataset (notebook-equivalent).

Mirrors ml_classifier.ipynb: flow-disjoint split, log normalization, class weights,
artifact export for Track B (random_forest.joblib, xgboost.json, encoders).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from llm_agent.feature_serializer import log_normalize_matrix

DROP_COLS = ("Src IP", "Dst IP", "Timestamp", "Attack Name")
RANDOM_STATE_DEFAULT = 43


@dataclass
class TrainResult:
    """Summary metrics and paths after a training run."""

    output_dir: str
    class_names: List[str]
    n_train: int
    n_test: int
    rf_accuracy: float
    rf_macro_f1: float
    xgb_accuracy: float
    xgb_macro_f1: float
    rf_report: Dict[str, Any]
    xgb_report: Dict[str, Any]
    saved_files: List[str]

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_combined_dataframe(
    data_dir: Path,
    *,
    dos_tcp_nrows: int = 100_000,
) -> pd.DataFrame:
    """Load four class CSVs (same filenames as EDA / ml_classifier notebooks)."""
    data_dir = Path(data_dir)
    benign = pd.read_csv(data_dir / "Benign Traffic.csv")
    ddos = pd.read_csv(data_dir / "DDoS UDP Flood.csv")
    dos_tcp = pd.read_csv(data_dir / "DoS TCP Flood.csv", nrows=dos_tcp_nrows)
    recon = pd.read_csv(data_dir / "Recon Port Scan.csv")

    benign["Label"] = "BENIGN"
    ddos["Label"] = "DDOS_UDP"
    dos_tcp["Label"] = "DOS_TCP"
    recon["Label"] = "RECON"

    return pd.concat([benign, ddos, dos_tcp, recon], ignore_index=True)


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop identifier columns; return df with Flow ID + features + Label."""
    drop = [c for c in DROP_COLS if c in df.columns]
    out = df.drop(columns=drop, errors="ignore").copy()
    feature_cols = [c for c in out.columns if c not in ("Label", "Flow ID")]
    return out, feature_cols


def train_and_save_artifacts(
    *,
    data_dir: Path,
    output_dir: Path,
    random_state: int = RANDOM_STATE_DEFAULT,
    sgkf_n_splits: int = 4,
    dos_tcp_nrows: int = 100_000,
    train_rf: bool = True,
    train_xgb: bool = True,
    save_plots: bool = True,
    shap_subsample: int = 2000,
) -> TrainResult:
    """
    Full pipeline from raw CSVs to models/ artifacts.

    Parameters mirror ml_classifier.ipynb (StratifiedGroupKFold first fold as test).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_combined_dataframe(data_dir, dos_tcp_nrows=dos_tcp_nrows)
    df, feature_cols = preprocess_dataframe(df)

    groups = df["Flow ID"].values
    X = df[feature_cols].values.astype(float)
    y_raw = df["Label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    sgkf = StratifiedGroupKFold(n_splits=sgkf_n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train = log_normalize_matrix(X_train)
    X_test = log_normalize_matrix(X_test)

    saved: List[str] = []

    rf = None
    y_pred_rf = None
    rf_acc = rf_f1 = 0.0
    rf_report: Dict[str, Any] = {}

    if train_rf:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_acc = float(accuracy_score(y_test, y_pred_rf))
        rf_f1 = float(f1_score(y_test, y_pred_rf, average="macro"))
        rf_report = classification_report(
            y_test, y_pred_rf, target_names=class_names, output_dict=True, zero_division=0
        )
        rf_path = output_dir / "random_forest.joblib"
        joblib.dump(rf, rf_path)
        saved.append(str(rf_path))

        if save_plots:
            _save_confusion_plot(
                y_test,
                y_pred_rf,
                class_names,
                output_dir / "rf_confusion_matrix.png",
                "Random Forest — Confusion Matrix",
            )
            _save_rf_feature_importance(rf, feature_cols, output_dir / "rf_feature_importance.png")

    xgb_clf = None
    y_pred_xgb = None
    xgb_acc = xgb_f1 = 0.0
    xgb_report: Dict[str, Any] = {}

    if train_xgb:
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        sample_weights = class_weights[y_train]

        xgb_clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=len(class_names),
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=random_state,
            verbosity=0,
        )
        xgb_clf.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        y_pred_xgb = xgb_clf.predict(X_test)
        xgb_acc = float(accuracy_score(y_test, y_pred_xgb))
        xgb_f1 = float(f1_score(y_test, y_pred_xgb, average="macro"))
        xgb_report = classification_report(
            y_test, y_pred_xgb, target_names=class_names, output_dict=True, zero_division=0
        )
        xgb_path = output_dir / "xgboost.json"
        xgb_clf.save_model(str(xgb_path))
        saved.append(str(xgb_path))

        if save_plots:
            _save_confusion_plot(
                y_test,
                y_pred_xgb,
                class_names,
                output_dir / "xgb_confusion_matrix.png",
                "XGBoost — Confusion Matrix",
            )
            _save_xgb_shap_plot(
                xgb_clf,
                X_test,
                feature_cols,
                output_dir / "xgb_shap_importance.png",
                shap_subsample,
            )

    le_path = output_dir / "label_encoder.joblib"
    fc_path = output_dir / "feature_cols.joblib"
    joblib.dump(le, le_path)
    joblib.dump(feature_cols, fc_path)
    saved.extend([str(le_path), str(fc_path)])

    metrics_path = output_dir / "training_metrics.json"
    result = TrainResult(
        output_dir=str(output_dir.resolve()),
        class_names=class_names,
        n_train=len(train_idx),
        n_test=len(test_idx),
        rf_accuracy=rf_acc,
        rf_macro_f1=rf_f1,
        xgb_accuracy=xgb_acc,
        xgb_macro_f1=xgb_f1,
        rf_report=rf_report,
        xgb_report=xgb_report,
        saved_files=saved,
    )
    metrics_path.write_text(json.dumps(result.to_json_dict(), indent=2), encoding="utf-8")
    saved.append(str(metrics_path))

    return result


def _configure_matplotlib_backend() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401 — backend registration

    plt.ioff()


def _save_confusion_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    path: Path,
    title: str,
) -> None:
    _configure_matplotlib_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues" if "RF" in title else "Oranges")
    ax.set_title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _save_rf_feature_importance(rf: RandomForestClassifier, feature_cols: List[str], path: Path) -> None:
    _configure_matplotlib_backend()
    import matplotlib.pyplot as plt

    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(9, 6))
    imp.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest — Top 25 Feature Importances")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _save_xgb_shap_plot(
    xgb_clf: xgb.XGBClassifier,
    X_test: np.ndarray,
    feature_cols: List[str],
    path: Path,
    subsample: int,
) -> None:
    _configure_matplotlib_backend()
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(RANDOM_STATE_DEFAULT)
    n = min(subsample, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)
    X_shap = X_test[idx]

    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        # Multiclass: one matrix per class — average mean |SHAP| across classes.
        per_class = [np.abs(sv).mean(axis=0) for sv in shap_values]
        mean_abs_shap = np.mean(np.stack(per_class, axis=0), axis=0)
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            mean_abs_shap = np.abs(arr).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(arr).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 6))
    shap_series.plot(kind="barh", ax=ax, color="#DD8452")
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("XGBoost — Top 20 SHAP Feature Importances")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)
