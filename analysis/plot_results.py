"""리포트(JSON) 기반 시각화: 혼동행렬, 지표 바 차트, PR 곡선."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def plot_confusion_matrix(cm: List[List[int]], labels: List[str], out_path: Path) -> None:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, cm_arr[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pr_curves(y_true, y_proba, labels: List[str], out_path: Path) -> None:
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    y_true_idx = np.array([label_to_idx[lab] for lab in y_true])
    y_proba_arr = np.array(y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, lab in enumerate(labels):
        y_bin = (y_true_idx == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_bin, y_proba_arr[:, i])
        ap = average_precision_score(y_bin, y_proba_arr[:, i])
        ax.plot(recall, precision, label=f"{lab} (AP={ap:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bar_metrics(report: dict, labels: List[str], out_path: Path) -> None:
    names = []
    precisions = []
    recalls = []
    f1s = []
    for lab in labels:
        if lab in report:
            names.append(lab)
            precisions.append(report[lab]["precision"])
            recalls.append(report[lab]["recall"])
            f1s.append(report[lab]["f1-score"])
    if not names:
        return
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precisions, width, label="precision")
    ax.bar(x, recalls, width, label="recall")
    ax.bar(x + width, f1s, width, label="f1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="리포트 JSON으로부터 혼동행렬/지표/PR 시각화")
    parser.add_argument("--report", type=Path, required=True, help="svm_report.json 또는 dl_report.json")
    parser.add_argument("--out-prefix", type=Path, default=Path("experiments/ml/svm"), help="저장 파일 prefix (확장자 제외)")
    args = parser.parse_args()

    with args.report.open(encoding="utf-8") as f:
        meta = json.load(f)

    labels = meta.get("labels")
    cm = meta.get("confusion_matrix") or meta.get("val_confusion_matrix")
    if cm is None or labels is None:
        raise ValueError("confusion_matrix 또는 labels가 리포트에 없습니다.")

    out_cm = args.out_prefix.with_name(args.out_prefix.name + "_cm.png")
    out_bar = args.out_prefix.with_name(args.out_prefix.name + "_metrics.png")
    out_pr = args.out_prefix.with_name(args.out_prefix.name + "_pr.png")

    plot_confusion_matrix(cm, labels, out_cm)
    print(f"혼동행렬 이미지 저장: {out_cm}")

    report = meta.get("classification_report") or meta.get("val_classification_report")
    if report:
        plot_bar_metrics(report, labels, out_bar)
        print(f"정밀도/재현율/F1 바 차트 저장: {out_bar}")

    y_true = meta.get("y_true")
    y_proba = meta.get("y_proba")
    if y_true and y_proba:
        plot_pr_curves(y_true, y_proba, labels, out_pr)
        print(f"PR 곡선 저장: {out_pr}")


if __name__ == "__main__":
    main()
