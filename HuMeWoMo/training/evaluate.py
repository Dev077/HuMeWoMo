import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

PACTIVITY_THRESHOLD = 6.0

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from HuMeWoMo.datasets.homo_binding_dataset import get_homo_dataloaders
from HuMeWoMo.models.homo_binding_model import HomoBindingModel

FIGS_DIR = os.path.join(project_root, "training/figs")
os.makedirs(FIGS_DIR, exist_ok=True)


def evaluate(model_path, data_dir="./bindingdb_data/final_dataset", batch_size=48, split="val"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    print(f"Loading model from: {model_path}")

    # Load model
    model = HomoBindingModel(
        drug_in_dim=50,
        enzyme_in_dim=27,
        hidden_dim=128,
        n_heads=4,
        num_drug_layers=3,
        num_enzyme_layers=3,
        num_decoder_layers=3,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}, val_loss {checkpoint.get('val_loss', '?')})")
    else:
        model.load_state_dict(checkpoint)
        print("  Loaded raw state_dict")

    # Load data
    loader = get_homo_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4 if device.type == "cuda" else 0,
        split=split,
    )

    # Run inference
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating [{split}]"):
            batch = batch.to(device)
            preds = model(batch)
            all_preds.append(preds.cpu())
            all_targets.append(batch.y.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # Classification: threshold at pActivity >= 6
    pred_labels = (preds >= PACTIVITY_THRESHOLD).long().numpy()
    true_labels = (targets >= PACTIVITY_THRESHOLD).long().numpy()

    acc = np.mean(pred_labels == true_labels)
    #acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, zero_division=0)
    rec = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    auc = roc_auc_score(true_labels, preds.numpy())
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()

    n_pos = int(true_labels.sum())
    n_neg = len(true_labels) - n_pos

    print(f"\n{'=' * 50}")
    print(f"  Classification results on {split} set ({len(targets)} samples)")
    print(f"  Threshold: pActivity >= {PACTIVITY_THRESHOLD}")
    print(f"  Class balance: {n_pos} active ({100*n_pos/len(true_labels):.1f}%) / {n_neg} inactive ({100*n_neg/len(true_labels):.1f}%)")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"{'=' * 50}")
    print(f"  Confusion Matrix:")
    print(f"    TP={tp}  FP={fp}")
    print(f"    FN={fn}  TN={tn}")
    print(f"{'=' * 50}")

    # Scatter plot: predicted vs actual, colored by classification
    preds_np = preds.numpy()
    targets_np = targets.numpy()

    plt.figure(figsize=(8, 8))
    correct = pred_labels == true_labels
    plt.scatter(targets_np[correct], preds_np[correct], alpha=0.3, s=10, c="tab:blue", label="Correct")
    plt.scatter(targets_np[~correct], preds_np[~correct], alpha=0.3, s=10, c="tab:red", label="Misclassified")

    lo = min(targets_np.min(), preds_np.min())
    hi = max(targets_np.max(), preds_np.max())
    plt.axhline(PACTIVITY_THRESHOLD, color="gray", linestyle="--", alpha=0.7, label=f"Threshold ({PACTIVITY_THRESHOLD})")
    plt.axvline(PACTIVITY_THRESHOLD, color="gray", linestyle="--", alpha=0.7)
    plt.plot([lo, hi], [lo, hi], "r--", alpha=0.5, label="Perfect prediction")

    plt.xlabel("Actual pActivity")
    plt.ylabel("Predicted pActivity")
    plt.title(f"Predicted vs Actual ({split}) — Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(FIGS_DIR, f"eval_{split}_scatter.png")
    plt.savefig(plot_path, dpi=150)
    print(f"  Scatter plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained HomoBindingModel")
    parser.add_argument("model_path", type=str, help="Path to .pt model checkpoint")
    parser.add_argument("--data-dir", type=str, default="./bindingdb_data/final_dataset")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    evaluate(args.model_path, args.data_dir, args.batch_size, args.split)
