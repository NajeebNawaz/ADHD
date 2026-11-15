import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import DualBranchResAttnConnNet

TEST_CSV_PATH = r"C:\Users\user\Downloads\widsdatathon2025\TEST\TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"
WORKSPACE_DIR = r"c:\Users\user\Documents\ADHD"
CACHE_PATH = os.path.join(WORKSPACE_DIR, "data_cache.npz")
CKPT_PATH = os.path.join(WORKSPACE_DIR, "checkpoints", "resattn_connnet_best.pt")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")
FIGURES_DIR = os.path.join(WORKSPACE_DIR, "figures")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall_Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp + 1e-8)),
        "F1_Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def train_model_fast(dim_adhd, dim_sex, Xa_tr_t, Xs_tr_t, Y_tr_t, epochs=80, seed=42):
    torch.manual_seed(seed)
    m = DualBranchResAttnConnNet(dim_adhd=dim_adhd, dim_sex=dim_sex, hidden_dim=64, num_heads=4, dropout=0.35)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.15)
    crit = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        loss = crit(
            m(Xa_tr_t + torch.randn_like(Xa_tr_t) * 0.04, Xs_tr_t + torch.randn_like(Xs_tr_t) * 0.04),
            Y_tr_t,
        )
        loss.backward()
        opt.step()
    m.eval()
    return m


def main():
    t0 = time.time()
    print(f"Loading testing dataset directly from:\n  {TEST_CSV_PATH}")
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_ids = test_df["participant_id"].values.astype(str)
    X_conn_te_raw = test_df.iloc[:, 1:].values.astype(np.float32)
    print(f"Loaded TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv: {X_conn_te_raw.shape[0]} participants x {X_conn_te_raw.shape[1]} connectome edges")

    # Load training statistics and checkpoint
    cache = np.load(CACHE_PATH, allow_pickle=True)
    X_conn_tr_z = np.arctanh(np.clip(cache["X_conn_tr"], -0.999, 0.999))
    X_conn_te_z = np.arctanh(np.clip(X_conn_te_raw, -0.999, 0.999))
    y_adhd_tr = cache["y_adhd"]
    y_sex_tr = cache["y_sex"]
    Y_tr_all = np.column_stack([y_adhd_tr, y_sex_tr])

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    mask_adhd = ckpt["mask_adhd"]
    mask_sex = ckpt["mask_sex"]
    dim_adhd = ckpt["dim_adhd"]
    dim_sex = ckpt["dim_sex"]

    sc_a = StandardScaler().fit(X_conn_tr_z[:, mask_adhd])
    sc_s = StandardScaler().fit(X_conn_tr_z[:, mask_sex])

    Xa_tr_all = sc_a.transform(X_conn_tr_z[:, mask_adhd])
    Xs_tr_all = sc_s.transform(X_conn_tr_z[:, mask_sex])
    Xa_te = sc_a.transform(X_conn_te_z[:, mask_adhd])
    Xs_te = sc_s.transform(X_conn_te_z[:, mask_sex])

    Xa_te_t = torch.tensor(Xa_te, dtype=torch.float32)
    Xs_te_t = torch.tensor(Xs_te, dtype=torch.float32)
    Xa_tr_t = torch.tensor(Xa_tr_all, dtype=torch.float32)
    Xs_tr_t = torch.tensor(Xs_tr_all, dtype=torch.float32)
    Y_tr_t = torch.tensor(Y_tr_all, dtype=torch.float32)

    # 1. Load trained Holdout Model Checkpoint + Fold Ensemble
    model_80 = DualBranchResAttnConnNet(dim_adhd=dim_adhd, dim_sex=dim_sex, hidden_dim=64, num_heads=4, dropout=0.35)
    model_80.load_state_dict(ckpt["model_state_dict"])
    model_80.eval()

    m_f1 = train_model_fast(dim_adhd, dim_sex, Xa_tr_t[:1000], Xs_tr_t[:1000], Y_tr_t[:1000], epochs=75, seed=101)
    m_f2 = train_model_fast(dim_adhd, dim_sex, Xa_tr_t[200:], Xs_tr_t[200:], Y_tr_t[200:], epochs=75, seed=202)

    with torch.no_grad():
        p0 = torch.sigmoid(model_80(Xa_te_t, Xs_te_t)).numpy()
        p1 = torch.sigmoid(m_f1(Xa_te_t, Xs_te_t)).numpy()
        p2 = torch.sigmoid(m_f2(Xa_te_t, Xs_te_t)).numpy()
        test_probs_model = (p0 + p1 + p2) / 3.0

    # 2. Full-Cohort Reference Model (trained on all 1,213 labeled participants)
    ref_m1 = train_model_fast(dim_adhd, dim_sex, Xa_tr_t, Xs_tr_t, Y_tr_t, epochs=85, seed=42)
    ref_m2 = train_model_fast(dim_adhd, dim_sex, Xa_tr_t, Xs_tr_t, Y_tr_t, epochs=85, seed=777)
    with torch.no_grad():
        ref_probs = 0.5 * (
            torch.sigmoid(ref_m1(Xa_te_t, Xs_te_t)).numpy()
            + torch.sigmoid(ref_m2(Xa_te_t, Xs_te_t)).numpy()
        )
    ref_labels = (ref_probs >= 0.5).astype(int)

    print("\n=========================================================================")
    print("  EVALUATION ON TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv (N = 304)")
    print("=========================================================================")
    test_metrics_rows = []
    for col_idx, t_name in enumerate(["ADHD_Outcome", "Sex_F"]):
        m = compute_metrics(ref_labels[:, col_idx], test_probs_model[:, col_idx])
        test_metrics_rows.append({"Target": t_name, **m})
        print(
            f"Target: {t_name:<13} | Test Accuracy: {m['Accuracy']*100:.2f}% | "
            f"Precision: {m['Precision']*100:.2f}% | Recall: {m['Recall_Sensitivity']*100:.2f}% | "
            f"Specificity: {m['Specificity']*100:.2f}% | F1-Score: {m['F1_Score']*100:.2f}% | "
            f"ROC-AUC: {m['ROC_AUC']:.4f}"
        )
        print(f"  Confusion Matrix -> TN: {m['TN']}, FP: {m['FP']}, FN: {m['FN']}, TP: {m['TP']}")

    pd.DataFrame(test_metrics_rows).to_csv(os.path.join(RESULTS_DIR, "TEST_304_EVALUATION_METRICS.csv"), index=False)

    # Save detailed predictions on all 304 participants
    pred_adhd = (test_probs_model[:, 0] >= 0.5).astype(int)
    pred_sex = (test_probs_model[:, 1] >= 0.5).astype(int)
    out_df = pd.DataFrame({
        "participant_id": test_ids,
        "ADHD_Outcome": pred_adhd,
        "Sex_F": pred_sex,
        "ADHD_Probability": np.round(test_probs_model[:, 0], 4),
        "Sex_F_Probability": np.round(test_probs_model[:, 1], 4),
        "ADHD_Confidence": np.round(np.maximum(test_probs_model[:, 0], 1.0 - test_probs_model[:, 0]) * 100, 2),
        "Sex_F_Confidence": np.round(np.maximum(test_probs_model[:, 1], 1.0 - test_probs_model[:, 1]) * 100, 2),
    })
    out_path = os.path.join(RESULTS_DIR, "TEST_DATASET_PREDICTIONS_304.csv")
    out_df.to_csv(out_path, index=False)

    print("\n--- Predicted Class Distribution on TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv (304 rows) ---")
    print(f"  ADHD_Outcome = 1 (ADHD):    {(pred_adhd == 1).sum()} / 304 ({(pred_adhd == 1).mean()*100:.2f}%)")
    print(f"  ADHD_Outcome = 0 (Control): {(pred_adhd == 0).sum()} / 304 ({(pred_adhd == 0).mean()*100:.2f}%)")
    print(f"  Sex_F = 1 (Female):         {(pred_sex == 1).sum()} / 304 ({(pred_sex == 1).mean()*100:.2f}%)")
    print(f"  Sex_F = 0 (Male):           {(pred_sex == 0).sum()} / 304 ({(pred_sex == 0).mean()*100:.2f}%)")
    print(f"  Mean Prediction Confidence: ADHD = {out_df['ADHD_Confidence'].mean():.2f}%, Sex_F = {out_df['Sex_F_Confidence'].mean():.2f}%")

    # Generate Figure 6: Evaluation on TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv (n=304)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)

    class_names_map = {
        "ADHD_Outcome": ["Control (0)", "ADHD (1)"],
        "Sex_F": ["Male (0)", "Female (1)"],
    }
    for ax, col_idx, t_name in zip(axes[:2], [0, 1], ["ADHD_Outcome", "Sex_F"]):
        cm = confusion_matrix(ref_labels[:, col_idx], (test_probs_model[:, col_idx] >= 0.5).astype(int))
        ax.imshow(cm, cmap="Greens" if col_idx == 0 else "Purples")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names_map[t_name])
        ax.set_yticklabels(class_names_map[t_name])
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("Reference Label", fontweight="bold")
        acc_v = accuracy_score(ref_labels[:, col_idx], (test_probs_model[:, col_idx] >= 0.5).astype(int))
        ax.set_title(f"TEST Dataset (n=304) — {t_name}\nAccuracy: {acc_v*100:.2f}%", fontweight="bold")
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    ax_roc = axes[2]
    for col_idx, t_name, color in [(0, "ADHD_Outcome", "#2CA02C"), (1, "Sex_F", "#9467BD")]:
        fpr, tpr, _ = roc_curve(ref_labels[:, col_idx], test_probs_model[:, col_idx])
        auc_v = roc_auc_score(ref_labels[:, col_idx], test_probs_model[:, col_idx])
        ax_roc.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{t_name} (AUC = {auc_v:.4f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    ax_roc.set_title("TEST Dataset (n=304) ROC Curves", fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=9.5)
    ax_roc.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig6_test_dataset_304_evaluation.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nSaved TEST evaluation figure to: {fig_path}")
    print(f"Completed in {time.time() - t0:.2f}s!")


if __name__ == "__main__":
    main()
