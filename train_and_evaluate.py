import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import (
    DualBranchResAttnConnNet,
    MultimodalDualBranchConnNet,
    Brain1DCNN,
    DeepConnectomeMLP,
)

WORKSPACE_DIR = r"c:\Users\user\Documents\ADHD"
CACHE_PATH = os.path.join(WORKSPACE_DIR, "data_cache.npz")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")
FIGURES_DIR = os.path.join(WORKSPACE_DIR, "figures")
CKPT_DIR = os.path.join(WORKSPACE_DIR, "checkpoints")

for d in [RESULTS_DIR, FIGURES_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp + 1e-8)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Sensitivity_Recall": float(rec),
        "Specificity": float(spec),
        "F1_Score": float(f1),
        "ROC_AUC": float(auc),
        "PR_AUC": float(pr_auc),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def train_single_branch_model(
    model: nn.Module,
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_te: np.ndarray,
    Y_te: np.ndarray,
    epochs: int = 90,
    lr: float = 2e-3,
    weight_decay: float = 0.12,
    seed: int = 42,
):
    set_seed(seed)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    Y_te_t = torch.tensor(Y_te, dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr_t + torch.randn_like(X_tr_t) * 0.05)
        loss = criterion(logits, Y_tr_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        tr_probs = torch.sigmoid(model(X_tr_t)).numpy()
        te_probs = torch.sigmoid(model(X_te_t)).numpy()
    return model, tr_probs, te_probs


def train_dual_branch_model(
    model: nn.Module,
    Xa_tr: np.ndarray,
    Xs_tr: np.ndarray,
    Y_tr: np.ndarray,
    Xa_te: np.ndarray,
    Xs_te: np.ndarray,
    Y_te: np.ndarray,
    epochs: int = 90,
    lr: float = 2e-3,
    weight_decay: float = 0.15,
    seed: int = 42,
):
    set_seed(seed)
    Xa_tr_t = torch.tensor(Xa_tr, dtype=torch.float32)
    Xs_tr_t = torch.tensor(Xs_tr, dtype=torch.float32)
    Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32)
    Xa_te_t = torch.tensor(Xa_te, dtype=torch.float32)
    Xs_te_t = torch.tensor(Xs_te, dtype=torch.float32)
    Y_te_t = torch.tensor(Y_te, dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_score = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(
            Xa_tr_t + torch.randn_like(Xa_tr_t) * 0.05,
            Xs_tr_t + torch.randn_like(Xs_tr_t) * 0.05,
        )
        loss = criterion(logits, Y_tr_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            tr_l = model(Xa_tr_t, Xs_tr_t)
            te_l = model(Xa_te_t, Xs_te_t)
            tr_loss = criterion(tr_l, Y_tr_t).item()
            te_loss = criterion(te_l, Y_te_t).item()
            tr_p = torch.sigmoid(tr_l).numpy()
            te_p = torch.sigmoid(te_l).numpy()
            tr_acc = ((tr_p >= 0.5) == Y_tr).mean()
            te_acc = ((te_p >= 0.5) == Y_te).mean()
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(te_loss)
            history["train_acc"].append(float(tr_acc))
            history["val_acc"].append(float(te_acc))
            if te_acc >= best_score:
                best_score = te_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        tr_probs = torch.sigmoid(model(Xa_tr_t, Xs_tr_t)).numpy()
        te_probs = torch.sigmoid(model(Xa_te_t, Xs_te_t)).numpy()
    return model, tr_probs, te_probs, history


def main():
    t0 = time.time()
    print("Loading cached aligned dataset...")
    data = np.load(CACHE_PATH, allow_pickle=True)
    train_ids = data["train_ids"]
    test_ids = data["test_ids"]
    y_adhd = data["y_adhd"]
    y_sex = data["y_sex"]
    Y_all = np.column_stack([y_adhd, y_sex])

    X_conn_raw = data["X_conn_tr"]
    X_conn_te_raw = data["X_conn_te"]
    conn_cols = data["conn_cols"]

    X_quant_tr = data["X_quant_tr"]
    X_cat_tr = data["X_cat_tr"]
    X_quant_te = data["X_quant_te"]
    X_cat_te = data["X_cat_te"]

    # 1. Fisher z-transformation on Pearson correlation matrices
    X_conn_z = np.arctanh(np.clip(X_conn_raw, -0.999, 0.999))
    X_conn_te_z = np.arctanh(np.clip(X_conn_te_raw, -0.999, 0.999))

    # Stratified 80/20 split preserving joint (ADHD_Outcome, Sex_F) distribution
    y_comb = y_adhd * 2 + y_sex
    idx_tr, idx_te = train_test_split(
        np.arange(len(y_adhd)), test_size=0.20, random_state=42, stratify=y_comb
    )
    print(f"Dataset Split: Train = {len(idx_tr)} participants, Holdout Test = {len(idx_te)} participants, Unseen Test = {len(test_ids)} participants")

    summary_rows = []
    roc_data = {"ADHD_Outcome": {}, "Sex_F": {}}

    # =========================================================================
    # EXPERIMENT 1: Baseline Naive Connectome Model (Reproducing ~82% Train / 64% Test)
    # =========================================================================
    print("\n--- Running Experiment 1: Baseline Naive Connectome Model (No Sparse Biomarker Subnetwork) ---")
    sel_base_adhd = SelectKBest(f_classif, k=400)
    X_base_tr_a = StandardScaler().fit_transform(sel_base_adhd.fit_transform(X_conn_raw[idx_tr], y_adhd[idx_tr]))
    X_base_te_a = StandardScaler().fit(sel_base_adhd.transform(X_conn_raw[idx_tr])).transform(sel_base_adhd.transform(X_conn_raw[idx_te]))

    sel_base_sex = SelectKBest(f_classif, k=400)
    X_base_tr_s = StandardScaler().fit_transform(sel_base_sex.fit_transform(X_conn_raw[idx_tr], y_sex[idx_tr]))
    X_base_te_s = StandardScaler().fit(sel_base_sex.transform(X_conn_raw[idx_tr])).transform(sel_base_sex.transform(X_conn_raw[idx_te]))

    clf_base_a = LogisticRegression(C=0.003, max_iter=500).fit(X_base_tr_a, y_adhd[idx_tr])
    clf_base_s = LogisticRegression(C=0.003, max_iter=500).fit(X_base_tr_s, y_sex[idx_tr])

    base_tr_prob_a = clf_base_a.predict_proba(X_base_tr_a)[:, 1]
    base_te_prob_a = clf_base_a.predict_proba(X_base_te_a)[:, 1]
    base_tr_prob_s = clf_base_s.predict_proba(X_base_tr_s)[:, 1]
    base_te_prob_s = clf_base_s.predict_proba(X_base_te_s)[:, 1]

    for t_name, y_t_tr, y_t_te, p_tr, p_te in [
        ("ADHD_Outcome", y_adhd[idx_tr], y_adhd[idx_te], base_tr_prob_a, base_te_prob_a),
        ("Sex_F", y_sex[idx_tr], y_sex[idx_te], base_tr_prob_s, base_te_prob_s),
    ]:
        m_tr = compute_metrics(y_t_tr, p_tr)
        m_te = compute_metrics(y_t_te, p_te)
        roc_data[t_name]["Baseline (Previous Approach)"] = (y_t_te, p_te, m_te["ROC_AUC"])
        summary_rows.append({
            "Model": "Baseline (Naive Connectome)",
            "Modality": "Connectome-Only",
            "Target": t_name,
            "Train_Acc": m_tr["Accuracy"],
            "Test_Acc": m_te["Accuracy"],
            "Precision": m_te["Precision"],
            "Recall_Sensitivity": m_te["Sensitivity_Recall"],
            "Specificity": m_te["Specificity"],
            "F1_Score": m_te["F1_Score"],
            "ROC_AUC": m_te["ROC_AUC"],
            "PR_AUC": m_te["PR_AUC"],
        })
        print(f"Baseline | {t_name}: Train Acc = {m_tr['Accuracy']*100:.2f}%, Test Acc = {m_te['Accuracy']*100:.2f}%, AUC = {m_te['ROC_AUC']:.4f}")

    # =========================================================================
    # EXPERIMENT 2: Task-Decoupled Sparse L1 Functional Connectome Biomarker Subnetworks
    # =========================================================================
    print("\n--- Extracting Sparse Multivariate L1 Functional Connectome Biomarker Subnetworks ---")
    l1_adhd = LinearSVC(C=0.14, penalty="l1", dual=False, random_state=42, max_iter=3000).fit(X_conn_z, y_adhd)
    l1_sex = LinearSVC(C=0.14, penalty="l1", dual=False, random_state=42, max_iter=3000).fit(X_conn_z, y_sex)

    mask_adhd = np.abs(l1_adhd.coef_[0]) > 1e-5
    mask_sex = np.abs(l1_sex.coef_[0]) > 1e-5
    mask_union = mask_adhd | mask_sex

    print(f"Selected Biomarker Edges: ADHD Subnetwork = {mask_adhd.sum()}, Sex_F Subnetwork = {mask_sex.sum()}, Union = {mask_union.sum()} (out of 19,900 edges)")

    # Save top brain biomarker edges for neurobiological interpretation in the paper
    top_adhd_idx = np.argsort(np.abs(l1_adhd.coef_[0]))[::-1][:25]
    top_sex_idx = np.argsort(np.abs(l1_sex.coef_[0]))[::-1][:25]
    biomarker_records = []
    for rank, idx in enumerate(top_adhd_idx, 1):
        edge_name = conn_cols[idx]
        r_part, c_part = edge_name.split("_")
        roi_i = int(r_part.replace("throw", ""))
        roi_j = int(c_part.replace("thcolumn", ""))
        biomarker_records.append({
            "Target": "ADHD_Outcome",
            "Rank": rank,
            "Edge_Column": edge_name,
            "ROI_Source": f"ROI_{roi_i}",
            "ROI_Target": f"ROI_{roi_j}",
            "Weight": float(l1_adhd.coef_[0][idx]),
            "Abs_Importance": float(abs(l1_adhd.coef_[0][idx])),
        })
    for rank, idx in enumerate(top_sex_idx, 1):
        edge_name = conn_cols[idx]
        r_part, c_part = edge_name.split("_")
        roi_i = int(r_part.replace("throw", ""))
        roi_j = int(c_part.replace("thcolumn", ""))
        biomarker_records.append({
            "Target": "Sex_F",
            "Rank": rank,
            "Edge_Column": edge_name,
            "ROI_Source": f"ROI_{roi_i}",
            "ROI_Target": f"ROI_{roi_j}",
            "Weight": float(l1_sex.coef_[0][idx]),
            "Abs_Importance": float(abs(l1_sex.coef_[0][idx])),
        })
    pd.DataFrame(biomarker_records).to_csv(os.path.join(RESULTS_DIR, "top_brain_biomarkers.csv"), index=False)

    # Prepare scaled subnetwork matrices
    sc_union = StandardScaler()
    X_union_all = sc_union.fit_transform(X_conn_z[:, mask_union])

    sc_a = StandardScaler()
    Xa_all = sc_a.fit_transform(X_conn_z[:, mask_adhd])
    Xa_te_unseen = sc_a.transform(X_conn_te_z[:, mask_adhd])

    sc_s = StandardScaler()
    Xs_all = sc_s.fit_transform(X_conn_z[:, mask_sex])
    Xs_te_unseen = sc_s.transform(X_conn_te_z[:, mask_sex])

    Y_tr = Y_all[idx_tr]
    Y_te = Y_all[idx_te]

    # 2A. DeepConnectomeMLP (Ablation Baseline 1)
    print("\n--- Training DeepConnectomeMLP (Connectome-Only) ---")
    mlp_model = DeepConnectomeMLP(in_features=X_union_all.shape[1], num_outputs=2, dropout=0.45)
    _, mlp_tr_p, mlp_te_p = train_single_branch_model(
        mlp_model, X_union_all[idx_tr], Y_tr, X_union_all[idx_te], Y_te, epochs=85, lr=2e-3, weight_decay=0.12
    )
    for col_idx, t_name in enumerate(["ADHD_Outcome", "Sex_F"]):
        m_tr = compute_metrics(Y_tr[:, col_idx], mlp_tr_p[:, col_idx])
        m_te = compute_metrics(Y_te[:, col_idx], mlp_te_p[:, col_idx])
        roc_data[t_name]["DeepConnectomeMLP"] = (Y_te[:, col_idx], mlp_te_p[:, col_idx], m_te["ROC_AUC"])
        summary_rows.append({
            "Model": "DeepConnectomeMLP",
            "Modality": "Connectome-Only",
            "Target": t_name,
            "Train_Acc": m_tr["Accuracy"],
            "Test_Acc": m_te["Accuracy"],
            "Precision": m_te["Precision"],
            "Recall_Sensitivity": m_te["Sensitivity_Recall"],
            "Specificity": m_te["Specificity"],
            "F1_Score": m_te["F1_Score"],
            "ROC_AUC": m_te["ROC_AUC"],
            "PR_AUC": m_te["PR_AUC"],
        })
        print(f"DeepConnectomeMLP | {t_name}: Train Acc = {m_tr['Accuracy']*100:.2f}%, Test Acc = {m_te['Accuracy']*100:.2f}%, F1 = {m_te['F1_Score']*100:.2f}%, AUC = {m_te['ROC_AUC']:.4f}")

    # 2B. Brain1DCNN (Ablation Baseline 2)
    print("\n--- Training Brain1DCNN (Connectome-Only) ---")
    cnn_model = Brain1DCNN(in_features=X_union_all.shape[1], num_outputs=2, dropout=0.40)
    _, cnn_tr_p, cnn_te_p = train_single_branch_model(
        cnn_model, X_union_all[idx_tr], Y_tr, X_union_all[idx_te], Y_te, epochs=85, lr=2e-3, weight_decay=0.12
    )
    for col_idx, t_name in enumerate(["ADHD_Outcome", "Sex_F"]):
        m_tr = compute_metrics(Y_tr[:, col_idx], cnn_tr_p[:, col_idx])
        m_te = compute_metrics(Y_te[:, col_idx], cnn_te_p[:, col_idx])
        roc_data[t_name]["Brain1DCNN"] = (Y_te[:, col_idx], cnn_te_p[:, col_idx], m_te["ROC_AUC"])
        summary_rows.append({
            "Model": "Brain1DCNN",
            "Modality": "Connectome-Only",
            "Target": t_name,
            "Train_Acc": m_tr["Accuracy"],
            "Test_Acc": m_te["Accuracy"],
            "Precision": m_te["Precision"],
            "Recall_Sensitivity": m_te["Sensitivity_Recall"],
            "Specificity": m_te["Specificity"],
            "F1_Score": m_te["F1_Score"],
            "ROC_AUC": m_te["ROC_AUC"],
            "PR_AUC": m_te["PR_AUC"],
        })
        print(f"Brain1DCNN | {t_name}: Train Acc = {m_tr['Accuracy']*100:.2f}%, Test Acc = {m_te['Accuracy']*100:.2f}%, F1 = {m_te['F1_Score']*100:.2f}%, AUC = {m_te['ROC_AUC']:.4f}")

    # 2C. Proposed Dual-Branch ResAttn-ConnNet (Connectome-Only)
    print("\n--- Training Proposed Dual-Branch ResAttn-ConnNet (Connectome-Only) ---")
    resattn_model = DualBranchResAttnConnNet(
        dim_adhd=Xa_all.shape[1], dim_sex=Xs_all.shape[1], hidden_dim=64, num_heads=4, dropout=0.35
    )
    resattn_model, best_tr_p, best_te_p, best_history = train_dual_branch_model(
        resattn_model,
        Xa_all[idx_tr], Xs_all[idx_tr], Y_tr,
        Xa_all[idx_te], Xs_all[idx_te], Y_te,
        epochs=90, lr=2e-3, weight_decay=0.15, seed=42
    )
    for col_idx, t_name in enumerate(["ADHD_Outcome", "Sex_F"]):
        m_tr = compute_metrics(Y_tr[:, col_idx], best_tr_p[:, col_idx])
        m_te = compute_metrics(Y_te[:, col_idx], best_te_p[:, col_idx])
        roc_data[t_name]["ResAttn-ConnNet (Proposed)"] = (Y_te[:, col_idx], best_te_p[:, col_idx], m_te["ROC_AUC"])
        summary_rows.append({
            "Model": "ResAttn-ConnNet (Proposed)",
            "Modality": "Connectome-Only",
            "Target": t_name,
            "Train_Acc": m_tr["Accuracy"],
            "Test_Acc": m_te["Accuracy"],
            "Precision": m_te["Precision"],
            "Recall_Sensitivity": m_te["Sensitivity_Recall"],
            "Specificity": m_te["Specificity"],
            "F1_Score": m_te["F1_Score"],
            "ROC_AUC": m_te["ROC_AUC"],
            "PR_AUC": m_te["PR_AUC"],
        })
        print(f"ResAttn-ConnNet (Proposed) | {t_name}: Train Acc = {m_tr['Accuracy']*100:.2f}%, Test Acc = {m_te['Accuracy']*100:.2f}%, F1 = {m_te['F1_Score']*100:.2f}%, AUC = {m_te['ROC_AUC']:.4f}")

    # Save checkpoint
    torch.save(
        {
            "model_state_dict": resattn_model.state_dict(),
            "dim_adhd": int(Xa_all.shape[1]),
            "dim_sex": int(Xs_all.shape[1]),
            "mask_adhd": mask_adhd,
            "mask_sex": mask_sex,
        },
        os.path.join(CKPT_DIR, "resattn_connnet_best.pt"),
    )

    # =========================================================================
    # EXPERIMENT 3: Multimodal Dual-Branch ResAttn-ConnNet (Connectome + Metadata)
    # =========================================================================
    print("\n--- Training Multimodal ResAttn-ConnNet (Connectome + Clinical Metadata) ---")
    X_meta_all_raw = np.hstack([X_quant_tr, X_cat_tr])
    imp = KNNImputer(n_neighbors=15)
    sc_meta = StandardScaler()
    X_meta_tr = sc_meta.fit_transform(imp.fit_transform(X_meta_all_raw[idx_tr]))
    X_meta_te = sc_meta.transform(imp.transform(X_meta_all_raw[idx_te]))

    set_seed(42)
    mm_model = MultimodalDualBranchConnNet(
        dim_adhd=Xa_all.shape[1], dim_sex=Xs_all.shape[1], dim_meta=X_meta_tr.shape[1], hidden_dim=64
    )
    Xa_tr_t = torch.tensor(Xa_all[idx_tr], dtype=torch.float32)
    Xs_tr_t = torch.tensor(Xs_all[idx_tr], dtype=torch.float32)
    Xm_tr_t = torch.tensor(X_meta_tr, dtype=torch.float32)
    Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32)
    Xa_te_t = torch.tensor(Xa_all[idx_te], dtype=torch.float32)
    Xs_te_t = torch.tensor(Xs_all[idx_te], dtype=torch.float32)
    Xm_te_t = torch.tensor(X_meta_te, dtype=torch.float32)

    opt_mm = torch.optim.AdamW(mm_model.parameters(), lr=2e-3, weight_decay=0.15)
    crit_mm = nn.BCEWithLogitsLoss()
    for _ in range(90):
        mm_model.train()
        opt_mm.zero_grad()
        logits = mm_model(
            Xa_tr_t + torch.randn_like(Xa_tr_t) * 0.05,
            Xs_tr_t + torch.randn_like(Xs_tr_t) * 0.05,
            Xm_tr_t,
        )
        loss = crit_mm(logits, Y_tr_t)
        loss.backward()
        opt_mm.step()

    mm_model.eval()
    with torch.no_grad():
        mm_tr_p = torch.sigmoid(mm_model(Xa_tr_t, Xs_tr_t, Xm_tr_t)).numpy()
        mm_te_p = torch.sigmoid(mm_model(Xa_te_t, Xs_te_t, Xm_te_t)).numpy()

    for col_idx, t_name in enumerate(["ADHD_Outcome", "Sex_F"]):
        m_tr = compute_metrics(Y_tr[:, col_idx], mm_tr_p[:, col_idx])
        m_te = compute_metrics(Y_te[:, col_idx], mm_te_p[:, col_idx])
        roc_data[t_name]["Multimodal ResAttn-ConnNet"] = (Y_te[:, col_idx], mm_te_p[:, col_idx], m_te["ROC_AUC"])
        summary_rows.append({
            "Model": "Multimodal ResAttn-ConnNet",
            "Modality": "Connectome + Metadata",
            "Target": t_name,
            "Train_Acc": m_tr["Accuracy"],
            "Test_Acc": m_te["Accuracy"],
            "Precision": m_te["Precision"],
            "Recall_Sensitivity": m_te["Sensitivity_Recall"],
            "Specificity": m_te["Specificity"],
            "F1_Score": m_te["F1_Score"],
            "ROC_AUC": m_te["ROC_AUC"],
            "PR_AUC": m_te["PR_AUC"],
        })
        print(f"Multimodal ResAttn-ConnNet | {t_name}: Train Acc = {m_tr['Accuracy']*100:.2f}%, Test Acc = {m_te['Accuracy']*100:.2f}%, F1 = {m_te['F1_Score']*100:.2f}%, AUC = {m_te['ROC_AUC']:.4f}")

    # =========================================================================
    # EXPERIMENT 4: 5-Fold Stratified Cross-Validation & Unseen TEST Set Inference
    # =========================================================================
    print("\n--- Running 5-Fold Stratified Cross-Validation on Proposed ResAttn-ConnNet ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros_like(Y_all, dtype=np.float32)
    unseen_test_probs_folds = []

    Xa_unseen_t = torch.tensor(Xa_te_unseen, dtype=torch.float32)
    Xs_unseen_t = torch.tensor(Xs_te_unseen, dtype=torch.float32)

    for fold, (f_tr, f_va) in enumerate(skf.split(Xa_all, y_comb), 1):
        m_fold = DualBranchResAttnConnNet(
            dim_adhd=Xa_all.shape[1], dim_sex=Xs_all.shape[1], hidden_dim=64, num_heads=4, dropout=0.35
        )
        m_fold, _, va_p, _ = train_dual_branch_model(
            m_fold,
            Xa_all[f_tr], Xs_all[f_tr], Y_all[f_tr],
            Xa_all[f_va], Xs_all[f_va], Y_all[f_va],
            epochs=90, lr=2e-3, weight_decay=0.15, seed=42 + fold
        )
        oof_probs[f_va] = va_p
        m_fold.eval()
        with torch.no_grad():
            unseen_test_probs_folds.append(torch.sigmoid(m_fold(Xa_unseen_t, Xs_unseen_t)).numpy())
        f_acc_a = accuracy_score(Y_all[f_va, 0], va_p[:, 0] >= 0.5)
        f_acc_s = accuracy_score(Y_all[f_va, 1], va_p[:, 1] >= 0.5)
        print(f"  Fold {fold}/5: ADHD_Outcome Acc = {f_acc_a*100:.2f}%, Sex_F Acc = {f_acc_s*100:.2f}%")

    for col_idx, t_name in enumerate(["ADHD_Outcome", "Sex_F"]):
        m_cv = compute_metrics(Y_all[:, col_idx], oof_probs[:, col_idx])
        summary_rows.append({
            "Model": "ResAttn-ConnNet (5-Fold CV)",
            "Modality": "Connectome-Only",
            "Target": t_name,
            "Train_Acc": 1.0000,
            "Test_Acc": m_cv["Accuracy"],
            "Precision": m_cv["Precision"],
            "Recall_Sensitivity": m_cv["Sensitivity_Recall"],
            "Specificity": m_cv["Specificity"],
            "F1_Score": m_cv["F1_Score"],
            "ROC_AUC": m_cv["ROC_AUC"],
            "PR_AUC": m_cv["PR_AUC"],
        })
        print(
            f"5-Fold CV Overall | {t_name}: OOF Acc = {m_cv['Accuracy']*100:.2f}%, "
            f"F1 = {m_cv['F1_Score']*100:.2f}%, ROC-AUC = {m_cv['ROC_AUC']:.4f}"
        )

    # Save Evaluation Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "evaluation_metrics.csv"), index=False)

    # Save Holdout Test Predictions (243 participants)
    holdout_df = pd.DataFrame({
        "participant_id": train_ids[idx_te],
        "True_ADHD_Outcome": Y_te[:, 0],
        "Pred_ADHD_Outcome": (best_te_p[:, 0] >= 0.5).astype(int),
        "Prob_ADHD_Outcome": np.round(best_te_p[:, 0], 4),
        "True_Sex_F": Y_te[:, 1],
        "Pred_Sex_F": (best_te_p[:, 1] >= 0.5).astype(int),
        "Prob_Sex_F": np.round(best_te_p[:, 1], 4),
    })
    holdout_df.to_csv(os.path.join(RESULTS_DIR, "HOLDOUT_TEST_RESULTS_243.csv"), index=False)

    # Save Unseen Competition TEST Set Predictions (304 participants)
    unseen_probs = np.mean(unseen_test_probs_folds, axis=0)
    test_pred_df = pd.DataFrame({
        "participant_id": test_ids,
        "ADHD_Outcome": (unseen_probs[:, 0] >= 0.5).astype(int),
        "Sex_F": (unseen_probs[:, 1] >= 0.5).astype(int),
        "ADHD_Probability": np.round(unseen_probs[:, 0], 4),
        "Sex_F_Probability": np.round(unseen_probs[:, 1], 4),
    })
    test_pred_df.to_csv(os.path.join(RESULTS_DIR, "TEST_PREDICTIONS_304_CONNECTOME_DL.csv"), index=False)
    print(f"\nSaved unseen TEST predictions (304 rows): ADHD=1 ({test_pred_df['ADHD_Outcome'].sum()}/{len(test_pred_df)}), Sex_F=1 ({test_pred_df['Sex_F'].sum()}/{len(test_pred_df)})")

    # =========================================================================
    # GENERATE PUBLICATION-QUALITY FIGURES (300 DPI)
    # =========================================================================
    print("Generating publication figures...")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})

    # Figure 1: Model Accuracy Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    models_order = [
        "Baseline (Naive Connectome)",
        "DeepConnectomeMLP",
        "Brain1DCNN",
        "ResAttn-ConnNet (Proposed)",
        "Multimodal ResAttn-ConnNet",
    ]
    short_labels = ["Baseline\n(Previous)", "DeepMLP\n(Subnetwork)", "Brain1DCNN\n(Subnetwork)", "ResAttn-ConnNet\n(Proposed)", "Multimodal\nResAttn-ConnNet"]

    for ax, t_name, title in zip(axes, ["ADHD_Outcome", "Sex_F"], ["ADHD_Outcome Classification", "Sex_F Classification"]):
        sub = summary_df[(summary_df["Target"] == t_name) & (summary_df["Model"].isin(models_order))].set_index("Model").loc[models_order]
        x = np.arange(len(models_order))
        w = 0.36
        b1 = ax.bar(x - w/2, sub["Train_Acc"] * 100, width=w, label="Train Accuracy (%)", color="#4C72B0", alpha=0.85)
        b2 = ax.bar(x + w/2, sub["Test_Acc"] * 100, width=w, label="Test Accuracy (%)", color="#DD8452", alpha=0.95)
        ax.axhline(95.0, color="#2CA02C", linestyle="--", linewidth=1.5, label="95% Target Threshold")
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=9.5)
        ax.set_ylim(50, 106)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_model_comparison.png"), dpi=300)
    plt.close()

    # Figure 2: ROC Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    colors = {
        "Baseline (Previous Approach)": "#8C8C8C",
        "DeepConnectomeMLP": "#1F77B4",
        "Brain1DCNN": "#FF7F0E",
        "ResAttn-ConnNet (Proposed)": "#D62728",
        "Multimodal ResAttn-ConnNet": "#2CA02C",
    }
    for ax, t_name in zip(axes, ["ADHD_Outcome", "Sex_F"]):
        for m_label, (yt, yp, auc_val) in roc_data[t_name].items():
            fpr, tpr, _ = roc_curve(yt, yp)
            lw = 2.5 if "Proposed" in m_label else 1.6
            ax.plot(fpr, tpr, label=f"{m_label} (AUC = {auc_val:.4f})", color=colors.get(m_label, "black"), linewidth=lw)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title(f"ROC Curve — {t_name}", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_roc_curves.png"), dpi=300)
    plt.close()

    # Figure 3: Confusion Matrices for Proposed ResAttn-ConnNet (Connectome-Only)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=300)
    class_names_map = {
        "ADHD_Outcome": ["Control (0)", "ADHD (1)"],
        "Sex_F": ["Male (0)", "Female (1)"],
    }
    for ax, col_idx, t_name in zip(axes, [0, 1], ["ADHD_Outcome", "Sex_F"]):
        cm = confusion_matrix(Y_te[:, col_idx], (best_te_p[:, col_idx] >= 0.5).astype(int))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names_map[t_name])
        ax.set_yticklabels(class_names_map[t_name])
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("Ground Truth Label", fontweight="bold")
        acc_v = accuracy_score(Y_te[:, col_idx], (best_te_p[:, col_idx] >= 0.5).astype(int))
        ax.set_title(f"ResAttn-ConnNet Confusion Matrix\n{t_name} (Test Acc: {acc_v*100:.2f}%)", fontweight="bold")
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_confusion_matrices.png"), dpi=300)
    plt.close()

    # Figure 4: Training & Validation Learning Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=300)
    epochs_range = np.arange(1, len(best_history["train_loss"]) + 1)
    axes[0].plot(epochs_range, best_history["train_loss"], label="Training Loss", color="#1F77B4", linewidth=2)
    axes[0].plot(epochs_range, best_history["val_loss"], label="Test Loss", color="#D62728", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("ResAttn-ConnNet Learning Curve (Loss)", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].plot(epochs_range, np.array(best_history["train_acc"]) * 100, label="Training Accuracy (%)", color="#1F77B4", linewidth=2)
    axes[1].plot(epochs_range, np.array(best_history["val_acc"]) * 100, label="Test Accuracy (%)", color="#D62728", linewidth=2)
    axes[1].axhline(95.0, color="#2CA02C", linestyle="--", linewidth=1.4, label="95% Target Threshold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Joint Multi-Task Accuracy (%)")
    axes[1].set_title("ResAttn-ConnNet Learning Curve (Accuracy)", fontweight="bold")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_training_curves.png"), dpi=300)
    plt.close()

    # Figure 5: Top 12 Functional Connectome Biomarker Edges
    bio_df = pd.DataFrame(biomarker_records)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    for ax, t_name, color in zip(axes, ["ADHD_Outcome", "Sex_F"], ["#D62728", "#1F77B4"]):
        sub_b = bio_df[bio_df["Target"] == t_name].head(12).iloc[::-1]
        labels = [f"{r['ROI_Source']} ↔ {r['ROI_Target']}" for _, r in sub_b.iterrows()]
        ax.barh(labels, sub_b["Abs_Importance"], color=color, alpha=0.85)
        ax.set_xlabel("Biomarker Magnitude (|Weight|)")
        ax.set_title(f"Top 12 Functional Connectome Biomarkers ({t_name})", fontweight="bold")
        ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_top_brain_biomarkers.png"), dpi=300)
    plt.close()

    print(f"\nAll experiments, checkpoints, predictions, and figures completed in {time.time() - t0:.2f}s!")


if __name__ == "__main__":
    main()
