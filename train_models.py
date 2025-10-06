# train_models.py
from pathlib import Path
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve)
from xgboost import XGBClassifier
from preprocess import preprocess

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_metrics(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return dict(
        Accuracy=accuracy_score(y_true, pred),
        Precision=precision_score(y_true, pred, zero_division=0),
        Recall=recall_score(y_true, pred, zero_division=0),
        F1=f1_score(y_true, pred, zero_division=0),
        ROC_AUC=roc_auc_score(y_true, proba),
        PR_AUC=average_precision_score(y_true, proba),
        Confusion_Matrix=confusion_matrix(y_true, pred).tolist()
    )

def train_and_eval(name, model, X_train, y_train, X_test, y_test, pos_ratio):
    t0 = time.time()
    if isinstance(model, XGBClassifier):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        spw = (1 - pos_ratio) / pos_ratio
        model.set_params(scale_pos_weight=spw)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            early_stopping_rounds=100,
            verbose=False
        )
        best_n = getattr(model, "best_iteration", None)
        if best_n is not None:
            model.set_params(n_estimators=best_n + 1)
        model.fit(X_train, y_train, verbose=False)
    else:
        model.fit(X_train, y_train)
    train_s = time.time() - t0

    t1 = time.time()
    proba = (model.predict_proba(X_test)[:, 1]
             if hasattr(model, "predict_proba")
             else model.decision_function(X_test))
    pred_s = time.time() - t1

    met = compute_metrics(y_test, proba)
    met.update(Model=name, Train_Time_s=train_s, Predict_Time_s=pred_s)
    return met, proba

def main():
    _, X_tr, X_te, y_tr, y_te = preprocess()
    pos_ratio = y_tr.mean()

    models = {
        "Logistic Regression": LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            max_iter=1000, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=1.0,
            reg_lambda=1.0, objective="binary:logistic", tree_method="hist",
            n_jobs=-1, random_state=42
        ),
    }

    rows, curves = [], {}
    for name, mdl in models.items():
        res, proba = train_and_eval(name, mdl, X_tr, y_tr, X_te, y_te, pos_ratio)
        rows.append(res); curves[name] = (y_te.values, proba)

    df = pd.DataFrame(rows)[
        ["Model","Accuracy","Precision","Recall","F1","ROC_AUC","PR_AUC",
         "Train_Time_s","Predict_Time_s","Confusion_Matrix"]
    ].sort_values("ROC_AUC", ascending=False)
    (OUTPUT_DIR / "results.csv").write_text(df.to_csv(index=False))
    print(df.drop(columns=["Confusion_Matrix"]).round(4).to_string(index=False))


if True:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    for name, (yt, pr) in curves.items():
        fpr, tpr, _ = roc_curve(yt, pr)
        auc = roc_auc_score(yt, pr)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", lw=1)
    ax[0].set_title("ROC")
    ax[0].legend()

    any_y = next(iter(curves.values()))[0]
    base = any_y.mean()
    for name, (yt, pr) in curves.items():
        p, r, _ = precision_recall_curve(yt, pr)
        ap = average_precision_score(yt, pr)
        ax[1].plot(r, p, label=f"{name} (AP={ap:.3f})")
    ax[1].axhline(base, ls="--", lw=1, label=f"Baseline={base:.3f}")
    ax[1].set_title("Precision-Recall")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "curves.png", dpi=300)
