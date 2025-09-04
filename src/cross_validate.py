import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from model_credit import build_model

def cv_evaluate(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    aucs = []
    for train_idx, test_idx in kf.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]
        model = build_model(X.shape[1])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        y_scores = model.predict(X_test).ravel()
        aucs.append(roc_auc_score(y_test, y_scores))
    return aucs

if __name__ == "__main__":
    data = np.load("data/german_scaled.npz")
    X, y = data["X"], data["y"]
    y = (y == 1).astype(int)
    aucs = cv_evaluate(X, y, k=5)
    print(f"5-Fold AUC scores: {aucs}")
    print(f"Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
