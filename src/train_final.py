import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from model_credit import build_model

def main():
    # Muat data terproses
    data = np.load("data/german_scaled.npz")
    X, y = data["X"], data["y"]
    y = (y == 1).astype(int)

    # Split: 85% train+val, 15% test
    n = len(y)
    idx = np.arange(n); np.random.seed(0); np.random.shuffle(idx)
    split = int(0.85 * n)
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # Bangun model dengan parameter terbaik
    model = build_model(X.shape[1])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="binary_crossentropy",
                  metrics=["AUC"])
    # Latih pada data train+val
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2)

    # Evaluasi pada test set
    y_scores = model.predict(X_test).ravel()
    auc = roc_auc_score(y_test, y_scores)
    # Threshold 0.60
    y_pred = (y_scores >= 0.60).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr, fnr = fp/(fp+tn), fn/(fn+tp)

    print(f"✔️ Final Test AUC: {auc:.4f}")
    print(f"✔️ Final FPR (thr=0.60): {fpr:.4f}, FNR: {fnr:.4f}")

    # Simpan model
    model.save("models/credit_model_final.keras")
    print("✔️ Model final disimpan: models/credit_model_final.keras")

if __name__ == "__main__":
    main()
