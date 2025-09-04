import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from model_credit import build_model


def main():
    # 1. Muat data terproses
    data = np.load("data/german_scaled.npz")
    X, y = data["X"], data["y"]
    # Ubah label: 1=good→1, 2=bad→0
    y = (y == 1).astype(int)

    # 2. Split: 70% train, 15% val, 15% test
    n = len(y)
    idx = np.arange(n)
    np.random.shuffle(idx)
    t, v = int(0.7 * n), int(0.85 * n)
    X_train, y_train = X[idx[:t]], y[idx[:t]]
    X_val, y_val = X[idx[t:v]], y[idx[t:v]]
    X_test, y_test = X[idx[v:]], y[idx[v:]]

    # 3. Bangun dan kompilasi model
    model = build_model(X.shape[1])
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["AUC"]
    )

    # 4. Latih model
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=2,
    )

    # 5. Evaluasi pada test set
    p = model.predict(X_test).ravel()
    auc = roc_auc_score(y_test, p)
    tn, fp, fn, tp = confusion_matrix(y_test, (p > 0.5).astype(int)).ravel()
    fpr, fnr = fp / (fp + tn), fn / (fn + tp)
    print(f"✔️ Test AUC: {auc:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}")

    # 6. Simpan model
    model.save("models/credit_model.keras")
    print("✔️ Model tersimpan: models/credit_model.keras")


if __name__ == "__main__":
    main()
