import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow import keras


def evaluate_thresholds(y_true, y_scores, thresholds):
    results = []
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        results.append((t, fpr, fnr))
    return results


if __name__ == "__main__":
    # Muat data terproses
    data = np.load("data/german_scaled.npz")
    X, y = data["X"], data["y"]
    y = (y == 1).astype(int)
    # Split ulang untuk test set (sama dengan train_credit)
    n = len(y)
    idx = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(idx)
    t, v = int(0.7 * n), int(0.85 * n)
    X_test, y_test = X[idx[v:]], y[idx[v:]]
    # Muat model
    model = keras.models.load_model("models/credit_model.keras")
    # Hitung probabilitas
    y_scores = model.predict(X_test).ravel()
    # Evaluasi berbagai threshold
    thresholds = np.linspace(0.1, 0.9, 17)
    results = evaluate_thresholds(y_test, y_scores, thresholds)
    # Cetak hasil
    print("Threshold |  FPR  |  FNR")
    for t, fpr, fnr in results:
        print(f"{t:.2f}      | {fpr:.3f} | {fnr:.3f}")
