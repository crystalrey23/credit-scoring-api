import numpy as np
import joblib
from tensorflow import keras

# Muat scaler dan model final
scaler = joblib.load("models/credit_scaler.joblib")
model = keras.models.load_model("models/credit_model_final.keras")

# Threshold untuk klasifikasi
THRESHOLD = 0.60


def predict_credit(features):
    """
    features: array-like shape (n_features,)
    Returns: dict with keys 'probability' dan 'label'
    """
    # Ubah ke array 2D
    x = np.array(features).reshape(1, -1)
    # Standarisasi
    x_scaled = scaler.transform(x)
    # Hitung probabilitas
    prob = model.predict(x_scaled).ravel()[0]
    # Tentukan label
    label = int(prob >= THRESHOLD)
    return {"probability": float(prob), "label": label}


if __name__ == "__main__":
    # Contoh pemakaian dengan data dummy
    sample = [
        float(x) for x in input("Masukkan fitur (dipisah koma): ").split(",")
    ]
    result = predict_credit(sample)
    print(f"Probability (good credit): {result['probability']:.4f}")
    print(f"Label (1=good, 0=bad) at threshold {THRESHOLD}: {result['label']}")
