import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    # Baca data
    df = pd.read_csv("data/german.data-numeric", sep="\\s+", header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values  # 1=good, 2=bad

    # Standarisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Simpan scaler dan data
    joblib.dump(scaler, "models/credit_scaler.joblib")
    np.savez_compressed("data/german_scaled.npz", X=X_scaled, y=y)
    print("✔️ Preprocessing selesai")

if __name__ == "__main__":
    main()
