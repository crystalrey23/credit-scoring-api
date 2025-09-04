import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


# Fungsi untuk membangun model dengan parameter dinamis
def build_tuned_model(input_dim, units, lr):
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(input_dim,)),
            keras.layers.Dense(units, activation="relu"),
            keras.layers.Dense(units // 2, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    return model


if __name__ == "__main__":
    # Muat data terproses
    data = np.load("data/german_scaled.npz")
    X, y = data["X"], data["y"]
    y = (y == 1).astype(int)

    # Bagi data: 80% train, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Grid parameter
    param_grid = {"units": [32, 64], "lr": [1e-3, 1e-4]}

    best_auc = 0
    best_params = None

    # Grid search
    for units in param_grid["units"]:
        for lr in param_grid["lr"]:
            model = build_tuned_model(X.shape[1], units, lr)
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            y_scores = model.predict(X_val).ravel()
            auc = roc_auc_score(y_val, y_scores)
            print(f"Units={units}, LR={lr:.4f} -> Val AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_params = (units, lr)

    print(
        f"✔️ Best Val AUC: {best_auc:.4f} "
        f"with units={best_params[0]}, lr={best_params[1]:.4f}"
    )
