import pandas as pd
from model import build_model


def main():
    df = pd.read_csv("data/sintesis.csv")
    x = df[["fitur1", "fitur2"]].values
    y = df["label"].values

    print("Mulai training...")
    model = build_model(input_shape=(2,))
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    model.fit(x, y, epochs=5, batch_size=32, verbose=1)
    model.save("models/icame_model.keras")
    print("Training selesai. Model disimpan di models/icame_model.keras")


if __name__ == "__main__":
    main()
