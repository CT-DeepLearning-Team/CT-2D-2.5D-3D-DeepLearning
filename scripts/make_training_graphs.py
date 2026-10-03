import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


CURVE_DIR = Path(r"C:\CT-Dimensionality-Study\outputs\curves")

models = {
    "Model A": "model_A_history.npz",
    "Model B": "model_B_history.npz",
    "Model C": "model_C_history.npz",
    "Binary": "binary_model_history.npz"
}


for model_name, filename in models.items():

    history = np.load(CURVE_DIR / filename)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_f1 = history["val_macro_f1"]

    epochs = np.arange(1, len(train_loss) + 1)

    best_epoch = int(np.argmax(val_f1)) + 1
    best_f1 = float(np.max(val_f1))

    # Training and validation loss
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_loss,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"{model_name} - Training and Validation Loss"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_file = (
        CURVE_DIR /
        f"{model_name.replace(' ', '_')}_Loss.png"
    )

    plt.savefig(loss_file, dpi=200)
    plt.close()


    # Validation Macro-F1
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        val_f1,
        marker="o",
        label="Validation Macro-F1"
    )

    plt.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best Epoch ({best_epoch})"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(
        f"{model_name} - Validation Macro-F1"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    f1_file = (
        CURVE_DIR /
        f"{model_name.replace(' ', '_')}_Validation_MacroF1.png"
    )

    plt.savefig(f1_file, dpi=200)
    plt.close()

    print(
        f"{model_name}: "
        f"Best epoch = {best_epoch}, "
        f"Best validation Macro-F1 = {best_f1:.4f}"
    )


print()
print("All training graphs created.")
print("Saved in:")
print(CURVE_DIR)