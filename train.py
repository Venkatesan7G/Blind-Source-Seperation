import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # reduce TF logs

import tensorflow as tf

# --- GPU setup ---
gpus = tf.config.list_physical_devices("GPU")
print("GPUs detected:", gpus)

if gpus:
    try:
        # Use first GPU only (optional)
        tf.config.set_visible_devices(gpus[0], "GPU")

        # Prevent TF from allocating all VRAM at once
        tf.config.experimental.set_memory_growth(gpus[0], True)

        print("✅ GPU enabled:", gpus[0])
    except Exception as e:
        print("⚠️ GPU setup failed, will run on CPU. Error:", e)
else:
    print("⚠️ No GPU detected by TensorFlow. Running on CPU.")
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_generator import BSSMaskGenerator
from model import build_bss_mask_model

def main():
    os.makedirs("checkpoints", exist_ok=True)

    train_gen = BSSMaskGenerator("train", batch_size=8, shuffle=True)
    val_gen   = BSSMaskGenerator("val", batch_size=8, shuffle=False)

    model = build_bss_mask_model(freq_bins=256, time_frames=620, in_ch=5, base=32)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )

    callbacks = [
        ModelCheckpoint("checkpoints/best.keras", monitor="val_loss", save_best_only=True, mode="min"),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=callbacks,
        verbose=2
    )

    model.save("checkpoints/final.keras")
    print("Saved model to checkpoints/final.keras")

if __name__ == "__main__":
    main()
