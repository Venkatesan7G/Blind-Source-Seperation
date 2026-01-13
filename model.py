import tensorflow as tf
from tensorflow.keras import layers, models

# Keras 3-safe ops (avoid tf ops on KerasTensor)
try:
    from keras import ops as Kops
except Exception:
    Kops = None


def conv_block(x, filters: int):
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def build_bss_mask_model(freq_bins=256, time_frames=620, in_ch=5, base=32):
    """
    TF-domain mask estimator.

    Input:
      feats: (F, T, 5)  = [mag1, mag2, cos(IPD), sin(IPD), ILD]
    Output:
      masks: (F, T, 2)  = [M1, M2], each in [0,1], normalized so M1+M2=1 per TF-bin.
    """
    inp = layers.Input(shape=(freq_bins, time_frames, in_ch), name="feats")

    # Encoder
    c1 = conv_block(inp, base)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, base * 2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    b = conv_block(p2, base * 4)

    # Decoder
    u2 = layers.UpSampling2D((2, 2))(b)
    u2 = layers.Conv2D(base * 2, (2, 2), padding="same", activation="relu")(u2)
    m2 = layers.Concatenate()([u2, c2])
    c3 = conv_block(m2, base * 2)

    u1 = layers.UpSampling2D((2, 2))(c3)
    u1 = layers.Conv2D(base, (2, 2), padding="same", activation="relu")(u1)
    m1 = layers.Concatenate()([u1, c1])
    c4 = conv_block(m1, base)

    # Raw masks
    raw = layers.Conv2D(2, (1, 1), padding="same", activation="sigmoid", name="raw_masks")(c4)

    # Normalize masks with Keras-safe ops
    if Kops is not None:
        denom = Kops.sum(raw, axis=-1, keepdims=True) + 1e-8
        masks = raw / denom
    else:
        # Fallback if keras.ops not available
        def normalize(t):
            denom = tf.reduce_sum(t, axis=-1, keepdims=True) + 1e-8
            return t / denom
        masks = layers.Lambda(normalize, name="mask_norm")(raw)

    model = models.Model(inp, masks, name="TF_Stereo_BSS_MaskNet")
    return model


def main():
    """
    Optional self-test: build the model and run one forward pass.
    """
    print("Building model for self-test...")
    m = build_bss_mask_model(freq_bins=256, time_frames=620, in_ch=5, base=32)
    m.summary()

    # dummy batch (B=2)
    x = tf.random.normal((2, 256, 620, 5))
    y = m(x)
    print("Forward pass OK. Output shape:", y.shape)
    # sanity: masks sum close to 1
    s = tf.reduce_mean(tf.reduce_sum(y, axis=-1))
    print("Mean(sum masks) =", float(s))


if __name__ == "__main__":
    main()
