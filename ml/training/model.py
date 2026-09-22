"""CNN architecture for five-class facial expression recognition.

Design decisions:
- 4-block conv stack (32→64→128→256 filters) with BN + ReLU + MaxPool
- Spatial dropout after each block (more effective than standard dropout for conv layers)
- Global Average Pooling instead of Flatten — fewer parameters, less overfitting
- Dense head: 512 → 256 → 5 (softmax)
- Input: (48, 48, 1) grayscale, values in [0, 1]

The architecture is importable standalone — no dependency on training data or config files.
Class count and input shape are parameters so the factory is reusable.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_emotion_cnn(
    num_classes: int = 5,
    input_shape: tuple[int, int, int] = (48, 48, 1),
    dropout_rate: float = 0.4,
    spatial_dropout_rate: float = 0.2,
) -> keras.Model:
    """Build and return the EmotionAI CNN model (uncompiled).

    Parameters
    ----------
    num_classes : int
        Number of output classes. Must match the class mapping in shared/labels.json.
    input_shape : tuple
        (height, width, channels). Default 48×48 grayscale.
    dropout_rate : float
        Dropout rate for the dense classification head.
    spatial_dropout_rate : float
        SpatialDropout2D rate applied after each conv block.

    Returns
    -------
    keras.Model
        Uncompiled model. Caller is responsible for .compile().
    """
    inp = keras.Input(shape=input_shape, name="image_input")

    # ── Block 1: 32 filters ───────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1_1")(inp)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.ReLU(name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.ReLU(name="relu1_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.SpatialDropout2D(spatial_dropout_rate, name="sdrop1")(x)

    # ── Block 2: 64 filters ───────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.ReLU(name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.ReLU(name="relu2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.SpatialDropout2D(spatial_dropout_rate, name="sdrop2")(x)

    # ── Block 3: 128 filters ──────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.ReLU(name="relu3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.ReLU(name="relu3_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.SpatialDropout2D(spatial_dropout_rate, name="sdrop3")(x)

    # ── Block 4: 256 filters ──────────────────────────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.ReLU(name="relu4_1")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv4_2")(x)
    x = layers.BatchNormalization(name="bn4_2")(x)
    x = layers.ReLU(name="relu4_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)
    x = layers.SpatialDropout2D(spatial_dropout_rate, name="sdrop4")(x)

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, use_bias=False, name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc1")(x)
    x = layers.ReLU(name="relu_fc1")(x)
    x = layers.Dropout(dropout_rate, name="drop_fc1")(x)

    x = layers.Dense(256, use_bias=False, name="fc2")(x)
    x = layers.BatchNormalization(name="bn_fc2")(x)
    x = layers.ReLU(name="relu_fc2")(x)
    x = layers.Dropout(dropout_rate / 2, name="drop_fc2")(x)

    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs=inp, outputs=out, name="emotion_cnn")


def model_summary_str(model: keras.Model) -> str:
    """Return model summary as a string (useful for logging)."""
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)
