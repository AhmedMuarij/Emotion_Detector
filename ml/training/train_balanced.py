"""Balanced Training script for EmotionAI CNN.

Key Features:
1. Balanced dataset: 3,000 samples per class for all 5 emotions (15,000 total).
   - Surprise class (2,695 source samples) augmented up to 3,000.
   - Other classes downsampled randomly to exactly 3,000 to prevent class bias.
2. Dynamic online data augmentation via ImageDataGenerator:
   - Horizontal flip (facial symmetry)
   - Rotation (+/- 10 deg)
   - Width/Height shift (8%)
   - Zoom (8%)
3. Optimization:
   - Adam optimizer with initial lr=1e-3
   - ReduceLROnPlateau (halves lr if val_loss plateaus for 3 epochs)
   - ModelCheckpoint (always saves best model based on val_accuracy)
   - EarlyStopping with patience=7
4. Post-training evaluation:
   - Evaluates on the held-out test set
   - Computes per-class accuracy and confusion metrics
   - Updates models/emotion_cnn.keras, models/metadata.json, and data/reports/
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

# Force unbuffered output so logs appear in real-time
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "data" / "reports"
SHARED_LABELS = REPO_ROOT / "shared" / "labels.json"

# Ensure repo root is on Python path so 'ml' package can be imported
sys.path.insert(0, str(REPO_ROOT))


def balance_dataset(X: np.ndarray, y: np.ndarray, target_per_class: int = 3000, seed: int = 42):
    """Ensure exactly target_per_class samples for each class."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    np.random.seed(seed)
    unique_classes = np.unique(y)
    balanced_X = []
    balanced_y = []
    
    aug = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        fill_mode="nearest",
    )
    
    print(f"\n[Data Balancing] Target per class: {target_per_class}")
    for c in unique_classes:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        print(f"  Class {c}: original = {n_samples} samples")
        
        if n_samples >= target_per_class:
            chosen = np.random.choice(idx, size=target_per_class, replace=False)
            balanced_X.append(X[chosen])
            balanced_y.append(y[chosen])
        else:
            # Need to augment
            needed = target_per_class - n_samples
            chosen = X[idx]
            chosen_y = y[idx]
            balanced_X.append(chosen)
            balanced_y.append(chosen_y)
            
            # Augment remaining
            gen = aug.flow(chosen, chosen_y, batch_size=needed, shuffle=True)
            aug_x, aug_y = next(gen)
            balanced_X.append(aug_x[:needed])
            balanced_y.append(aug_y[:needed])
            print(f"    -> Augmented +{needed} samples for class {c}")
            
    X_out = np.concatenate(balanced_X, axis=0)
    y_out = np.concatenate(balanced_y, axis=0)
    
    # Shuffle combined dataset
    perm = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def main(epochs: int = 25, batch_size: int = 64, target_per_class: int = 3000):
    print("=" * 60)
    print("  EmotionAI — Balanced Dataset Training (25 Epochs)")
    print("=" * 60, flush=True)
    
    import tensorflow as tf
    from ml.training.model import build_emotion_cnn
    
    # Load labels
    with open(SHARED_LABELS, encoding="utf-8") as f:
        labels = json.load(f)
    num_classes = int(labels["num_classes"])
    index_to_class = labels["index_to_class"]
    
    # Load raw preprocessed arrays
    print(f"\nLoading arrays from {PROCESSED_DIR}...")
    X_train_raw = np.load(PROCESSED_DIR / "X_train.npy")
    y_train_raw = np.load(PROCESSED_DIR / "y_train.npy")
    X_val = np.load(PROCESSED_DIR / "X_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    
    # Balance training set to 3000 images per class
    X_train, y_train = balance_dataset(X_train_raw, y_train_raw, target_per_class=target_per_class)
    print(f"\nFinal balanced training set shape: {X_train.shape}, labels: {y_train.shape}")
    for c in range(num_classes):
        print(f"  {index_to_class[str(c)]} ({c}): {np.sum(y_train == c)} samples")
        
    # Setup online dynamic data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        fill_mode="nearest",
    )
    datagen.fit(X_train)
    train_gen = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
    steps_per_epoch = len(X_train) // batch_size
    
    # Build Model
    print("\nBuilding EmotionAI CNN architecture...")
    model = build_emotion_cnn(num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Total Parameters: {model.count_params():,}")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / "emotion_cnn.keras"
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    
    print(f"\nStarting training for {epochs} epochs (steps per epoch: {steps_per_epoch})...", flush=True)
    t0 = time.time()
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )
    total_time = time.time() - t0
    print(f"\nTraining completed in {total_time/60:.1f} minutes!", flush=True)
    
    # Load best checkpoint
    print(f"Loading best checkpoint from {checkpoint_path}...")
    best_model = tf.keras.models.load_model(str(checkpoint_path))
    
    # Evaluate on held-out test set
    print("\nEvaluating on held-out test set...")
    test_loss, test_acc = best_model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Test Loss:     {test_loss:.4f}")
    
    # Per-class evaluation
    test_preds = best_model.predict(X_test, batch_size=batch_size, verbose=0)
    pred_classes = np.argmax(test_preds, axis=1)
    
    per_class_acc = {}
    print("\nPer-class accuracy breakdown:")
    for c in range(num_classes):
        name = index_to_class[str(c)]
        mask = (y_test == c)
        c_acc = float(np.mean(pred_classes[mask] == c)) if np.sum(mask) > 0 else 0.0
        per_class_acc[name] = round(c_acc, 4)
        print(f"  {name:10s}: {c_acc*100:.1f}% ({np.sum(mask)} test samples)")
        
    # Update metadata.json
    metadata = {
        "model_name": "emotion_cnn",
        "model_version": "0.2.0",
        "model_file": "emotion_cnn.keras",
        "framework": "tensorflow.keras",
        "task": "facial_expression_recognition",
        "disclaimer": "Classifies visible facial expressions. Does not determine internal emotional state.",
        "input": {
            "height": 48,
            "width": 48,
            "channels": 1,
            "color_mode": "grayscale",
            "normalization": "divide_by_255",
            "dtype": "float32",
        },
        "num_classes": num_classes,
        "class_to_index": labels["class_to_index"],
        "index_to_class": index_to_class,
        "training": {
            "dataset": "FER2013",
            "balanced_samples_per_class": target_per_class,
            "total_train_samples": len(X_train),
            "epochs_planned": epochs,
            "epochs_run": len(history.history["loss"]),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "augmentation": "horizontal_flip, rotation, zoom, shift",
        },
        "evaluation": {
            "test_accuracy": round(float(test_acc), 4),
            "test_loss": round(float(test_loss), 4),
            "best_val_accuracy": round(float(max(history.history.get("val_accuracy", [0]))), 4),
            "per_class_accuracy": per_class_acc,
            "note": "Measured on held-out test set.",
        },
    }
    
    meta_path = MODELS_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved updated metadata -> {meta_path}")
    
    # Save evaluation summary
    eval_summary = {
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
        "per_class_accuracy": per_class_acc,
        "num_test_samples": len(y_test),
        "model_path": str(checkpoint_path),
        "note": "Model trained on balanced dataset (3000 samples per class) with data augmentation.",
    }
    with open(REPORTS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
        
    print("\n" + "=" * 60)
    print("  TRAINING PIPELINE COMPLETE & MODEL UPDATED!")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main(epochs=25, batch_size=64, target_per_class=3000)
