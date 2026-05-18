"""
============================================================
  Praktikum Kecerdasan Buatan — Pertemuan 8
  Jaringan Syaraf Tiruan 3: Convolutional Neural Network (CNN)
  Klasifikasi Gambar: Rock Paper Scissors
============================================================

Deskripsi:
    Program ini mengimplementasikan Convolutional Neural Network (CNN)
    menggunakan TensorFlow/Keras untuk mengklasifikasikan gambar
    batu (rock), kertas (paper), dan gunting (scissors).

Dataset:
    Rock Paper Scissors Images
    https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors

Struktur Dataset yang Diharapkan:
    rockpaperscissors/
    ├── rock/
    ├── paper/
    └── scissors/
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# ──────────────────────────────────────────────
# 0. Konfigurasi
# ──────────────────────────────────────────────
DATASET_PATH = "./rockpaperscissors"
TARGET_SIZE  = (150, 150)
BATCH_SIZE   = 32
EPOCHS       = 10
SEED         = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("  CNN — Klasifikasi Rock Paper Scissors")
print("  Praktikum Kecerdasan Buatan — Pertemuan 8")
print("=" * 60)

# ──────────────────────────────────────────────
# 1. Cek ketersediaan dataset
# ──────────────────────────────────────────────
if not os.path.exists(DATASET_PATH):
    print(f"\n[ERROR] Folder dataset '{DATASET_PATH}' tidak ditemukan.")
    print("Unduh dataset dari:")
    print("  https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors")
    print("Lalu ekstrak ke folder 'rockpaperscissors/' di direktori ini.\n")
    exit(1)

print(f"\n[INFO] Dataset ditemukan: {DATASET_PATH}")
for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if os.path.isdir(label_path):
        count = len(os.listdir(label_path))
        print(f"  • {label}: {count} gambar")

# ──────────────────────────────────────────────
# 2. Persiapan Data dengan ImageDataGenerator
# ──────────────────────────────────────────────
print("\n[STEP 1] Mempersiapkan data dengan ImageDataGenerator...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,         # Augmentasi: rotasi
    width_shift_range=0.1,     # Augmentasi: geser horizontal
    height_shift_range=0.1,    # Augmentasi: geser vertikal
    horizontal_flip=True,      # Augmentasi: balik horizontal
    zoom_range=0.1             # Augmentasi: zoom
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=SEED
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=SEED
)

print(f"\n  Kelas yang terdeteksi : {train_generator.class_indices}")
print(f"  Jumlah data latih     : {train_generator.samples}")
print(f"  Jumlah data validasi  : {validation_generator.samples}")

# ──────────────────────────────────────────────
# 3. Membangun Arsitektur Model CNN
# ──────────────────────────────────────────────
print("\n[STEP 2] Membangun arsitektur CNN...")

model = Sequential([
    # Blok Konvolusi 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3),
           name='conv2d_1'),
    MaxPooling2D(2, 2, name='maxpool_1'),

    # Blok Konvolusi 2
    Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
    MaxPooling2D(2, 2, name='maxpool_2'),

    # Blok Konvolusi 3
    Conv2D(128, (3, 3), activation='relu', name='conv2d_3'),
    MaxPooling2D(2, 2, name='maxpool_3'),

    # Flatten & Fully Connected
    Flatten(name='flatten'),
    Dropout(0.5, name='dropout'),
    Dense(512, activation='relu', name='dense_1'),
    Dense(3, activation='softmax', name='output')  # 3 kelas
], name='CNN_RockPaperScissors')

model.summary()

# ──────────────────────────────────────────────
# 4. Kompilasi Model
# ──────────────────────────────────────────────
print("\n[STEP 3] Mengompilasi model...")

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("  Loss function : categorical_crossentropy")
print("  Optimizer     : Adam")
print("  Metrics       : accuracy")

# ──────────────────────────────────────────────
# 5. Pelatihan Model (Model Fitting)
# ──────────────────────────────────────────────
print(f"\n[STEP 4] Melatih model selama {EPOCHS} epoch...\n")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# ──────────────────────────────────────────────
# 6. Evaluasi Model
# ──────────────────────────────────────────────
print("\n[STEP 5] Mengevaluasi model pada data validasi...")

val_loss, val_acc = model.evaluate(validation_generator, verbose=0)
print(f"\n  Validation Loss     : {val_loss:.4f}")
print(f"  Validation Accuracy : {val_acc:.4f} ({val_acc*100:.2f}%)")

# ──────────────────────────────────────────────
# 7. Prediksi
# ──────────────────────────────────────────────
print("\n[STEP 6] Melakukan prediksi pada data validasi...")

predictions = model.predict(validation_generator, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
true_classes      = validation_generator.classes

class_names = list(validation_generator.class_indices.keys())
print(f"\n  Contoh prediksi (10 sampel pertama):")
print(f"  {'Index':<8} {'True':<12} {'Predicted':<12} {'Confidence':>10}")
print(f"  {'-'*46}")
for i in range(min(10, len(predicted_classes))):
    true_label = class_names[true_classes[i]]
    pred_label = class_names[predicted_classes[i]]
    confidence = predictions[i][predicted_classes[i]]
    status = "✓" if true_label == pred_label else "✗"
    print(f"  {i:<8} {true_label:<12} {pred_label:<12} {confidence:>9.2%}  {status}")

# ──────────────────────────────────────────────
# 8. Visualisasi Kurva Training
# ──────────────────────────────────────────────
print("\n[STEP 7] Menyimpan visualisasi hasil training...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CNN Rock Paper Scissors — Training Results', fontsize=14, fontweight='bold')

# Kurva Accuracy
axes[0].plot(history.history['accuracy'],     label='Train Accuracy', color='royalblue',  linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='tomato',     linewidth=2, linestyle='--')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Kurva Loss
axes[1].plot(history.history['loss'],     label='Train Loss', color='royalblue',  linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss',   color='tomato',     linewidth=2, linestyle='--')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
print("  Grafik disimpan sebagai: training_results.png")

# ──────────────────────────────────────────────
# 9. Simpan Model
# ──────────────────────────────────────────────
model.save('cnn_rockpaperscissors_model.h5')
print("\n[INFO] Model disimpan sebagai: cnn_rockpaperscissors_model.h5")

print("\n" + "=" * 60)
print("  Selesai!")
print(f"  Akurasi validasi akhir: {val_acc*100:.2f}%")
print("=" * 60)
