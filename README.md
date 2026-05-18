# Klasifikasi Gambar Rock Paper Scissors menggunakan CNN
### Praktikum Kecerdasan Buatan — Pertemuan 8

---

## Deskripsi

Proyek ini merupakan implementasi **Convolutional Neural Network (CNN)** menggunakan **TensorFlow** dan **Keras** untuk mengklasifikasikan gambar tangan menjadi tiga kategori: **Rock (Batu)**, **Paper (Kertas)**, dan **Scissors (Gunting)**.

CNN dikembangkan untuk mengatasi keterbatasan MLP (Multi-Layer Perceptron) dalam menangani data berdimensi tinggi seperti gambar, dengan menggunakan lapisan konvolusi yang dapat mengekstrak fitur secara efisien.

---

## Tujuan

- Memahami konsep dan cara kerja **Convolutional Neural Network (CNN)**
- Mampu membangun, melatih, dan mengevaluasi model CNN untuk **klasifikasi gambar**

---

## Dataset

| Info | Detail |
|------|--------|
| Nama | Rock Paper Scissors Images |
| Sumber | [Kaggle — drgfreeman](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) |
| Kelas | Rock, Paper, Scissors |
| Input Size | 150 × 150 × 3 (RGB) |
| Split | 80% Training / 20% Validation |

Struktur folder dataset yang diharapkan:
```
rockpaperscissors/
├── rock/
│   ├── img_001.png
│   └── ...
├── paper/
│   ├── img_001.png
│   └── ...
└── scissors/
    ├── img_001.png
    └── ...
```

---

## Arsitektur Model CNN

```
Input (150 × 150 × 3)
         │
         ▼
  Conv2D(32, 3×3, ReLU)    ← Ekstraksi fitur level rendah
  MaxPooling2D(2×2)
         │
         ▼
  Conv2D(64, 3×3, ReLU)    ← Ekstraksi fitur level menengah
  MaxPooling2D(2×2)
         │
         ▼
  Conv2D(128, 3×3, ReLU)   ← Ekstraksi fitur level tinggi
  MaxPooling2D(2×2)
         │
         ▼
      Flatten()
      Dropout(0.5)
         │
         ▼
    Dense(512, ReLU)        ← Fully Connected Layer
         │
         ▼
    Dense(3, Softmax)       ← Output: [rock, paper, scissors]
```

| Layer | Output Shape | Parameter |
|-------|-------------|-----------|
| Conv2D (32 filter) | (148, 148, 32) | 896 |
| MaxPooling2D | (74, 74, 32) | 0 |
| Conv2D (64 filter) | (72, 72, 64) | 18,496 |
| MaxPooling2D | (36, 36, 64) | 0 |
| Conv2D (128 filter) | (34, 34, 128) | 73,856 |
| MaxPooling2D | (17, 17, 128) | 0 |
| Flatten | (36,992) | 0 |
| Dense (512) | (512) | 18,940,416 |
| Dense (3) | (3) | 1,539 |

---

## Konfigurasi Pelatihan

| Parameter | Nilai |
|-----------|-------|
| Loss Function | `categorical_crossentropy` |
| Optimizer | `Adam` |
| Metrics | `accuracy` |
| Epochs | 10 |
| Batch Size | 32 |
| Input Size | 150 × 150 |
| Augmentasi | Rotation, Shift, Flip, Zoom |

---

## Cara Penggunaan

### 1. Clone repositori
```bash
git clone https://github.com/Izazfalih/<NIM>-PraktikumKB-Pertemuan8.git
cd <NIM>-PraktikumKB-Pertemuan8
```

### 2. Install dependensi
```bash
pip install tensorflow numpy pandas matplotlib
```

### 3. Siapkan dataset
Unduh dataset dari Kaggle:
```
https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors
```
Ekstrak ke folder `rockpaperscissors/` dalam direktori proyek.

### 4. Jalankan program
```bash
python cnn_rock_paper_scissors.py
```

### 5. Output yang dihasilkan
- Ringkasan arsitektur model (`model.summary()`)
- Log pelatihan per epoch (loss & accuracy)
- Evaluasi pada data validasi
- Contoh prediksi 10 sampel pertama
- Grafik training: `training_results.png`
- Model tersimpan: `cnn_rockpaperscissors_model.h5`

---

## Langkah-langkah Percobaan

1. **Import library** — TensorFlow, Keras, NumPy, Matplotlib
2. **Persiapkan data** — `ImageDataGenerator` dengan augmentasi & split 80/20
3. **Bangun model CNN** — 3 blok Conv2D + MaxPooling, lalu Flatten + Dense
4. **Kompilasi model** — Optimizer Adam, loss categorical crossentropy
5. **Latih model** — 10 epoch dengan monitoring loss & accuracy
6. **Evaluasi model** — Ukur performa pada data validasi
7. **Prediksi** — Uji model pada gambar-gambar dari data validasi
8. **Visualisasi** — Kurva training accuracy & loss

---

## Konsep Utama CNN

### Convolutional Layer
Lapisan konvolusi mengekstraksi fitur dari gambar input dengan menerapkan filter (kernel) ke gambar. Filter bergerak melintasi setiap piksel dan menghitung **dot product** antara nilai piksel dengan bobot filter.

```python
tf.keras.layers.Conv2D(
    filters=32,         # Jumlah filter/feature map
    kernel_size=(3,3),  # Ukuran kernel
    strides=(1,1),      # Langkah pergeseran
    padding='valid',    # Tanpa padding
    activation='relu'   # Fungsi aktivasi
)
```

### Pooling Layer
Lapisan pooling mengurangi dimensi spasial dari feature map sambil menjaga fitur penting. **Max Pooling** mengambil nilai terbesar dalam setiap window.

```python
tf.keras.layers.MaxPooling2D(
    pool_size=(2,2),
    strides=(2,2),
    padding='valid'
)
```

### Fully Connected Layer
Setelah di-flatten, vektor diproses oleh Dense layer untuk melakukan **klasifikasi akhir** menggunakan Softmax.

```python
tf.keras.layers.Dense(
    units=3,              # 3 kelas output
    activation='softmax'  # Probabilitas multi-kelas
)
```

---

## Teknologi yang Digunakan

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-API-D00000?style=flat&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=python&logoColor=white)

---

## Informasi

- **Mata Kuliah** : Praktikum Kecerdasan Buatan
- **Pertemuan**   : 8 — Jaringan Syaraf Tiruan 3 (CNN)
- **Dataset**     : [Rock Paper Scissors Images — Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
- **Framework**   : TensorFlow 2.x / Keras
