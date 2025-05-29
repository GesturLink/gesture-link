# ✋ BISINDO Sign Language Recognition

Proyek ini digunakan untuk merekam dan melatih model klasifikasi gesture Bahasa Isyarat Indonesia (BISINDO) menggunakan landmark tangan dari MediaPipe. Hasil klasifikasi digunakan untuk mendeteksi huruf dan angka A–Z, 0–9.

---

## 🧹 Requirements

Gunakan **Python 3.10.x** untuk kompatibilitas terbaik di Windows dan macOS.

Install dependensi:

```bash
python -m venv venv
source venv/bin/activate  # atau .\venv\Scripts\activate di Windows
pip install -r requirements.txt
```

---

## 📦 Struktur Folder Dataset

```
dataset/
├── dataset_images/
│   ├── A/
│   ├── B/
│   └── ...
├── dataset_landmark_csv/
│   ├── A.csv
│   ├── B.csv
│   └── ...
```

---

## 🎥 Menambahkan Data Gesture Baru

Untuk berkontribusi menambahkan data gesture:

1. Jalankan script berikut:

```bash
python collect_landmark_data.py
```

2. Masukkan label huruf (contoh: `A`, `B`, `3`, dst)
3. Masukkan jumlah data yang ingin direkam (misal: `30`)
4. Sistem akan:

   * Menunggu tangan terdeteksi
   * Menampilkan countdown 10 detik
   * Mulai merekam dan menyimpan ke `dataset/`

> 🌟 File landmark akan ditambahkan ke `.csv` tanpa menghapus data lama
> 🖼️ Gambar disimpan otomatis ke folder `dataset_images/{label}/`

---

## 🛠 Kolaborasi GitHub

Jika kamu pertama kali clone:

```bash
git clone https://github.com/GesturLink/gesture-link.git
cd gesture-link
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🧠 Kontributor

* @LAI25-SS040
