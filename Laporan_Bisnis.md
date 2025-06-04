# Pendahuluan

Di era digital yang semakin inklusif, pemanfaatan teknologi untuk mendukung komunikasi bagi penyandang disabilitas menjadi semakin penting. Salah satu tantangan besar yang masih dihadapi adalah keterbatasan media pembelajaran bahasa isyarat yang mudah diakses, terutama bagi penyandang tuna rungu dan tuna wicara. Untuk menjawab tantangan ini, proyek **Gestur Link** dikembangkan sebagai solusi inovatif berbasis teknologi kecerdasan buatan yang mampu mengenali gestur tangan secara real-time sebagai bentuk pembelajaran bahasa isyarat yang adaptif dan inklusif.

Agar pengembangan solusi ini dapat dilakukan secara sistematis dan berbasis data, pendekatan **CRISP-DM (Cross-Industry Standard Process for Data Mining)** digunakan. Metodologi ini memberikan kerangka kerja terstruktur yang mencakup enam tahapan utama: *Business Understanding*, *Data Understanding*, *Data Preparation*, *Modeling*, *Evaluation*, dan *Deployment*. Dengan mengikuti tahapan-tahapan ini, proyek Gestur Link tidak hanya berfokus pada pengembangan teknologi, tetapi juga memastikan bahwa solusi yang dihasilkan benar-benar relevan dengan kebutuhan pengguna serta memiliki performa yang andal dan dapat diimplementasikan dalam dunia nyata.

## 1. Business Understanding

Proyek **Gestur Link** dibuat untuk membantu mengatasi kesulitan komunikasi yang dialami oleh penyandang tuna rungu dan tuna wicara. Salah satu penyebab utamanya adalah masih terbatasnya media pembelajaran bahasa isyarat yang interaktif, mudah digunakan, dan dapat diakses oleh semua kalangan. Kondisi ini tidak hanya menyulitkan penyandang disabilitas, tetapi juga keluarga dan tenaga pendidik yang ingin belajar berkomunikasi secara efektif dengan mereka.

Oleh karena itu, proyek ini bertujuan mengembangkan platform berbasis web yang mampu mengenali gestur tangan secara real-time sebagai bentuk pembelajaran bahasa isyarat yang inklusif dan adaptif. Inisiatif ini juga sejalan dengan semangat pemanfaatan teknologi digital untuk memperluas akses informasi dan pembelajaran bagi penyandang disabilitas serta seluruh pihak yang berperan dalam mendukung proses berkomunikasi (Sari et al., 2023).

Keberhasilan Gestur Link diharapkan dapat memberikan kontribusi nyata dalam bidang sosial dan pendidikan dengan memperluas akses terhadap pembelajaran bahasa isyarat bagi masyarakat secara luas. Solusi ini tidak hanya mempermudah proses komunikasi dua arah antara penyandang disabilitas dan masyarakat umum, tetapi juga berpotensi menjadi inovasi teknologi yang ramah disabilitas dan dapat diadopsi oleh lembaga pendidikan maupun organisasi sosial.

Penerapan teknologi **MediaPipe** dan model klasifikasi gestur berbasis kecerdasan buatan menjadi fondasi utama dalam mengotomatisasi penerjemahan gestur tangan secara real-time. Inovasi ini mendukung terciptanya lingkungan yang lebih inklusif dan meningkatkan kesadaran masyarakat akan pentingnya komunikasi yang setara bagi semua individu (Wulandari & Hidayat, 2022).

**Referensi:**
- Sari, R. P., Nugroho, A., & Wibowo, S. A. (2023). *Pengembangan Media Pembelajaran Interaktif untuk Penyandang Disabilitas dengan Pendekatan Inklusif Berbasis Teknologi Digital*. Jurnal Teknologi Informasi dan Pendidikan, 16(2), 112–120.
- Wulandari, D., & Hidayat, R. (2022). *Penerapan MediaPipe dan Machine Learning untuk Pengenalan Bahasa Isyarat Indonesia*. Jurnal Teknologi dan Sistem Komputer, 10(3), 435–442.

---

## 2. Data Understanding

Dalam proyek ini, data dikumpulkan secara mandiri dengan memanfaatkan kamera laptop yang dijalankan secara lokal menggunakan program Python. Sistem dibuat untuk mendeteksi keberadaan tangan secara real-time menggunakan teknologi deteksi **landmark**, di mana setiap tangan memiliki 21 titik landmark. Jika dua tangan terdeteksi, maka sistem akan menghasilkan 42 titik, dengan masing-masing titik memiliki koordinat x dan y, sehingga total terdapat **84 nilai numerik** untuk satu frame.

Sebelum proses pengambilan gambar dimulai, pengguna menentukan label gestur terlebih dahulu, serta jumlah gambar yang ingin dikumpulkan (misalnya 50 gambar per label). Program akan menangkap gambar secara otomatis ketika tangan terdeteksi dan landmark muncul dengan jelas di kamera.

Data yang dihasilkan tersimpan secara terstruktur:
- Gambar disimpan ke dalam folder `Dataset/DataGambar/` sesuai dengan nama labelnya.
- Data koordinat landmark disimpan dalam format `.csv` ke dalam folder `Dataset/Landmark/`, juga berdasarkan label. Contoh: `Dataset/Landmark/A.csv` untuk label **A**.

Empat orang terlibat dalam proses pengumpulan data, di mana masing-masing bertanggung jawab mengumpulkan 50 data per label. Perangkat yang digunakan pun bervariasi, karena setiap orang menggunakan kamera laptop masing-masing. Hal ini menjadi nilai tambah karena membantu menciptakan variasi data dari segi resolusi, pencahayaan, serta sudut pandang kamera. Variasi ini penting untuk meningkatkan kemampuan **generalisasi model** terhadap berbagai kondisi nyata di lapangan.

Secara keseluruhan, dataset yang dihasilkan bersifat spesifik, terorganisir, dan disesuaikan dengan kebutuhan pelatihan model klasifikasi gestur tangan berbasis bahasa isyarat **Bisindo**. Variasi perangkat dan kondisi pengambilan gambar menjadi bagian penting dalam memperkaya representasi data, sekaligus menjadi tantangan yang harus dipertimbangkan pada tahap *preprocessing* dan pelatihan model.
