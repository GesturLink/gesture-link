FROM python:3.12.6-slim-bookworm

# Install dependencies sistem yang dibutuhkan untuk OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxrender1 \
    ffmpeg \
    libgtk2.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory di dalam container
WORKDIR /app

# Copy file requirements.txt dan install semua library Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua kode aplikasi ke dalam container
COPY . .

# Perintah untuk menjalankan aplikasi Python
CMD ["python", "app.py"]
