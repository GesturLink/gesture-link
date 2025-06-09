FROM python:3.12.6-slim-bookworm

# Install dependencies sistem yang dibutuhkan untuk OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
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
