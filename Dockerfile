# Menggunakan image Python 3.10.6 slim sebagai base
FROM python:3.10.6-slim

# Mengatur variabel lingkungan
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# Mengatur direktori kerja
WORKDIR /app

# Menginstall dependensi sistem
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Menyalin requirements terlebih dahulu
COPY requirements.txt .

# Menginstall dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin file aplikasi
COPY . .

# Membuat direktori yang diperlukan dan memastikan keberadaannya
RUN mkdir -p models templates && \
    touch models/__init__.py

# Memverifikasi keberadaan file model
RUN ls -la models/

# Membuat user non-root
RUN useradd -m myuser && \
    chown -R myuser:myuser /app
USER myuser

# Mengekspos port yang digunakan aplikasi
EXPOSE $PORT

# Perintah untuk menjalankan aplikasi
CMD gunicorn --bind 0.0.0.0:$PORT app:app --timeout 120