FROM python:3.12-slim

WORKDIR /app

# Dépendances système pour OpenCV et YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ultralytics n'est pas dans requirements.txt — installé en premier (torch ~2 GB, cache stable)
RUN pip install --no-cache-dir ultralytics==8.4.62

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "run.py"]
