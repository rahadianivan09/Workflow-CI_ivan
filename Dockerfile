FROM python:3.10-slim

WORKDIR /app

# =========================
# SYSTEM DEPENDENCIES (IMPORTANT)
# =========================
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# =========================
# COPY PROJECT
# =========================
COPY MLProject /app/MLProject

# =========================
# INSTALL PYTHON DEPENDENCIES
# =========================
RUN pip install --upgrade pip && \
    pip install -r /app/MLProject/requirements.txt

# =========================
# RUN APP
# =========================
CMD ["python", "/app/MLProject/modelling.py"]