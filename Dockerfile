FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# PINNED on purpose: bench/baseline.json records this exact stack's scores, so an
# unpinned rebuild quietly moving torch or rfdetr would read as a model regression.
RUN pip install --no-cache-dir \
    rfdetr==1.8.3 \
    torch==2.13.0 \
    torchvision==0.28.0 \
    transformers==5.14.1 \
    numpy==2.5.1 \
    pillow==12.3.0 \
    supervision==0.29.1 \
    fastapi==0.139.2 \
    "uvicorn[standard]==0.51.0" \
    python-multipart==0.0.32
RUN python -c "from rfdetr import RFDETRBase; RFDETRBase()"
COPY server.py index.html bench.html /app/
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
