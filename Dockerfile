FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir rfdetr pillow supervision fastapi "uvicorn[standard]" python-multipart
RUN python -c "from rfdetr import RFDETRBase; RFDETRBase()"
COPY server.py index.html /app/
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
