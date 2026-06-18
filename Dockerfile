FROM python:3.11-slim

# Install Vektoria with the server, document ingestion, and the torch-free
# (ONNX) embedding backend — a complete self-host image without PyTorch.
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY vektoria ./vektoria
RUN pip install --no-cache-dir '.[server,ingest,embeddings-onnx]'

# Run as a non-root user; persist data under /data.
RUN useradd --create-home --uid 1000 vektoria \
    && mkdir -p /data && chown vektoria:vektoria /data
ENV VK_DATA_DIR=/data \
    VK_EMBED_BACKEND=fastembed \
    VK_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

USER vektoria
VOLUME /data
EXPOSE 8000

CMD ["vektoria", "serve", "--host", "0.0.0.0", "--port", "8000"]
