FROM python:3.12-slim

# glibc 默认会按 CPU 核数预留多个 64 MiB 的 malloc arena（虚拟地址空间），
# 在 384 MiB 地址空间限额下会挤占合成管线的预算；限制为 2 个。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MOSAIC_DATA_DIR=/data \
    MALLOC_ARENA_MAX=2

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN useradd --create-home --uid 10001 mosaic \
    && mkdir -p /data \
    && chown -R mosaic:mosaic /data /app
USER mosaic

EXPOSE 8000
VOLUME ["/data"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
