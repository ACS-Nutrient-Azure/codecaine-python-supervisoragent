# 빌드 스테이지
FROM --platform=linux/arm64 python:3.12-slim AS builder

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 실행 스테이지 (컴파일러 없음)
FROM --platform=linux/arm64 python:3.12-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
# COPY app/ ./app/
# COPY lpi_vector_db/ ./lpi_vector_db/

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]