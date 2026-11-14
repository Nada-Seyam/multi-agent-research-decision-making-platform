FROM node:20-alpine AS frontend-build

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY services/analytics/requirements.txt ./services/analytics/

RUN pip install --no-cache-dir -r requirements.txt -r services/analytics/requirements.txt

ARG EMBEDDING_MODEL_ID=BAAI/bge-m3
ARG EMBEDDING_MODEL_REVISION=142964af7e05de16511657561de8e8750fc153a0
ARG RERANKER_MODEL_ID=cross-encoder/ms-marco-MiniLM-L-6-v2
ARG RERANKER_MODEL_REVISION=4bebbd56fc380a66525f95b03d4ec1a4b41a4f1e

COPY scripts/cache_models.py ./scripts/cache_models.py

RUN python scripts/cache_models.py \
    --embedding-model "$EMBEDDING_MODEL_ID" \
    --embedding-revision "$EMBEDDING_MODEL_REVISION" \
    --embedding-destination /opt/models/bge-m3 \
    --reranker-model "$RERANKER_MODEL_ID" \
    --reranker-revision "$RERANKER_MODEL_REVISION" \
    --reranker-destination /opt/models/ms-marco-MiniLM-L-6-v2 \
    --cache-dir /tmp/huggingface-build-cache

ARG NLI_MODEL_ID=cross-encoder/nli-deberta-v3-base
ARG NLI_MODEL_REVISION=0233b6e1207aa367aa01ac84abd8efecc842f2d1

COPY scripts/cache_nli_model.py ./scripts/cache_nli_model.py

RUN python scripts/cache_nli_model.py \
    --model "$NLI_MODEL_ID" \
    --revision "$NLI_MODEL_REVISION" \
    --destination /opt/models/nli-deberta-v3-base \
    --cache-dir /tmp/huggingface-nli-build-cache

COPY main.py conftest.py ./
COPY services/discussion-engine/ services/discussion-engine/
COPY services/analytics/ services/analytics/
COPY --from=frontend-build /frontend/dist /app/static

ENV DISCUSSION_API_HOST=0.0.0.0 \
    ANALYTICS_API_HOST=0.0.0.0 \
    DISCUSSION_STATIC_DIR=/app/static \
    MODEL_NAME=/opt/models/bge-m3 \
    RETRIEVAL_RERANKER_MODEL=/opt/models/ms-marco-MiniLM-L-6-v2 \
    ANALYTICS_NLI_MODEL=/opt/models/nli-deberta-v3-base \
    HF_HOME=/opt/huggingface \
    HF_HUB_CACHE=/opt/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/opt/huggingface/sentence-transformers \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000 8001 8002

CMD ["python", "main.py"]
