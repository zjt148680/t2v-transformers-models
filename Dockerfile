FROM python:3.11-slim AS base_image

WORKDIR /app

RUN apt-get update
RUN pip install --upgrade pip setuptools

COPY requirements.txt .
RUN pip3 install -r requirements.txt

FROM base_image AS download_model

WORKDIR /app

ARG TARGETARCH
ARG MODEL_NAME
ARG ONNX_RUNTIME
ENV ONNX_CPU=${TARGETARCH}
ARG TRUST_REMOTE_CODE
ARG USE_SENTENCE_TRANSFORMERS_VECTORIZER
RUN mkdir nltk_data
COPY download.py .
RUN python3 ./download.py

FROM base_image AS t2v_transformers

WORKDIR /app
COPY --from=download_model /app/models /app/models
COPY --from=download_model /app/nltk_data /app/nltk_data
COPY . .

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["uvicorn app:app --host 0.0.0.0 --port 8080"]

# docker build --build-arg "MODEL_NAME=sentence-transformers/multi-qa-MiniLM-L6-cos-v1" --build-arg "ONNX_RUNTIME=false" --build-arg "TRUST_REMOTE_CODE=false" --build-arg "USE_SENTENCE_TRANSFORMERS_VECTORIZER=false" --build-arg "HTTP_PROXY=http://host.docker.internal:10809" --build-arg "HTTPS_PROXY=http://host.docker.internal:10809" --build-arg "NO_PROXY=localhost,127.0.0.1,host.docker.internal" -t "custom-t2v-transformers:sentence-transformers-multi-qa-MiniLM-L6-cos-v1" .
