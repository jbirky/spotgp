FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/spotgp

RUN pip install --no-cache-dir \
        /opt/spotgp[jax,hdf5] \
        blackjax \
        dynesty \
        lightkurve \
        streamlit \
        dvc \
        mlflow \
    && rm -rf /opt/spotgp

WORKDIR /work
ENTRYPOINT ["python"]
