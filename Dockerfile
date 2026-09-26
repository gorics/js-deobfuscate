FROM python:3.12

ARG FLYBRAIN_COMMIT=4e9f4722c337e7d85cee3072ddb78bfa99234ca8

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/theflyRH/thefly-brain.git flybrain-src \
    && cd flybrain-src \
    && git checkout "$FLYBRAIN_COMMIT"

RUN pip install --no-cache-dir -r /opt/flybrain-src/requirements.txt cython setuptools
RUN bash /opt/flybrain-src/data/download.sh

ENV FLYBRAIN_DATA=/opt/flybrain-src/data \
    PYTHONUNBUFFERED=1

WORKDIR /opt/flybrain-src
EXPOSE 8765

CMD ["sh", "-c", "python -m flybrain.server --host 0.0.0.0 --port ${PORT:-8765}"]
