#!/bin/sh
set -eu

echo RES_BOOTSTRAP_START
apk add --no-cache python3 py3-pip py3-numpy py3-pyarrow py3-scipy py3-sympy py3-jinja2 py3-parsing py3-packaging py3-pandas curl ca-certificates >/dev/null
python3 -m pip install --break-system-packages --no-cache-dir --no-deps brian2==2.10.0 >/dev/null

BASE=https://raw.githubusercontent.com/gorics/js-deobfuscate/flybrain-chat
mkdir -p /srv/flybrain/data
curl -fsSL "$BASE/flybrain_reservoir_chat.py" -o /srv/flybrain/flybrain_reservoir_chat.py
curl -fsSL "$BASE/flybrain_proxy.py" -o /srv/flybrain/flybrain_proxy.py
curl -fsSL "$BASE/flybrain_poller.py" -o /srv/flybrain/flybrain_poller.py
curl -fsSL https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Completeness_783.csv -o /srv/flybrain/data/Completeness_783.csv
curl -fsSL https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet -o /srv/flybrain/data/Connectivity_783.parquet

echo RES_INPUTS_READY
PYTHONUNBUFFERED=1 python3 /srv/flybrain/flybrain_proxy.py &
PROXY_PID=$!
PYTHONUNBUFFERED=1 python3 /srv/flybrain/flybrain_poller.py &
POLLER_PID=$!
trap 'kill $PROXY_PID $POLLER_PID 2>/dev/null || true' EXIT INT TERM

export FLYBRAIN_DATA=/srv/flybrain/data
export PYTHONUNBUFFERED=1
export PORT=3001
exec python3 /srv/flybrain/flybrain_reservoir_chat.py
