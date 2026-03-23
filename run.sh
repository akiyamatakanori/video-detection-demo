#!/bin/bash
# ============================================================
# Video Intelligence Terminal — 起動スクリプト
# 使い方: bash run.sh
# ============================================================

cd /Users/takaakiy/video-detection-demo

# ── 証明書がなければ自動生成 ──
if [ ! -f "certs/cert.pem" ] || [ ! -f "certs/key.pem" ]; then
    echo "[setup] SSL証明書を生成します..."
    mkdir -p certs
    OLD_CERT="/Users/takaakiy/nvidia_vss/ssl/cert.pem"
    OLD_KEY="/Users/takaakiy/nvidia_vss/ssl/key.pem"
    if [ -f "$OLD_CERT" ] && [ -f "$OLD_KEY" ]; then
        cp "$OLD_CERT" certs/cert.pem
        cp "$OLD_KEY"  certs/key.pem
        echo "[setup] 旧環境の証明書をコピーしました"
    else
        openssl req -x509 -newkey rsa:4096 \
            -keyout certs/key.pem -out certs/cert.pem \
            -days 3650 -nodes -subj "/CN=localhost" 2>/dev/null
        echo "[setup] 新規証明書を生成しました"
    fi
fi

mkdir -p downloads

# ── 社内プロキシ SSL 対策 ──
export HF_HUB_DISABLE_SSL_VERIFICATION=1
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
# TRANSFORMERS_OFFLINE は使用しない → local_files_only=True で制御

echo "[start] https://localhost:8501"
streamlit run app.py \
  --server.sslCertFile certs/cert.pem \
  --server.sslKeyFile  certs/key.pem
