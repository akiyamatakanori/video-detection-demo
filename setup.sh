#!/bin/bash
# ============================================================
# Video Intelligence Terminal — セットアップスクリプト
# 使い方: bash setup.sh
# ============================================================

set -e
BASE="/Users/takaakiy/video-detection-demo"
cd "$BASE"

echo "==> [1/4] フォルダ作成..."
mkdir -p certs downloads

echo "==> [2/4] SSL証明書を旧環境からコピー..."
OLD_CERT="/Users/takaakiy/nvidia_vss/ssl/cert.pem"
OLD_KEY="/Users/takaakiy/nvidia_vss/ssl/key.pem"

if [ -f "$OLD_CERT" ] && [ -f "$OLD_KEY" ]; then
    cp "$OLD_CERT" certs/cert.pem
    cp "$OLD_KEY"  certs/key.pem
    echo "    既存証明書をコピーしました"
else
    echo "    旧証明書が見つからないため新規生成します..."
    openssl req -x509 -newkey rsa:4096 \
        -keyout certs/key.pem -out certs/cert.pem \
        -days 3650 -nodes -subj "/CN=localhost"
    echo "    新規証明書を生成しました"
fi

echo "==> [3/4] 仮想環境セットアップ..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "    .venv を作成しました"
else
    echo "    既存の .venv を使用します"
fi

source .venv/bin/activate

echo "==> [4/4] パッケージインストール..."
pip install --quiet --upgrade pip
pip install --quiet \
    streamlit \
    opencv-python-headless \
    Pillow \
    numpy \
    requests \
    python-dotenv \
    yt-dlp \
    torch torchvision \
    transformers \
    accelerate \
    huggingface-hub

echo ""
echo "============================================"
echo " セットアップ完了！以下のコマンドで起動:"
echo ""
echo "   cd $BASE"
echo "   source .venv/bin/activate"
echo "   streamlit run streamlit_app.py \\"
echo "     --server.sslCertFile certs/cert.pem \\"
echo "     --server.sslKeyFile  certs/key.pem"
echo ""
echo " ブラウザ: https://localhost:8501"
echo "============================================"
