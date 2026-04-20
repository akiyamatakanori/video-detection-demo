# Video Intelligence Terminal

RT-DETR物体検出 + ローカルVLM（Ollama）による日本語映像分析アプリ。  
NVIDIA NIM APIとのハイブリッド動作対応。

<!-- スクリーンショットはここに追加 -->
<img width="1326" height="710" alt="image" src="https://github.com/user-attachments/assets/f6ac268d-ea67-45e4-a1be-f098a5381ee1" />

---

- **LIVE DETECTION**: ライブストリーム映像のRT-DETR物体検出 + VLMによる日本語シーン説明
- **VIDEO SEARCH**: 分析ログのキーワード検索（画像付き）
- **SUMMARIZATION**: OllamaローカルモデルまたはNIM APIによる要約作成
- **HIGHLIGHTS**: ハイライトの抽出
- **ANALYSIS LOG**: フレーム分割したログ履歴

---

## アクセス方法

**ブラウザで以下のURLを開くだけで利用できます：**

```
http://10.71.129.9:8503
```

- ターミナル操作不要
- SSHトンネル不要
- 誰のPCからでもアクセス可能
- 両サーバーはサーバー起動時に自動起動
- 複数端末からの同時アクセス可能（同時にSTARTする場合は推論速度が低下します）

---

## 構成

| 機能 | モデル | 動作場所 |
|---|---|---|
| 物体検出（バウンディングボックス） | RT-DETR v1/v2 | GPU Server（CUDA H100 / CPU 切替） |
| 日本語シーン説明 | Qwen2.5-VL 7B 等 | GPU Server（Ollama / localhost） |
| 高精度VLM | Llama 4 / Phi-4 等 | NVIDIA NIM API（クラウド） |
| テキスト要約 | Ollama全モデル / NIM API | GPU Server または クラウド |

### アーキテクチャ

```
ブラウザ（誰のPCからでも）
  ↓ http://10.71.129.9:8503
video-ai-demo (10.71.129.9)
  └── nginx リバースプロキシ（systemd・自動起動）
        ↓ 自動転送
GPU サーバー (192.168.11.111 / ailab5)
  ├── Streamlit アプリ (app.py) ← systemd・自動起動
  ├── RT-DETR v2（デフォルト）/ v1（手動ON）→ GPU ON: CUDA (H100) / GPU OFF: CPU
  └── Ollama VLM → keep_alive=300（5分間モデルをVRAMに保持・高速化）
                   接続先: localhost:11434（同一サーバー・ネットワーク遅延ゼロ）
```

---

## セットアップ手順

### 1. リポジトリをクローン（GPU サーバー上で実行）

```bash
git clone https://github.com/akiyamatakanori/video-detection-demo.git
cd video-detection-demo
```

### 2. 環境変数を設定

```bash
cp .env.example .env
# .env を編集して NVIDIA_API_KEY 等を入力
```

### 3. Pythonパッケージをインストール

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. CUDA対応PyTorchに入れ替え（H100用）

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5. RT-DETRモデルをダウンロード（社外ネットワーク必須・約600MB）

```bash
python3 download_models.py
```

完了後、`~/.cache/huggingface/hub/` に以下が生成されます：
- `models--PekingU--rtdetr_r50vd`
- `models--PekingU--rtdetr_v2_r50vd`

社内ネットワーク環境の場合は上記フォルダを別PCからコピーしてください。

### 6. OllamaでVLMモデルをダウンロード

```bash
ollama pull qwen2.5vl:7b   # 軽量モデル（推奨）
ollama pull qwen2.5vl:32b  # 高精度モデル
```

### 7. Ollama systemd設定（GPU利用・推奨設定）

```bash
sudo systemctl edit ollama --force
```

以下を入力して保存：

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"
Environment="OLLAMA_NEW_ENGINE=true"
Environment="LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib/ollama"
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

> **OLLAMA_NEW_ENGINE=true**: 新エンジンを有効化し、CUDA H100を確実に利用します。  
> **OLLAMA_KEEP_ALIVE はアプリ側で `keep_alive=300`（5分）を指定**しているため systemd への記載は不要です。

### 8. Streamlit systemdサービス登録（GPU サーバーで実行）

```bash
sudo tee /etc/systemd/system/streamlit-vit.service << 'EOF'
[Unit]
Description=Video Intelligence Terminal
After=network.target ollama.service

[Service]
User=ailab
WorkingDirectory=/home/ailab/video-detection-demo
ExecStart=/home/ailab/video-detection-demo/.venv/bin/streamlit run app.py --server.port 8503 --server.address 0.0.0.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable streamlit-vit
sudo systemctl start streamlit-vit
```

### 9. nginxリバースプロキシ設定（video-ai-demoで実行）

```bash
sudo apt install -y nginx

sudo tee /etc/nginx/sites-available/streamlit << 'EOF'
server {
    listen 8503;
    location / {
        proxy_pass http://192.168.11.111:8503;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo systemctl enable nginx
sudo systemctl restart nginx
```

ブラウザで `http://10.71.129.9:8503` を開く。

---

## ファイル構成

```
video-detection-demo/
├── app.py                  # メインアプリ
├── run.sh                  # 起動スクリプト（Mac開発用）
├── download_models.py      # RT-DETRモデルダウンロード
├── requirements.txt        # Pythonパッケージ一覧
├── .env.example            # 環境変数テンプレート（.envをコピーして作成）
├── .gitignore
├── README.md
└── data/                   # ★ 生成ファイル集約フォルダ（自動作成・Git管理外）
    ├── downloads/          # YouTube ダウンロード動画
    ├── logs/               # 分析ログ JSONL（日付別・自動保存）
    ├── summaries/          # 要約テキスト（タイムスタンプ付き・自動保存）
    └── exports/            # Export JSON（タイムスタンプ付き・自動保存）
```

---

## 環境変数（.env）

```env
# ── APIキー ───────────────────────────────────────────────
NVIDIA_API_KEY=nvapi-xxxx...      # NVIDIA NIM APIキー（NIM使用時のみ必要）
HF_TOKEN=hf_xxxx...               # HuggingFaceトークン（任意）

# ── ディレクトリ設定 ──────────────────────────────────────
DATA_DIR=/home/ailab/video-detection-demo/data
DOWNLOAD_DIR=/home/ailab/video-detection-demo/data/downloads
LOG_DIR=/home/ailab/video-detection-demo/data/logs
SUMMARY_DIR=/home/ailab/video-detection-demo/data/summaries
EXPORT_DIR=/home/ailab/video-detection-demo/data/exports

# ── Local File のデフォルト参照フォルダ ─────────────────
DEFAULT_VIDEO_FOLDER=/home/ailab/videos
```

---

## GPU / CPU モード切り替え

ヘッダー右上に **GPU トグル** と **START トグル** の2つがあります。

| 操作 | 動作 |
|---|---|
| GPU トグル **ON** | CUDA H100モード（RT-DETR CUDA + 大型VLMモデル） |
| GPU トグル **OFF** | CPUモード（RT-DETR CPU + 軽量VLMモデル） |
| START トグル **ON** | 映像分析を開始 |
| START トグル **OFF / Stop ボタン** | 分析を停止 |

| モード | RT-DETR | Ollama VLM | 用途 |
|---|---|---|---|
| **⚡ GPU ON** | CUDA (H100) | ビジョンモデル全般 | 高精度・デモ用 |
| **💻 CPU** | CPU | 軽量モデル (7B) | 省リソース・確認用 |

> **設定の保持**: GPUトグルを切り替えてもURL・モデル選択・スライダー設定はリセットされません。

---

## SUMMARIZATION タブ

Ollamaのローカルモデル（GPU上）またはNIM APIを使って分析ログを要約します。

- **Ollamaモデル**: NVIDIA_API_KEY不要。GPU上で高速に処理。全48モデルから選択可能。
- **NIM APIモデル** (`[NIM]` 表記): NVIDIA_API_KEY が必要。クラウド処理。
- LIVE DETECTIONで分析ログが蓄積された後に実行してください。

---

## 対応VLMモデル一覧

### GPU Server（Ollama / localhost:11434）

#### ビジョン対応モデル（LIVE DETECTIONに使用・推奨）

| モデル名（UI表示） | Ollama ID | 特徴 |
|---|---|---|
| Qwen2.5-VL 7B (GPU Server) | `qwen2.5vl:7b` | 軽量・高速・**推奨** |
| Qwen2.5-VL 32B (GPU Server) | `qwen2.5vl:32b` | 高精度 |
| Qwen3-VL 8B (GPU Server) | `qwen3-vl:8b` | 最新世代・軽量 |
| Qwen3-VL 32B (GPU Server) | `qwen3-vl:32b` | 最新世代・高精度 |
| Llama3.2-Vision 11B FP16 (GPU Server) | `llama3.2-vision:11b-instruct-fp16` | FP16高精度 |
| Llama3.2-Vision 11B (GPU Server) | `llama3.2-vision:latest` | 標準版 |
| Llama3.2-Vision 90B (GPU Server) | `llama3.2-vision:90b` | 最大規模 |
| Llama4 Scout 108B (GPU Server) | `llama4:scout` | 最新世代・大規模 |
| Gemma4 31B (GPU Server) | `gemma4:31b` | Google最新世代 |
| Gemma3 27B (GPU Server) | `gemma3:27b` | マルチモーダル |
| Gemma3 12B FP16 (GPU Server) | `gemma3:12b-it-fp16` | FP16・高精度 |
| Gemma3n E4B (GPU Server) | `gemma3n:e4b` | 軽量・効率型 |
| GLM-4.7 Flash BF16 (GPU Server) | `glm-4.7-flash:bf16` | 中国語強化 |
| GLM-4.7 Flash BF16 192K (GPU Server) | `glm-4.7-flash:bf16-192k` | 超長文コンテキスト |

#### テキストモデル（SUMMARIZATION用・LIVE DETECTIONには非推奨）

Qwen3.5・Qwen3・Llama3.x・GPT-OSS・Nemotron・Mistral Small・Phi4・Command-A/R+・Cogito・Aya Expanse など全48モデル対応。

> **注意:** テキスト専用モデルをLIVE DETECTIONで使用するとエラーが発生します。  
> ビジョン対応モデルは名前に `VL`・`Vision`・`Gemma`・`GLM`・`Llama4` が含まれるものです。

### NVIDIA NIM API（クラウド）

| モデル名（UI表示） | 特徴 |
|---|---|
| Llama 3.2 11B Vision (NIM) | バランス型 |
| Llama 3.2 90B Vision (NIM) | 高精度 |
| Llama 4 Maverick 17B (NIM) | 最新世代 |
| Llama 4 Scout 17B (NIM) | 効率型 |
| Nemotron Nano VL 8B (NIM) | 物体検出強化 |
| Phi-4 Multimodal (NIM) | 最新型 |
| Phi-3.5 Vision (NIM) | 小型・高速 |

### HuggingFace（無料枠）

| モデル名（UI表示） |
|---|
| Qwen2-VL 7B (HuggingFace) |
| Llama 3.2 11B Vision (HuggingFace) |
| Pixtral 12B (HuggingFace) |

---

## パフォーマンスチューニング

### 速度に関係するパラメータ（サイドバーで調整）

| パラメータ | デフォルト | 役割 | 速くするには |
|---|---|---|---|
| **VLM Interval (s)** | 5 | 分析間隔 | LATENCYの値に合わせる |
| **Max Tokens** | 300 | 出力文字数上限 | 200に下げる |
| **Image Resize (%)** | 80 | 表示サイズ（VLM送信は内部で50%固定） | 変更不要 |
| **RT-DETR v1** | OFF | v1検出（v2と併用） | OFFのまま推奨 |
| **RT-DETR v2** | ON | メイン検出エンジン | ONのまま |

### 重要な仕組み

```
実際の分析間隔 = max(VLM Interval, 推論時間)
```

- **表示用画像**: Image Resize (%) の値で表示（画質維持）
- **VLM送信画像**: 内部で50%に縮小して高速化（認識精度への影響は軽微）
- **モデルのVRAM保持**: `keep_alive=300`（5分間）により2回目以降の推論が高速化

### 環境別推奨設定

| パラメータ | ⚡ GPU モード | 💻 CPU モード |
|---|---|---|
| VLM Interval (s) | 3〜5 | 10〜15 |
| Max Tokens | 300 | 200 |
| Image Resize (%) | 80 | 60 |

---

## トラブルシューティング

### アプリが開かない（502 Bad Gateway）

GPUサーバーのStreamlitが停止しています：

```bash
# video-ai-demoからSSH
ssh ailab@192.168.11.111
sudo systemctl status streamlit-vit
sudo systemctl restart streamlit-vit
```

### OllamaがGPUを使っていない（推論が遅い）

```bash
journalctl -u ollama -n 20 --no-pager | grep -i "cuda\|H100"
# 正常: ggml_cuda_init: found 2 CUDA devices: NVIDIA H100 PCIe
# 正常: load_backend: loaded CUDA backend from .../cuda_v12/libggml-cuda.so

# CUDAバックエンドが見つからない場合は再インストール
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl restart ollama
```

### VLLMコンテナがGPUを占有する

```bash
nvidia-smi | grep -i vllm
sudo docker stop llm-jp-4-thinking-vllm-server
sudo docker update --restart=no llm-jp-4-thinking-vllm-server
```

### OllamaがVRAMをほぼ全て使っている

前回の推論でモデルがVRAMに残っている場合：

```bash
sudo systemctl restart ollama
sleep 5
nvidia-smi | grep MiB  # 4MiBになればOK
```

### YouTubeのボット判定エラー

```bash
# Mac側でCookieをエクスポート
yt-dlp --cookies-from-browser chrome --cookies cookies.txt "https://youtu.be/XXXX" --skip-download

# video-ai-demo経由でGPUサーバーへ転送
scp cookies.txt ailab@10.71.129.9:/tmp/
ssh ailab@10.71.129.9
scp /tmp/cookies.txt ailab@192.168.11.111:/home/ailab/video-detection-demo/
```

`cookies.txt` が存在する場合、アプリが自動的に認証に使用します。

---

## サービス管理コマンド

### GPUサーバー（ailab5）

```bash
# Streamlit
sudo systemctl status streamlit-vit
sudo systemctl restart streamlit-vit
sudo systemctl stop streamlit-vit

# Ollama
sudo systemctl status ollama
sudo systemctl restart ollama
```

### video-ai-demo（踏み台サーバー）

```bash
# nginx リバースプロキシ
sudo systemctl status nginx
sudo systemctl restart nginx
```

---

## 注意事項

- `.env` はGitに含まれません（APIキーを保護）
- `data/` はGitに含まれません（生成ファイルを保護）
- `cookies.txt` はGitに含まれません（認証情報を保護）
- `certs/` はGitに含まれません（`run.sh` が自動生成）
- Ollamaが起動していない場合はローカルVLMは使えません（NIMモデルは引き続き使用可）
- テキスト専用モデルをLIVE DETECTIONで使用するとOllamaエラーが発生します
- 複数人が同時にSTARTすると推論速度が低下します

---

## 必要なAPIキー

| キー | 取得場所 | 用途 |
|---|---|---|
| `NVIDIA_API_KEY` | https://build.nvidia.com | NIM VLMモデル・テキストモデル（任意） |
| `HF_TOKEN`（任意） | https://huggingface.co/settings/tokens | HF無料モデル |
