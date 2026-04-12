# Video Intelligence Terminal

RT-DETR物体検出 + ローカルVLM（Ollama）による日本語映像分析アプリ。  
NVIDIA NIM APIとのハイブリッド動作対応。

<!-- スクリーンショットはここに追加 -->
<img width="1326" height="710" alt="image" src="https://github.com/user-attachments/assets/f6ac268d-ea67-45e4-a1be-f098a5381ee1" />

---

- **LIVE DETECTION**: ライブストリーム映像のRT-DETR物体検出 + VLMによる日本語シーン説明
- **VIDEO SEARCH**: LLMによるアノテーション（タグ）の検索
- **SUMMARIZATION**: LLMによる要約作成
- **HIGHLIGHTS**: ハイライトの抽出
- **ANALYSIS LOG**: フレーム分割したログ履歴
- **PERFORMANCE**: GPU / CPU モード別レイテンシ比較
- **LIVE FEED** *(NEW)*: 高解像度YouTube埋め込みプレーヤー（42インチ以上のTV投影用）

---

## 構成

| 機能 | モデル | 動作場所 |
|---|---|---|
| 物体検出（バウンディングボックス） | RT-DETR v1/v2 | GPU Server（CUDA H100 / CPU 切替） |
| 日本語シーン説明 | Qwen2.5-VL 32B 等 | GPU Server（Ollama / localhost） |
| 高精度VLM | Llama 4 / Phi-4 等 | NVIDIA NIM API（クラウド） |
| テキスト要約 | Llama 3.3 70B 等 | NVIDIA NIM API（クラウド） |

### アーキテクチャ

```
ブラウザ
  ↓ http://192.168.11.111:8503
GPU サーバー (192.168.11.111 / ailab5)
  ├── Streamlit アプリ (app.py)  ← GPU サーバー上で直接稼働
  ├── RT-DETR v1/v2  → GPU ON: CUDA (H100) / GPU OFF: CPU
  └── Ollama VLM     → GPU ON: 大型モデル (32B+) / GPU OFF: 軽量モデル (7B)
                        接続先: localhost:11434 (同一サーバー・ネットワーク遅延ゼロ)
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

GPUサーバー上のOllamaに使用するモデルをpullします。

```bash
# 例: 軽量モデル（7B）
ollama pull qwen2.5vl:7b

# 例: 高精度モデル（32B）
ollama pull qwen2.5vl:32b
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
Environment="OLLAMA_KEEP_ALIVE=0"
Environment="OLLAMA_NEW_ENGINE=true"
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

> **OLLAMA_KEEP_ALIVE=0**: モデル推論後にVRAMを即時解放。モデル切替時の競合を防ぎます。  
> **OLLAMA_NEW_ENGINE=true**: 新エンジンを有効化し、CUDA H100を確実に利用します。

### 8. 起動

```bash
source .venv/bin/activate
streamlit run app.py --server.port 8503 --server.address 0.0.0.0
```

ブラウザで `http://192.168.11.111:8503` を開く。

---

## ファイル構成

```
video-detection-demo/
├── app.py                  # メインアプリ
├── run.sh                  # 起動スクリプト
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

> **注意:** `data/` 配下のファイルはすべて自動保存されます。  
> Gitには含まれないため、バックアップは別途実施してください。

---

## 環境変数（.env）

```env
# ── APIキー ───────────────────────────────────────────────
NVIDIA_API_KEY=nvapi-xxxx...      # NVIDIA NIM APIキー
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

> **ポイント:** `DATA_DIR` をNASやマウントされたストレージのパスに変更するだけで、  
> すべての生成ファイルの保存先をまとめて変更できます。

---

## GPU / CPU モード切り替え

ヘッダー右上の **GPU トグルボタン1つ**で切り替えます。

| 操作 | 動作 |
|---|---|
| GPU トグル **ON** | GPUモードで自動START（CUDA H100 + 大型VLMモデル） |
| GPU トグル **OFF** | CPUモードに切替えて自動再起動（軽量VLMモデル） |
| 右上 **Stop** ボタン | 処理を停止 |

| モード | RT-DETR | Ollama VLM | 用途 |
|---|---|---|---|
| **⚡ GPU ON** | CUDA (H100) | 大型モデル (32B+) | 高精度・デモ用 |
| **💻 CPU** | CPU | 軽量モデル (7B) | 省リソース・確認用 |

---

## LIVE FEED タブ（TV投影用）

**PERFORMANCE** タブの右隣に **LIVE FEED** タブを追加しました。

- YouTubeネイティブプレーヤーをフル幅（16:9）で埋め込み
- 最大4K・1080p自動選択
- **F11キー**でフルスクリーン → 42インチ以上のTV・プロジェクター投影対応
- サイドバーでYouTube LiveのURLを接続済みの場合に自動で映像を表示
- 近未来UIテーマに統一した英語表記

---

## 対応VLMモデル一覧

### GPU Server（Ollama / localhost:11434）

#### ビジョン対応モデル（映像分析に使用）

| モデル名（UI表示） | Ollama ID | 特徴 |
|---|---|---|
| Qwen2.5-VL 7B (GPU Server) | `qwen2.5vl:7b` | 軽量・高速 |
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

#### テキストモデル（映像分析には非推奨）

| モデル名 | 特徴 |
|---|---|
| Qwen3.5 4B〜122B | 各サイズ対応 |
| Qwen3 32B / 235B A22B | 超大規模MoE |
| Llama3.1〜3.3 各種 | テキスト推論 |
| GPT-OSS 20B〜120B | GPT互換 |
| Nemotron 各種 | NVIDIA推論特化 |
| Mistral Small 3.1/3.2 | 軽量・高速 |
| Phi4 / Phi4 Mini | Microsoft最新 |
| Command-A / Command-R+ | Cohere製 |
| Cogito 32B | 推論特化 |
| Aya Expanse 32B | 多言語特化 |

> **注意:** テキスト専用モデルを映像分析に使用するとエラーが発生します。  
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

| パラメータ | 役割 | 速くするには |
|---|---|---|
| **VLM Interval (s)** | 次フレームを送るまでの待機時間 | 推論時間に合わせて設定 |
| **Max Tokens** | 1回の出力文字数上限 | **200〜300に下げる**（最も効果大） |
| **Image Resize (%)** | VLMに送る画像サイズ | **50〜60%に下げる** |

### 重要な仕組み

```
実際の分析間隔 = max(VLM Interval, 推論時間)
```

**ステータスバーの LATENCY 値**が現在の推論時間の目安です。

### 環境別推奨設定

| パラメータ | ⚡ GPU モード | 💻 CPU モード |
|---|---|---|
| VLM Interval (s) | 3〜5 | 10〜15 |
| Max Tokens | 600 | 200〜300 |
| Image Resize (%) | 80〜100 | 50〜60 |

---

## トラブルシューティング

### OllamaがGPUを使っていない

```bash
# GPU認識確認
journalctl -u ollama -n 20 --no-pager | grep -i "cuda\|H100\|inference"

# 正常時の出力例:
# ggml_cuda_init: found 2 CUDA devices: NVIDIA H100 PCIe
# load_backend: loaded CUDA backend from /usr/local/lib/ollama/cuda_v12/libggml-cuda.so
```

CUDAバックエンドが見つからない場合はOllamaを再インストール：

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl restart ollama
```

### VLLMコンテナがGPUを占有する

別のVLLMコンテナが起動してOllamaのVRAMを奪うことがあります：

```bash
nvidia-smi | grep -i vllm   # 確認
sudo docker stop llm-jp-4-thinking-vllm-server
sudo docker update --restart=no llm-jp-4-thinking-vllm-server
```

### YouTubeのボット判定エラー

```
ERROR: Sign in to confirm you're not a bot.
```

```bash
# Mac側でCookieをエクスポート
yt-dlp --cookies-from-browser chrome --cookies cookies.txt "https://youtu.be/XXXX" --skip-download

# GPUサーバーへ転送
scp cookies.txt ailab@192.168.11.111:/home/ailab/video-detection-demo/
```

`cookies.txt` が存在する場合、アプリが自動的に認証に使用します。

---

## 注意事項

- `.env` はGitに含まれません（APIキーを保護）
- `data/` はGitに含まれません（生成ファイルを保護）
- `cookies.txt` はGitに含まれません（認証情報を保護）
- `certs/` はGitに含まれません（`run.sh` が自動生成）
- Ollamaが起動していない場合はローカルVLMは使えません（NIMモデルは引き続き使用可）
- 社内プロキシ環境ではHuggingFaceへのアクセスがブロックされる場合があります
- テキスト専用モデルを映像分析モードで使用するとOllamaエラーが発生します

---

## 必要なAPIキー

| キー | 取得場所 | 用途 |
|---|---|---|
| `NVIDIA_API_KEY` | https://build.nvidia.com | NIM VLMモデル・テキストモデル |
| `HF_TOKEN`（任意） | https://huggingface.co/settings/tokens | HF無料モデル |
