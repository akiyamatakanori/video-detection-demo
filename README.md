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

---

## 構成

| 機能 | モデル | 動作場所 |
|---|---|---|
| 物体検出（バウンディングボックス） | RT-DETR v1/v2 | GPU Server（CUDA / CPU 切替） |
| 日本語シーン説明 | Qwen2.5-VL 32B 等 | GPU Server（Ollama / localhost） |
| 高精度VLM | Llama 4 / Phi-4 等 | NVIDIA NIM API（クラウド） |
| テキスト要約 | Llama 3.3 70B 等 | NVIDIA NIM API（クラウド） |

### アーキテクチャ

```
ブラウザ
  ↓ http://192.168.11.111:8503
GPU サーバー (192.168.11.111)
  ├── Streamlit アプリ (app.py)
  ├── RT-DETR v1/v2  → GPU ON: CUDA (H100) / GPU OFF: CPU
  └── Ollama VLM     → GPU ON: 大型モデル (32B+) / GPU OFF: 軽量モデル (7B)
                        接続先: localhost:11434 (同一サーバー)
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

# Ollamaサーバーを起動（未起動の場合）
ollama serve
```

### 7. 起動

```bash
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

`.env` ファイルで各種パスや動作を変更できます。

```env
# ── APIキー ───────────────────────────────────────────────
NVIDIA_API_KEY=nvapi-xxxx...      # NVIDIA NIM APIキー
HF_TOKEN=hf_xxxx...               # HuggingFaceトークン（任意）

# ── ディレクトリ設定 ──────────────────────────────────────
# DATA_DIR を変更すると data/ 配下の全フォルダがまとめて移動します
DATA_DIR=/home/ailab/video-detection-demo/data

# 個別に変更したい場合は以下で上書き可能
DOWNLOAD_DIR=/home/ailab/video-detection-demo/data/downloads
LOG_DIR=/home/ailab/video-detection-demo/data/logs
SUMMARY_DIR=/home/ailab/video-detection-demo/data/summaries
EXPORT_DIR=/home/ailab/video-detection-demo/data/exports

# ── Local File のデフォルト参照フォルダ ─────────────────
# サイドバーの Input Source > Local Folder で最初に表示されるフォルダ
# サイドバーから任意のパスに切り替えることも可能
DEFAULT_VIDEO_FOLDER=/home/ailab/videos
```

> **ポイント:** `DATA_DIR` をNASやマウントされたストレージのパスに変更するだけで、  
> すべての生成ファイルの保存先をまとめて変更できます。

---

## GPU / CPU モード切り替え

ヘッダー右上の **⚡ GPU / 💻 CPU トグルボタン** で切り替えます。

| モード | RT-DETR | Ollama VLM | 用途 |
|---|---|---|---|
| **⚡ GPU ON** | CUDA (H100) | 大型モデル (32B+) | 高精度・デモ用 |
| **💻 CPU** | CPU | 軽量モデル (7B) | 省リソース・確認用 |

---

## 対応VLMモデル一覧

### GPU Server（Ollama / localhost:11434）

| モデル名（UI表示） | Ollama ID | 特徴 |
|---|---|---|
| Qwen2.5-VL 7B (GPU Server) | `qwen2.5vl:7b` | 軽量・高速 |
| Qwen2.5-VL 32B (GPU Server) | `qwen2.5vl:32b` | 高精度 |
| Qwen3-VL 8B (GPU Server) | `qwen3-vl:8b` | 最新世代・軽量 |
| Qwen3-VL 32B (GPU Server) | `qwen3-vl:32b` | 最新世代・高精度 |
| Llama3.2-Vision 11B FP16 (GPU Server) | `llama3.2-vision:11b-instruct-fp16` | FP16高精度 |
| Llama3.2-Vision 11B (GPU Server) | `x/llama3.2-vision:latest` | 標準版 |
| Llama3.2-Vision 90B (GPU Server) | `llama3.2-vision:90b` | 最大規模 |
| Llama4 Scout 108B (GPU Server) | `llama4:scout` | 最新世代・大規模 |
| Gemma4 31B (GPU Server) | `gemma4:31b` | Google最新世代 |
| Gemma3 27B (GPU Server) | `gemma3:27b` | マルチモーダル |
| Gemma3 12B FP16 (GPU Server) | `gemma3:12b-it-fp16` | FP16・高精度 |
| Gemma3n E4B (GPU Server) | `gemma3n:e4b` | 軽量・効率型 |
| GLM-4.7 Flash BF16 (GPU Server) | `glm-4.7-flash:bf16` | 中国語強化 |
| GLM-4.7 Flash BF16 192K (GPU Server) | `glm-4.7-flash:bf16-192k` | 超長文コンテキスト |

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

実際の分析間隔は以下の式で決まります。

```
実際の分析間隔 = max(VLM Interval, 推論時間)
```

VLM Interval を短くしても、前の推論が終わるまで次のフレームは送られません。  
**ステータスバーの LATENCY 値**が現在の推論時間の目安です。

### 環境別推奨設定

| パラメータ | ⚡ GPU モード | 💻 CPU モード |
|---|---|---|
| VLM Interval (s) | 3〜5 | 10〜15 |
| Max Tokens | 600 | 200〜300 |
| Image Resize (%) | 80〜100 | 50〜60 |

---

## 注意事項

- `.env` はGitに含まれません（APIキーを保護）
- `data/` はGitに含まれません（生成ファイルを保護）
- `certs/` はGitに含まれません（`run.sh` が自動生成）
- Ollamaが起動していない場合はローカルVLMは使えません（NIMモデルは引き続き使用可）
- 社内プロキシ環境ではHuggingFaceへのアクセスがブロックされる場合があります

---

## 必要なAPIキー

| キー | 取得場所 | 用途 |
|---|---|---|
| `NVIDIA_API_KEY` | https://build.nvidia.com | NIM VLMモデル・テキストモデル |
| `HF_TOKEN`（任意） | https://huggingface.co/settings/tokens | HF無料モデル |
