# Video Intelligence Terminal

RT-DETR物体検出 + ローカルVLM（Qwen2.5-VL）による日本語映像分析アプリ。  
NVIDIA NIM APIとのハイブリッド動作対応。

<img width="1817" height="782" alt="image" src="https://github.com/user-attachments/assets/7c9c9f4b-33af-4a0a-b883-0768ed90808b" />

---
- LIVE DETECTION: ライブストリーム映像のVLM物体検出と自然言語による説明
- VIDEO SEARCH: LLMによるアノテーション（タグ）の検索
- SUMMARIZATION: LLMによる要約作成
- HIGHLIGHTS: ハイライトの抽出
- ANALYSIS LOG: フレーム分割したログ履歴


## 構成

| 機能 | モデル | 動作場所 |
|---|---|---|
| 物体検出（バウンディングボックス） | RT-DETR v1/v2 | ローカル（MPS/CPU） |
| 日本語シーン説明 | Qwen2.5-VL 7B | ローカル（Ollama） |
| 高精度VLM | Llama 4 / Phi-4 等 | NVIDIA NIM API（クラウド） |

---

## セットアップ手順

### 1. リポジトリをクローン

```bash
git clone https://github.com/YOUR_USERNAME/video-detection-demo.git
cd video-detection-demo
```

### 2. 環境変数を設定

```bash
cp .env.example .env
# .env を編集して NVIDIA_API_KEY 等を入力
```

### 3. Pythonパッケージをインストール

```bash
pip install -r requirements.txt
```

### 4. RT-DETRモデルをダウンロード（社外ネットワーク必須・約300MB）

```bash
python3 download_models.py
```

完了後、`~/.cache/huggingface/hub/` に以下が生成されます：
- `models--PekingU--rtdetr_r50vd`
- `models--PekingU--rtdetr_v2_r50vd`

社内ネットワーク環境の場合は上記フォルダを別PCからコピーしてください。

### 5. Ollamaをインストール・モデルをダウンロード

```bash
# Ollamaインストール（未インストールの場合）
brew install ollama   # macOS

# ビジョンモデルをダウンロード（約6GB）
ollama pull qwen2.5vl:7b

# Ollamaサーバーを起動（初回・再起動後）
ollama serve
```

---

## 起動方法

```bash
bash run.sh
```

ブラウザで `https://localhost:8501` を開く。  
（自己署名証明書のため「詳細設定」→「アクセスする」をクリック）

---

## ファイル構成

```
video-detection-demo/
├── app.py                  # メインアプリ
├── run.sh                  # 起動スクリプト（証明書自動生成含む）
├── download_models.py      # RT-DETRモデルダウンロード
├── requirements.txt        # Pythonパッケージ一覧
├── .env.example            # 環境変数テンプレート（.envをコピーして作成）
├── .gitignore
└── README.md
```

---

## パフォーマンスチューニング

### 速度に関係するパラメータ（サイドバーで調整）

| パラメータ | 場所 | 役割 | 速くするには |
|---|---|---|---|
| **VLM Interval (s)** | Analysis Settings | 次フレームを送るまでの待機時間 | 推論時間に合わせて設定 |
| **Max Tokens** | Analysis Settings | 1回の出力文字数上限 | **200〜300に下げる**（最も効果大） |
| **Image Resize (%)** | Analysis Settings | VLMに送る画像サイズ | **50〜60%に下げる** |
| **Top-k** | Generation Parameters | 生成トークンの絞り込み | 10〜20に下げると若干速くなる |

### 重要な仕組み

実際の分析間隔は以下の式で決まります。

```
実際の分析間隔 = max(VLM Interval, 推論時間)
```

VLM Interval を短くしても、前の推論が終わるまで次のフレームは送られません。  
**ステータスバーの LATENCY 値**が現在の推論時間の目安です。

### 環境別推奨設定

| パラメータ | MacBook M4（デフォルト） | 高性能GPUサーバー |
|---|---|---|
| VLM Interval (s) | 10〜15 | 3〜5 |
| Max Tokens | 200〜300 | 600 |
| Image Resize (%) | 50〜60 | 80〜100 |
| Thinking Mode | OFF | ON可 |

---

## 注意事項

- `.env` はGitに含まれません（APIキーを保護）
- `certs/` はGitに含まれません（`run.sh` が自動生成）
- Ollamaが起動していない場合はローカルVLMは使えません（NIMモデルは引き続き使用可）
- 社内プロキシ環境ではHuggingFaceへのアクセスがブロックされる場合があります

## 必要なAPIキー

| キー | 取得場所 | 用途 |
|---|---|---|
| NVIDIA_API_KEY | https://build.nvidia.com | NIM VLMモデル |
| HF_TOKEN（任意） | https://huggingface.co/settings/tokens | HF無料モデル |
