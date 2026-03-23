"""
Video Intelligence Terminal
============================
リアルタイム物体検知 (RT-DETR / RT-DETRv2) と AI 日本語状況説明を並列処理するアプリ

主な機能:
- RT-DETR v1 / RT-DETRv2 を個別ON/OFFで選択してローカル物体検知（バウンディングボックス表示）
- NVIDIA NIM Vision API / HuggingFace 無料モデルによる日本語状況説明（並列処理）
- YouTube 動画 / ライブ配信 / ローカルフォルダ のファイル解析対応
- 社内プロキシ環境対応: SSL 検証を全 import より前に無効化（transformers/huggingface_hub 対応）
- 受付大画面向け 24/365 常時表示対応
"""

# ─────────────────────────────────────────────
# 【最優先】社内プロキシ SSL 対策
# 環境変数だけでは効かない場合に備え requests.Session 自体をモンキーパッチして
# 全 HTTP 通信を強制 verify=False にする。
# ─────────────────────────────────────────────
import os, ssl

# 環境変数による無効化（一部ライブラリはこれを参照）
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["HF_DATASETS_OFFLINE"]              = "1"

# RT-DETRモデルキャッシュ確認 → キャッシュあり: オフライン / なし: ロードをスキップ
_v1_cache     = os.path.expanduser("~/.cache/huggingface/hub/models--PekingU--rtdetr_r50vd")
_v2_cache     = os.path.expanduser("~/.cache/huggingface/hub/models--PekingU--rtdetr_v2_r50vd")
_qwen2vl_cache = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct")
_RTDETR_CACHED  = os.path.isdir(_v1_cache) and os.path.isdir(_v2_cache)
_QWEN2VL_CACHED = os.path.isdir(_qwen2vl_cache)
# TRANSFORMERS_OFFLINE は使用しない
# → local_files_only=True を from_pretrained に直接渡してキャッシュのみ参照する
os.environ["CURL_CA_BUNDLE"]                  = ""
os.environ["REQUESTS_CA_BUNDLE"]              = ""
os.environ["PYTHONHTTPSVERIFY"]               = "0"

# Python ssl モジュールのデフォルト検証を無効化
ssl._create_default_https_context = ssl._create_unverified_context

# urllib3 警告抑制
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── requests.Session をモンキーパッチ ──────────────────────────
# transformers / huggingface_hub が内部で生成する Session も含め
# 全ての verify= 引数を強制的に False に上書きする。
import requests
_OrigSession = requests.Session

class _NoVerifySession(_OrigSession):
    """全リクエストで SSL 検証を強制無効化するセッション（社内プロキシ対応）"""
    def request(self, method, url, **kwargs):
        kwargs["verify"] = False
        return super().request(method, url, **kwargs)

requests.Session = _NoVerifySession

# huggingface_hub が既にインポート済みの場合に備えセッションを差し替え
try:
    import huggingface_hub.utils._http as _hf_http
    if hasattr(_hf_http, "get_session"):
        _hf_http.get_session = lambda: _NoVerifySession()
except Exception:
    pass

# ─────────────────────────────────────────────
# 通常 import（SSL 無効化済みの後）
# ─────────────────────────────────────────────
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import io
import requests
import time
import json
import re
import threading
from datetime import datetime
from pathlib import Path

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

try:
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

from dotenv import load_dotenv

# ─────────────────────────────────────────────
# .env 読み込み
# ─────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(str(_env_path))
else:
    for _c in [".env", "/Users/takaakiy/nvidia_vss/.env"]:
        if os.path.exists(_c):
            with open(_c) as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _k, _v = _line.split("=", 1)
                        os.environ.setdefault(_k.strip(), _v.strip())
            break

# ─────────────────────────────────────────────
# 設定定数
# ─────────────────────────────────────────────
NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY", "")
HF_TOKEN        = os.getenv("HF_TOKEN", "")
NIM_BASE_URL    = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
DOWNLOAD_DIR    = os.getenv("DOWNLOAD_DIR", str(Path(__file__).parent / "downloads"))
DEFAULT_FOLDER  = os.getenv("DEFAULT_VIDEO_FOLDER", DOWNLOAD_DIR)
SSL_CERT        = os.getenv("SSL_CERT_FILE", "")
SSL_KEY         = os.getenv("SSL_KEY_FILE", "")
# 社内 CA バンドルがある場合は REQUESTS_CA_BUNDLE に設定すると正規検証も可能
_ca_bundle      = os.getenv("REQUESTS_CA_BUNDLE", "")
SSL_VERIFY      = _ca_bundle if _ca_bundle else False

Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# NVIDIA NIM ビジョンモデル
# ─────────────────────────────────────────────
NIM_VISION_MODELS = {
    "Llama 3.2 11B Vision": {
        "id": "meta/llama-3.2-11b-vision-instruct",
        "desc": "バランス型・高速",
        "backend": "nim",
        "display": "Llama 3.2 11B Vision Instruct",
    },
    "Llama 3.2 90B Vision": {
        "id": "meta/llama-3.2-90b-vision-instruct",
        "desc": "最高精度・やや低速",
        "backend": "nim",
        "display": "Llama 3.2 90B Vision Instruct",
    },
    "Llama 4 Maverick 17B": {
        "id": "meta/llama-4-maverick-17b-128e-instruct",
        "desc": "最新世代モデル",
        "backend": "nim",
        "display": "Llama 4 Maverick 17B 128E Instruct",
    },
    "Llama 4 Scout 17B": {
        "id": "meta/llama-4-scout-17b-16e-instruct",
        "desc": "最新世代・効率型",
        "backend": "nim",
        "display": "Llama 4 Scout 17B 16E Instruct",
    },
    "NVIDIA Nemotron Nano VL 8B": {
        "id": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        "desc": "物体検出に強い",
        "backend": "nim",
        "display": "NVIDIA Nemotron Nano VL 8B v1",
    },
    "Microsoft Phi-4 Multimodal": {
        "id": "microsoft/phi-4-multimodal-instruct",
        "desc": "マルチモーダル最新型",
        "backend": "nim",
        "display": "Microsoft Phi-4 Multimodal Instruct",
    },
    "Microsoft Phi-3.5 Vision": {
        "id": "microsoft/phi-3.5-vision-instruct",
        "desc": "小型・高速",
        "backend": "nim",
        "display": "Microsoft Phi-3.5 Vision Instruct",
    },
}

# HuggingFace 無料ビジョンモデル（NIM 代替）
HF_VISION_MODELS = {
    "HF: Qwen2-VL 7B (Free)": {
        "id": "Qwen/Qwen2-VL-7B-Instruct",
        "desc": "HuggingFace 無料枠 / 高性能",
        "backend": "hf",
        "display": "Qwen2-VL 7B Instruct (HuggingFace)",
    },
    "HF: Llama 3.2 11B Vision (Free)": {
        "id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "desc": "HuggingFace 無料枠 / Meta Llama",
        "backend": "hf",
        "display": "Llama 3.2 11B Vision Instruct (HuggingFace)",
    },
    "HF: Pixtral 12B (Free)": {
        "id": "mistralai/Pixtral-12B-2409",
        "desc": "HuggingFace 無料枠 / Mistral",
        "backend": "hf",
        "display": "Pixtral 12B 2409 (HuggingFace)",
    },
}

# ローカル推論モデル（HuggingFace からダウンロード済み・完全オフライン動作）
LOCAL_VISION_MODELS = {
    "LOCAL: Qwen2.5-VL 7B": {
        "id":      "qwen2.5vl:7b",   # Ollama モデル名（ollama pull qwen2.5vl:7b）
        "desc":    "完全ローカル推論 / 日本語高精度 / Ollama",
        "backend": "ollama",
        "display": "Qwen2.5-VL 7B [LOCAL / Ollama]",
    },
}

ALL_VISION_MODELS = {**LOCAL_VISION_MODELS, **NIM_VISION_MODELS, **HF_VISION_MODELS}

TEXT_MODELS = {
    "Llama 3.3 70B":     {"id": "meta/llama-3.3-70b-instruct",                 "desc": "要約・分析向け"},
    "Llama 3.1 70B":     {"id": "meta/llama-3.1-70b-instruct",                 "desc": "バランス型"},
    "Nemotron Super 49B":{"id": "nvidia/llama-3.3-nemotron-super-49b-v1",      "desc": "推論特化・高精度"},
}

# RT-DETR 検知モデル（v1/v2 を個別管理）
DETECTION_MODELS = {
    "v1": {
        "id":        "PekingU/rtdetr_r50vd",
        "label":     "RT-DETR",
        "color":     (0, 200, 230),   # シアン BGR
        "color_hex": "#00c8e6",
    },
    "v2": {
        "id":        "PekingU/rtdetr_v2_r50vd",
        "label":     "RT-DETRv2",
        "color":     (80, 210, 110),  # グリーン BGR
        "color_hex": "#50d26e",
    },
}

# ─────────────────────────────────────────────
# プロンプトプリセット
# ─────────────────────────────────────────────
PROMPT_PRESETS = {
    "General Detection": (
        "この画像に写っているすべての要素を日本語で詳しく報告してください。"
        "【検出対象】人物・動物・車両・建物・自然物・テキスト・看板・製品・その他の物体。"
        "【報告形式】①主要な物体一覧（種類・数・位置）②人物の行動や状態 "
        "③背景・環境の特徴 ④特筆すべき異常や注目点"
    ),
    "Traffic / Vehicle": (
        "交通・車両分析の専門家として以下を日本語で報告してください。"
        "【必須検出項目】"
        "①車両：種類（乗用車/トラック/バス/二輪車）・台数・色・メーカー推定・ナンバープレート（読み取れる場合）"
        "②信号機：色（赤/黄/青）・矢印表示の有無"
        "③交通標識：種類・内容"
        "④歩行者・自転車：人数・位置・行動"
        "⑤道路状況：車線数・路面状態・工事や障害物の有無"
        "⑥総合判定：交通流の状態（スムーズ/混雑/停滞）"
    ),
    "Congestion Monitor": (
        "道路渋滞判定システムとして分析してください。"
        "【判定基準】車両台数10台以上または車間距離1台分以下 → 渋滞 / それ以外 → 順調"
        "【出力形式（必ずこの形式のみで回答）】"
        "上り車線：渋滞 or 順調（車両数：XX台）"
        "下り車線：渋滞 or 順調（車両数：XX台）"
        "判定根拠：（1行で簡潔に）"
    ),
    "Person / Action": (
        "人物行動分析の専門家として画像内のすべての人物を検出し、日本語で報告してください。"
        "【各人物について報告】"
        "①外見：服装の色・スタイル、推定年齢層（子供/若者/中年/高齢者）、性別推定"
        "②姿勢・動作：立つ/座る/歩く/走る/作業中 等の具体的な行動"
        "③表情・感情：判別できる場合のみ"
        "④位置関係：画面のどこにいるか、他の人物との関係"
        "⑤持ち物・携帯品：カバン・スマートフォン・その他"
        "⑥異常行動・注目点：不審な行動、危険な状況があれば優先報告"
    ),
    "Text / Signage OCR": (
        "OCR・文字認識の専門家として画像内のすべてのテキストを抽出・報告してください。"
        "【抽出対象】看板・標識・ラベル・ポスター・画面表示・ナンバープレート・価格表示・その他文字全般"
        "【言語】日本語・英語・中国語・韓国語・その他すべて"
        "【報告形式】"
        "①抽出テキスト一覧（読み取れたものをすべて）"
        "②各テキストの種類（看板/標識/ラベル等）と画面内の位置"
        "③読み取り不確実な箇所は「？」で明示"
        "④ブランド名・ロゴが確認できれば特記"
    ),
    "Equipment / Product": (
        "設備・製品検査の専門家として画像内の機器・製品・設備を技術的に分析してください。"
        "【検出・報告項目】"
        "①機器・製品の特定：種類・用途・ブランド名・型番（確認できる場合）"
        "②状態評価：正常/劣化/損傷/汚れ/異常の有無"
        "③ケーブル・配線・接続部の状態"
        "④表示パネル・ランプ・インジケータの状態（点灯/消灯/点滅・色）"
        "⑤異常・危険箇所：過熱の兆候・破損・不適切な設置等"
        "⑥保守推奨事項：気になる点があれば記載"
    ),
    "Security Monitor": (
        "セキュリティ監視の専門家として画像を安全管理の観点で分析してください。"
        "【監視項目】"
        "①人物：不審者の有無・行動パターン・滞留時間が長そうな人物"
        "②エリア：立入禁止区域への侵入・不正アクセスの疑い"
        "③物体：置き去り荷物・不審物・危険物の可能性があるもの"
        "④環境：照明状態・死角・セキュリティカメラの死角となりうる箇所"
        "⑤緊急度判定：異常なし / 要注意 / 要即時対応（いずれかを明記）"
        "⑥推奨アクション：必要な場合のみ記載"
    ),
    "Sports / Motion": (
        "スポーツ・動作分析の専門家として画像を分析し日本語で報告してください。"
        "【検出・分析項目】"
        "①競技種目の特定：スポーツ名・競技レベル推定（アマ/プロ）"
        "②選手・参加者：人数・ユニフォーム・ゼッケン番号"
        "③動作分析：フォーム・姿勢・動きの方向・スピード感"
        "④用具・器具：ボール・ラケット・ゴール・その他スポーツ用品"
        "⑤競技環境：競技場・コート・フィールドの種類と状態"
        "⑥試合状況：スコア表示・時間表示・審判の存在（確認できる場合）"
        "⑦注目ポイント：決定的瞬間・ファインプレー・異常な状況"
    ),
    "Shibuya Scramble": (
        "渋谷スクランブル交差点の定点観測システムとして分析してください。"
        "【報告項目】"
        "①人流：推定人数・密度（少/中/多/非常に多い）・流れの方向"
        "②車両：種類・台数・信号の色・交通状況"
        "③看板・広告：読み取れるブランド名・店舗名・広告内容"
        "④建物・ランドマーク：確認できる建物・施設名"
        "⑤天候・時間帯：昼/夜・晴れ/雨・混雑度の総合評価"
        "⑥特記事項：イベント・工事・異常な状況があれば優先報告"
    ),
}

SUMMARIZE_PROMPT = """あなたは動画解析の専門家です。
以下は動画から時系列で取得したフレーム分析結果（タイムスタンプ付き）です。
この情報を元に、動画全体の内容を日本語で包括的に要約してください。

要約には以下を含めてください：
1. 動画全体の概要  2. 主要な出来事・シーンの流れ（時系列）
3. 検出された主要な物体・人物・テキスト  4. 特筆すべき重要な瞬間やハイライト  5. 全体的な考察・結論

---
フレーム分析データ:
{analysis_data}
---
"""

# ─────────────────────────────────────────────
# Streamlit ページ設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Video Intelligence Terminal",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# カスタム CSS: 近未来風ダークブルー UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600&family=Share+Tech+Mono&display=swap');
.stApp { background-color: #010d1a; color: #b8d8e8; }
.stApp > header { background: transparent !important; }
div[data-testid="stSidebar"] { background: #020f1f; border-right: 1px solid #0a2a45; }
div[data-testid="stSidebar"] * { color: #8ab8cc !important; }
.vit-header {
    background: linear-gradient(135deg, #010d1a 0%, #021a35 40%, #02244a 100%);
    border: 1px solid #0a3a60; border-top: 2px solid #00b4d8;
    border-radius: 3px; padding: 18px 28px 14px; margin-bottom: 16px;
    position: relative; overflow: hidden;
}
.vit-header::before {
    content: ''; position: absolute; inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 3px,
        rgba(0,180,216,0.025) 3px, rgba(0,180,216,0.025) 4px);
    pointer-events: none;
}
.vit-header h1 {
    font-family: 'Orbitron', monospace; font-size: 1.55rem; font-weight: 900;
    color: #00d4f5; margin: 0; letter-spacing: 0.08em;
    text-shadow: 0 0 20px rgba(0,212,245,0.4);
}
.vit-header .subtitle {
    font-family: 'Rajdhani', sans-serif; font-size: 0.82rem; color: #3a7a99;
    margin: 4px 0 0; letter-spacing: 0.12em; text-transform: uppercase;
}
.vit-badge {
    display: inline-block; background: #00b4d8; color: #010d1a;
    font-family: 'Orbitron', monospace; font-size: 0.58rem; font-weight: 700;
    padding: 2px 7px; border-radius: 2px; margin-left: 10px; letter-spacing: 0.1em; vertical-align: middle;
}
.status-bar {
    display: flex; flex-wrap: wrap; gap: 16px; padding: 5px 14px;
    background: #020f1e; border: 1px solid #0a2a3e; border-radius: 2px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.67rem;
    margin-bottom: 10px; letter-spacing: 0.05em;
}
.status-item { color: #1e5570; text-transform: uppercase; }
.status-val  { color: #00b4d8; margin-left: 5px; }
.status-val.ok    { color: #00e5a0; }
.status-val.busy  { color: #ffd166; }
.status-val.error { color: #ef476f; }
.analysis-card {
    background: #020f1f; border: 1px solid #0a3a55;
    border-left: 3px solid #00b4d8; border-radius: 2px; padding: 10px 14px; margin: 6px 0;
}
.analysis-ts { font-family: 'Share Tech Mono', monospace; font-size: 0.68rem; color: #00b4d8; }
.analysis-body { color: #b0cce0; font-size: 0.84rem; margin-top: 5px; line-height: 1.6; }
.detection-card {
    background: #020f1f; border: 1px solid #0a3a55; border-radius: 2px;
    padding: 6px 10px; margin: 3px 0;
    font-family: 'Share Tech Mono', monospace; font-size: 0.7rem;
}
.det-v1 { border-left: 3px solid #00c8e6; }
.det-v2 { border-left: 3px solid #50d26e; }
.det-label { color: #d0e8f0; }
.det-score { color: #ffd166; margin-left: 8px; }
.det-tag { font-size: 0.58rem; color: #2a6070; margin-left: 6px; }
.search-highlight { background: #004d66; color: #00e5f0; padding: 0 3px; border-radius: 2px; }
.panel-title {
    font-family: 'Orbitron', monospace; font-size: 0.72rem; font-weight: 600;
    color: #2a7a9a; letter-spacing: 0.15em; text-transform: uppercase;
    margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid #0a2a40;
}
.section-label {
    font-family: 'Orbitron', monospace; font-size: 0.7rem;
    color: #2a6a88; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 8px;
}
.stButton > button {
    background: transparent; border: 1px solid #0a3a55; color: #5ab8d0;
    font-family: 'Rajdhani', sans-serif; font-size: 0.82rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; border-radius: 2px; transition: all 0.15s;
}
.stButton > button:hover { background: #0a2a40; border-color: #00b4d8; color: #00d4f5; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #003d5c, #00567a);
    border-color: #00b4d8; color: #00d4f5; font-weight: 700;
}
.stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #005070, #0070a0); }
div[data-testid="stTabs"] button {
    font-family: 'Orbitron', monospace !important; font-size: 0.68rem !important;
    letter-spacing: 0.1em !important; color: #2a6a88 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00d4f5 !important; border-bottom: 2px solid #00d4f5 !important;
}
.stTextInput input, .stTextArea textarea {
    background: #020f1f !important; border: 1px solid #0a3055 !important;
    color: #b0cce0 !important; border-radius: 2px !important;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #010d1a; }
::-webkit-scrollbar-thumb { background: #0a3a55; border-radius: 2px; }
.vit-footer {
    text-align: center; padding: 8px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.68rem; color: #1a4055;
    border-top: 1px solid #0a2a3e; margin-top: 16px; letter-spacing: 0.08em;
}
.det-model-header {
    font-family: 'Orbitron', monospace; font-size: 0.62rem;
    letter-spacing: 0.1em; padding: 3px 0 5px; margin-bottom: 4px;
    border-bottom: 1px solid #0a2a3e; text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ヘッダー
# ─────────────────────────────────────────────
st.markdown("""
<div class="vit-header">
    <h1>VIDEO INTELLIGENCE TERMINAL
        <span class="vit-badge">LIVE</span>
        <span style="font-family:'Rajdhani',sans-serif;font-size:0.9rem;color:#1a6080;
              font-weight:300;letter-spacing:0.05em;margin-left:14px;">
            RT-DETR Object Detection + AI Scene Analysis
        </span>
    </h1>
    <div class="subtitle">Powered by NVIDIA NIM / HuggingFace / RT-DETR v1+v2 | Parallel Processing Pipeline</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# APIキー確認
# ─────────────────────────────────────────────
if not NVIDIA_API_KEY and not HF_TOKEN:
    st.error("No API key found. Set NVIDIA_API_KEY or HF_TOKEN in your .env file.")
    st.code("NVIDIA_API_KEY=nvapi-xxxx...\nHF_TOKEN=hf_xxxx...", language="bash")
    st.stop()

# ─────────────────────────────────────────────
# セッション状態 初期化
# ─────────────────────────────────────────────
_defaults = {
    "processing":            False,
    "youtube_url":           "",
    "video_file":            None,
    "stream_url":            None,
    "is_live":               False,
    "stream_title":          "",
    "selected_model":        "LOCAL: Qwen2.5-VL 7B",
    "current_prompt":        PROMPT_PRESETS["General Detection"],
    "pending_prompt":        None,   # プリセット Apply 待ちプロンプト
    "analysis_log":          [],
    "summary_text":          "",
    "search_query":          "",
    "search_results":        [],
    "total_frames_analyzed": 0,
    "mode":                  "YouTube",
    "highlights":            [],
    "local_folder":          DEFAULT_FOLDER,
    # RT-DETR 個別 ON/OFF
    "det_v1_enabled":        True,
    "det_v2_enabled":        True,
    "last_det_count":        {"v1": 0, "v2": 0},
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────
# RT-DETR モデルキャッシュロード
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rtdetr_v1():
    """
    RT-DETR v1 をロード。
    キャッシュ未存在の場合は即座にスキップ（社内ネットワークでのタイムアウト防止）。
    社外で python3 download_models.py を実行後に有効になる。
    """
    if not DETECTION_AVAILABLE:
        return None, None, None, "torch/transformers not installed"
    if not _RTDETR_CACHED:
        return None, None, None, "モデル未キャッシュ。社外で download_models.py を実行してください"
    try:
        device = ("mps"  if torch.backends.mps.is_available() else
                  "cuda" if torch.cuda.is_available() else "cpu")
        cfg  = DETECTION_MODELS["v1"]
        proc  = AutoImageProcessor.from_pretrained(cfg["id"], trust_remote_code=False, local_files_only=True)
        model = AutoModelForObjectDetection.from_pretrained(cfg["id"], trust_remote_code=False, local_files_only=True)
        model.to(device)
        model.eval()
        return proc, model, device, None
    except Exception as exc:
        return None, None, None, str(exc)


@st.cache_resource(show_spinner=False)
def load_rtdetr_v2():
    """RT-DETRv2 をロード（キャッシュ未存在時は即スキップ）"""
    if not DETECTION_AVAILABLE:
        return None, None, None, "torch/transformers not installed"
    if not _RTDETR_CACHED:
        return None, None, None, "モデル未キャッシュ。社外で download_models.py を実行してください"
    try:
        device = ("mps"  if torch.backends.mps.is_available() else
                  "cuda" if torch.cuda.is_available() else "cpu")
        cfg  = DETECTION_MODELS["v2"]
        proc  = AutoImageProcessor.from_pretrained(cfg["id"], trust_remote_code=False, local_files_only=True)
        model = AutoModelForObjectDetection.from_pretrained(cfg["id"], trust_remote_code=False, local_files_only=True)
        model.to(device)
        model.eval()
        return proc, model, device, None
    except Exception as exc:
        return None, None, None, str(exc)


# ─────────────────────────────────────────────
# ユーティリティ関数
# ─────────────────────────────────────────────

def frame_to_base64(frame_rgb: np.ndarray, quality: int = 90) -> str:
    pil = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def resize_frame(frame: np.ndarray, scale_pct: int) -> np.ndarray:
    if scale_pct >= 100:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale_pct / 100), int(h * scale_pct / 100)),
                      interpolation=cv2.INTER_AREA)


def draw_detection_boxes(frame_rgb: np.ndarray, detections: dict) -> np.ndarray:
    """RT-DETR v1(シアン) / v2(グリーン) のボックスをフレームに描画"""
    frame = frame_rgb.copy()
    for ver, dets in detections.items():
        cfg   = DETECTION_MODELS.get(ver, {})
        color = cfg.get("color", (200, 200, 200))
        for det in (dets or []):
            x1, y1, x2, y2 = det["box"]
            score  = det["score"]
            lbl    = det["label"]
            text   = f"{lbl} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs   = 0.44
            th   = 1
            (tw, th_px), bl = cv2.getTextSize(text, font, fs, th)
            pad = 3
            cv2.rectangle(frame,
                (x1, max(0, y1 - th_px - pad * 2 - bl)),
                (x1 + tw + pad * 2, y1), color, -1)
            cv2.putText(frame, text,
                (x1 + pad, max(th_px + pad, y1 - bl - pad)),
                font, fs, (5, 5, 5), th, cv2.LINE_AA)
    return frame


def run_detection(frame_rgb: np.ndarray, use_v1: bool, use_v2: bool,
                  threshold: float = 0.45) -> dict:
    """
    有効なモデルだけ推論を実行。
    戻り値: {"v1": [...], "v2": [...], "error": None}
    """
    result = {"v1": [], "v2": [], "error": None}

    if use_v1:
        proc, model, device, err = load_rtdetr_v1()
        if err:
            result["error"] = f"v1: {err}"
        elif proc and model:
            try:
                pil = Image.fromarray(frame_rgb)
                inp = proc(images=pil, return_tensors="pt")
                inp = {k: v.to(device) for k, v in inp.items()}
                with torch.no_grad():
                    out = model(**inp)
                ts  = torch.tensor([[pil.height, pil.width]])
                preds = proc.post_process_object_detection(out, threshold=threshold, target_sizes=ts)[0]
                for sc, lid, box in zip(preds["scores"], preds["labels"], preds["boxes"]):
                    result["v1"].append({
                        "label": model.config.id2label[lid.item()],
                        "score": round(float(sc.item()), 3),
                        "box":   [round(float(v)) for v in box.tolist()],
                    })
            except Exception as exc:
                result["error"] = f"v1 inference: {exc}"

    if use_v2:
        proc, model, device, err = load_rtdetr_v2()
        if err:
            result["error"] = (result["error"] or "") + f" | v2: {err}"
        elif proc and model:
            try:
                pil = Image.fromarray(frame_rgb)
                inp = proc(images=pil, return_tensors="pt")
                inp = {k: v.to(device) for k, v in inp.items()}
                with torch.no_grad():
                    out = model(**inp)
                ts  = torch.tensor([[pil.height, pil.width]])
                preds = proc.post_process_object_detection(out, threshold=threshold, target_sizes=ts)[0]
                for sc, lid, box in zip(preds["scores"], preds["labels"], preds["boxes"]):
                    result["v2"].append({
                        "label": model.config.id2label[lid.item()],
                        "score": round(float(sc.item()), 3),
                        "box":   [round(float(v)) for v in box.tolist()],
                    })
            except Exception as exc:
                result["error"] = (result["error"] or "") + f" | v2 inference: {exc}"

    return result


def _ollama_analyze(frame_rgb, prompt, model_id, resize_pct, max_tokens) -> dict:
    """Ollama ローカル推論（Qwen2-VL 等のビジョンモデル対応）
    HuggingFace 版と異なり量子化済みモデルを使うため大幅に高速。
    事前に: ollama pull qwen2-vl:7b が必要。
    Ollama が起動していない場合は: ollama serve
    """
    if resize_pct < 100:
        frame_rgb = resize_frame(frame_rgb, resize_pct)
    img_b64 = frame_to_base64(frame_rgb)
    t0 = time.time()
    try:
        payload = {
            "model":  model_id,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "num_predict": min(max_tokens, 400),
                "temperature": 0.2,
            },
        }
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120,
            verify=False,
        )
        latency = time.time() - t0
        if resp.status_code == 200:
            return {
                "ok":      True,
                "text":    resp.json().get("response", ""),
                "latency": latency,
                "img_b64": img_b64,
            }
        elif resp.status_code == 404:
            return {
                "ok":      False,
                "text":    f"Ollamaモデル未インストール。Terminalで実行: ollama pull {model_id}",
                "latency": latency,
            }
        return {
            "ok":      False,
            "text":    f"Ollama HTTP {resp.status_code}: {resp.text[:200]}",
            "latency": latency,
        }
    except requests.exceptions.ConnectionError:
        return {
            "ok":      False,
            "text":    "Ollamaが起動していません。Terminalで: ollama serve",
            "latency": time.time() - t0,
        }
    except Exception as exc:
        return {"ok": False, "text": f"Ollamaエラー: {exc}", "latency": time.time() - t0}


def vlm_analyze(frame_rgb, prompt, model_name, resize_pct=100, max_tokens=600) -> dict:
    """VLM 分析（local / NIM / HF バックエンド自動選択）"""
    cfg = ALL_VISION_MODELS.get(model_name)
    if not cfg:
        return {"ok": False, "text": f"Unknown model: {model_name}", "latency": 0.0}
    if cfg["backend"] == "ollama":
        return _ollama_analyze(frame_rgb, prompt, cfg["id"], resize_pct, max_tokens)
    if cfg["backend"] == "nim":
        return _nim_analyze(frame_rgb, prompt, cfg["id"], resize_pct, max_tokens)
    return _hf_analyze(frame_rgb, prompt, cfg["id"], resize_pct, max_tokens)


def _nim_analyze(frame_rgb, prompt, model_id, resize_pct, max_tokens) -> dict:
    """NVIDIA NIM Vision API"""
    if resize_pct < 100:
        frame_rgb = resize_frame(frame_rgb, resize_pct)
    img_b64 = frame_to_base64(frame_rgb)
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]}],
        "max_tokens": max_tokens, "temperature": 0.2, "top_p": 0.7, "stream": False,
    }
    t0 = time.time()
    try:
        resp = requests.post(
            f"{NIM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"},
            json=payload, timeout=90, verify=SSL_VERIFY,
        )
        latency = time.time() - t0
        if resp.status_code == 200:
            return {"ok": True, "text": resp.json()["choices"][0]["message"]["content"],
                    "latency": latency, "img_b64": img_b64}
        return {"ok": False, "text": f"NIM {resp.status_code}: {resp.text[:300]}", "latency": latency}
    except requests.Timeout:
        return {"ok": False, "text": "Timeout (90s)", "latency": 90.0}
    except Exception as exc:
        return {"ok": False, "text": f"NIM error: {exc}", "latency": 0.0}


def _hf_analyze(frame_rgb, prompt, model_id, resize_pct, max_tokens) -> dict:
    """HuggingFace Inference API（NIM 代替・無料枠）"""
    if not HF_HUB_AVAILABLE:
        return {"ok": False, "text": "huggingface_hub not installed", "latency": 0.0}
    if not HF_TOKEN:
        return {"ok": False, "text": "HF_TOKEN not set in .env", "latency": 0.0}
    if resize_pct < 100:
        frame_rgb = resize_frame(frame_rgb, resize_pct)
    img_b64 = frame_to_base64(frame_rgb)
    t0 = time.time()
    try:
        client = InferenceClient(token=HF_TOKEN)
        response = client.chat_completion(
            model=model_id,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text",      "text": prompt},
            ]}],
            max_tokens=max_tokens,
        )
        return {"ok": True, "text": response.choices[0].message.content,
                "latency": time.time() - t0, "img_b64": img_b64}
    except Exception as exc:
        return {"ok": False, "text": f"HF error: {exc}", "latency": time.time() - t0}


def nim_text_summarize(analysis_log: list, model_name: str = "Llama 3.3 70B") -> dict:
    model_id = TEXT_MODELS.get(model_name, TEXT_MODELS["Llama 3.3 70B"])["id"]
    log_str  = "\n\n".join([f"[{e['ts']}] Frame {e['frame_idx']}: {e['text']}" for e in analysis_log])
    payload  = {
        "model": model_id,
        "messages": [{"role": "user", "content": SUMMARIZE_PROMPT.format(analysis_data=log_str)}],
        "max_tokens": 1500, "temperature": 0.3, "stream": False,
    }
    t0 = time.time()
    try:
        resp = requests.post(
            f"{NIM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"},
            json=payload, timeout=120, verify=SSL_VERIFY,
        )
        if resp.status_code == 200:
            return {"ok": True, "text": resp.json()["choices"][0]["message"]["content"],
                    "latency": time.time() - t0}
        return {"ok": False, "text": f"API {resp.status_code}: {resp.text[:300]}"}
    except Exception as exc:
        return {"ok": False, "text": str(exc)}


def search_analysis_log(log: list, query: str) -> list:
    if not query.strip():
        return []
    keywords = [k.strip().lower() for k in re.split(r"[\s\u3000,\u3001]", query) if k.strip()]
    return [e for e in log if any(kw in e["text"].lower() for kw in keywords)]


def extract_tags(log: list, top_n: int = 40) -> list:
    """
    分析ログから頻出キーワードを自動抽出してタグ一覧を返す。
    日本語・英語混在テキストを単語単位で分割し、
    ストップワードを除いて出現頻度順にソート。
    """
    import re
    from collections import Counter

    # 除外する短すぎる語・一般的すぎる語（日本語・英語）
    STOPWORDS = {
        "の","に","は","を","が","で","と","た","し","て","い","な","こ","れ","も","から",
        "です","ます","いる","ある","この","その","また","など","よう","これ","それ",
        "a","an","the","is","in","on","at","to","of","and","or","with","this","that",
        "are","was","were","be","been","has","have","it","its","for","as","by","from",
        "画像","写真","フレーム","場面","状況","内容","様子","見え","られ","なっ","おり",
    }

    counter = Counter()
    for entry in log:
        text = entry.get("text", "")
        # 日本語: 2文字以上の連続したひらがな・カタカナ・漢字
        jp_words = re.findall(r'[一-龯ぁ-んァ-ン]{2,}', text)
        # 英語: 3文字以上のアルファベット
        en_words = re.findall(r'[A-Za-z]{3,}', text)
        for w in jp_words + en_words:
            wl = w.lower()
            if wl not in STOPWORDS and len(wl) >= 2:
                counter[w] += 1

    # 出現2回以上のものをタグ候補とする
    tags = [w for w, c in counter.most_common(top_n) if c >= 1]
    return tags


def highlight_text(text: str, query: str) -> str:
    if not query:
        return text
    for kw in re.split(r"[\s\u3000,\u3001]", query):
        if kw.strip():
            text = re.sub(re.escape(kw.strip()),
                f'<span class="search-highlight">{kw.strip()}</span>',
                text, flags=re.IGNORECASE)
    return text


def detect_highlights(log: list, top_n: int = 5) -> list:
    if not log:
        return []
    return sorted(log, key=lambda x: len(x.get("text", "")), reverse=True)[:top_n]


def download_youtube(url: str) -> tuple:
    if not YT_DLP_AVAILABLE:
        return None, "yt-dlp not installed"
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = str(Path(DOWNLOAD_DIR) / f"video_{ts}.%(ext)s")
    opts = {"format": "best[height<=720][ext=mp4]/best[ext=mp4]/best",
            "outtmpl": out, "quiet": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info  = ydl.extract_info(url, download=True)
            title = info.get("title", "Unknown")
        for ext in [".mp4", ".webm", ".mkv", ".m4a"]:
            p = Path(DOWNLOAD_DIR) / f"video_{ts}{ext}"
            if p.exists():
                return str(p), title
        for f in Path(DOWNLOAD_DIR).iterdir():
            if f.name.startswith(f"video_{ts}"):
                return str(f), f.stem
        return None, "File not found after download"
    except Exception as exc:
        return None, str(exc)


def get_stream_url(url: str) -> tuple:
    if not YT_DLP_AVAILABLE:
        return None, "", False, "yt-dlp not installed"
    opts = {"format": "best[height<=720]/best", "quiet": True,
            "no_warnings": True, "noplaylist": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info       = ydl.extract_info(url, download=False)
            title      = info.get("title", "Unknown")
            is_live    = info.get("is_live", False)
            stream_url = info.get("url")
            if not stream_url and "formats" in info:
                fmts = info["formats"]
                for fmt in reversed(fmts):
                    h = fmt.get("height") or 9999
                    if h <= 720 and fmt.get("url"):
                        stream_url = fmt["url"]; break
                if not stream_url:
                    stream_url = fmts[-1].get("url", "")
            if stream_url:
                return stream_url, title, is_live, None
            return None, title, is_live, "Could not get stream URL"
    except Exception as exc:
        return None, "", False, str(exc)


def get_video_files(folder: str) -> list:
    video_exts = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v", ".ts"}
    try:
        return sorted([f for f in Path(folder).iterdir()
                       if f.suffix.lower() in video_exts], reverse=True)
    except Exception:
        return []


def make_status_bar(status, latency, frames, model_full, det_v1, det_v2) -> str:
    sc = "ok" if status in ("READY", "OK") else ("busy" if status == "ANALYZING" else "")
    return (
        f"<div class='status-bar'>"
        f"<span class='status-item'>STATUS <span class='status-val {sc}'>{status}</span></span>"
        f"<span class='status-item'>LATENCY <span class='status-val'>{latency}</span></span>"
        f"<span class='status-item'>FRAMES <span class='status-val'>{frames}</span></span>"
        f"<span class='status-item'>RT-DETR <span class='status-val'>{det_v1}</span>"
        f" / DETRv2 <span class='status-val'>{det_v2}</span></span>"
        f"<span class='status-item'>MODEL <span class='status-val'>{model_full}</span></span>"
        f"</div>"
    )

# ─────────────────────────────────────────────
# サイドバー
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Orbitron,monospace;font-size:0.78rem;color:#1a7090;"
        "letter-spacing:0.15em;padding:8px 0 4px;text-transform:uppercase;'>Configuration</div>",
        unsafe_allow_html=True,
    )

    # API Status
    with st.expander("API Status", expanded=False):
        if NVIDIA_API_KEY:
            st.success(f"NVIDIA API Key set  `...{NVIDIA_API_KEY[-8:]}`")
        else:
            st.warning("NVIDIA_API_KEY not set")
        if HF_TOKEN:
            st.success(f"HuggingFace Token set  `...{HF_TOKEN[-8:]}`")
        else:
            st.info("HF_TOKEN not set")
        ssl_mode = "CA Bundle: " + _ca_bundle if _ca_bundle else "SSL verify: OFF (proxy mode)"
        st.caption(ssl_mode)
        if st.button("Test NIM Connection", width='stretch'):
            with st.spinner("Testing..."):
                try:
                    r = requests.get(f"{NIM_BASE_URL}/models",
                                     headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
                                     timeout=10, verify=SSL_VERIFY)
                    if r.status_code == 200:
                        st.success("OK — NIM API 接続成功")
                    else:
                        st.error(f"HTTP {r.status_code}: {r.text[:100]}")
                except Exception as exc:
                    st.error(str(exc))

    st.divider()

    # Input Source
    st.markdown("<div class='section-label'>Input Source</div>", unsafe_allow_html=True)
    source_mode = st.radio(
        "Input Source",
        ["YouTube (Download)", "YouTube (Live)", "Local Folder"],
        label_visibility="collapsed",
    )
    if "Live" in source_mode:
        st.session_state.mode = "Live"
    elif "YouTube" in source_mode:
        st.session_state.mode = "YouTube"
    else:
        st.session_state.mode = "Local"

    if st.session_state.mode == "YouTube":
        yt_url = st.text_input("YouTube URL", value=st.session_state.youtube_url,
                               placeholder="https://youtu.be/xxxx")
        if yt_url != st.session_state.youtube_url:
            st.session_state.youtube_url = yt_url
            st.session_state.video_file  = None
            st.session_state.stream_url  = None
        with st.expander("URL Presets"):
            _pu = {
                "Shibuya Scramble": ("https://youtu.be/7RCvUO7omGM", "Shibuya Scramble"),
                "Road Traffic 1":   ("https://youtu.be/oe0apEwF2wM", "Traffic / Vehicle"),
                "Road Traffic 2":   ("https://youtu.be/PCCbzqWqyKc", "Traffic / Vehicle"),
            }
            for _lbl, (_url, _pk) in _pu.items():
                if st.button(_lbl, width='stretch', key=f"pyt_{_lbl}"):
                    st.session_state.youtube_url    = _url
                    st.session_state.current_prompt = PROMPT_PRESETS.get(_pk, PROMPT_PRESETS["General Detection"])
                    st.session_state.video_file = None; st.session_state.stream_url = None
                    st.rerun()
        if st.button("Download Video", width='stretch', type="primary"):
            if not st.session_state.youtube_url:
                st.warning("Enter a URL first")
            else:
                with st.spinner("Downloading..."):
                    path, title = download_youtube(st.session_state.youtube_url)
                    if path:
                        st.session_state.video_file = path; st.session_state.stream_url = None
                        st.success(f"Ready: {title}")
                    else:
                        st.error(f"Failed: {title}")
        if st.session_state.video_file and Path(st.session_state.video_file).exists():
            fsize = Path(st.session_state.video_file).stat().st_size / 1024 / 1024
            st.caption(f"{Path(st.session_state.video_file).name}  ({fsize:.1f} MB)")

    elif st.session_state.mode == "Live":
        st.caption("Stream is analyzed directly — no download needed.")
        live_url = st.text_input("Live Stream URL", value=st.session_state.youtube_url,
                                 placeholder="https://www.youtube.com/live/xxxx")
        if live_url != st.session_state.youtube_url:
            st.session_state.youtube_url = live_url; st.session_state.stream_url = None
        with st.expander("Live Presets", expanded=True):
            _lp = {
                "Shibuya Scramble (Live)": ("https://youtu.be/7RCvUO7omGM", "Shibuya Scramble"),
                "Tokyo Expressway (Live)": ("https://www.youtube.com/live/uL-DQhXR57I", "Congestion Monitor"),
            }
            for _lbl, (_url, _pk) in _lp.items():
                if st.button(_lbl, width='stretch', key=f"plv_{_lbl}"):
                    st.session_state.youtube_url    = _url
                    st.session_state.current_prompt = PROMPT_PRESETS.get(_pk, PROMPT_PRESETS["General Detection"])
                    st.session_state.stream_url = None; st.rerun()
        if st.button("Connect Stream", width='stretch', type="primary"):
            if not st.session_state.youtube_url:
                st.warning("Enter a stream URL first")
            else:
                with st.spinner("Resolving stream URL..."):
                    surl, title, is_live, err = get_stream_url(st.session_state.youtube_url)
                    if surl:
                        st.session_state.stream_url   = surl
                        st.session_state.stream_title = title
                        st.session_state.is_live      = is_live
                        st.session_state.video_file   = None
                        st.success(f"{'LIVE' if is_live else 'VOD'} — {title}")
                    else:
                        st.error(f"Failed: {err}")
        if st.session_state.stream_url:
            st.caption(f"{'LIVE' if st.session_state.is_live else 'VOD'}: {st.session_state.stream_title[:35]}...")

    else:  # Local Folder
        folder_path = st.text_input("Video Folder Path", value=st.session_state.local_folder,
                                    placeholder="/path/to/videos")
        if folder_path != st.session_state.local_folder:
            st.session_state.local_folder = folder_path
        local_files = get_video_files(st.session_state.local_folder)
        if local_files:
            selected = st.selectbox(
                "Select File", options=local_files,
                format_func=lambda p: f"{p.name}  ({p.stat().st_size/1024/1024:.1f} MB)",
            )
            st.session_state.video_file = str(selected)
            st.session_state.stream_url = None
        else:
            st.warning(f"No video files in: {st.session_state.local_folder}")

    st.divider()

    # Vision Model
    st.markdown("<div class='section-label'>Vision Model</div>", unsafe_allow_html=True)
    _opts    = list(ALL_VISION_MODELS.keys())
    _def_idx = _opts.index(st.session_state.selected_model) \
               if st.session_state.selected_model in _opts else 0
    selected_model = st.selectbox(
        "Vision Model", options=_opts, index=_def_idx,
        label_visibility="collapsed",
    )
    st.session_state.selected_model = selected_model
    _mcfg = ALL_VISION_MODELS[selected_model]
    st.caption(f"`{_mcfg['id']}`")
    st.caption(f"{_mcfg['backend'].upper()} — {_mcfg['desc']}")

    st.divider()

    # ── RT-DETR 物体検知: v1 / v2 を個別 ON/OFF ──
    st.markdown("<div class='section-label'>Object Detection</div>", unsafe_allow_html=True)

    det_v1_enabled = st.checkbox(
        "RT-DETR  (cyan box)",
        value=st.session_state.det_v1_enabled,
        help="PekingU/rtdetr_r50vd — シアンのボックスで描画",
    )
    det_v2_enabled = st.checkbox(
        "RT-DETRv2  (green box)",
        value=st.session_state.det_v2_enabled,
        help="PekingU/rtdetr_v2_r50vd — グリーンのボックスで描画",
    )
    st.session_state.det_v1_enabled = det_v1_enabled
    st.session_state.det_v2_enabled = det_v2_enabled

    det_threshold = st.slider("Confidence Threshold", 0.2, 0.9, 0.45, 0.05)
    det_interval  = st.slider("Detection Interval (s)", 0.5, 10.0, 2.0, 0.5)

    det_any = det_v1_enabled or det_v2_enabled
    if det_any and DETECTION_AVAILABLE:
        if st.button("Pre-load Detection Models", width='stretch'):
            with st.spinner("Loading from HuggingFace Hub..."):
                msgs = []
                if det_v1_enabled:
                    _, _, dev, err = load_rtdetr_v1()
                    msgs.append(f"RT-DETR: {'OK on ' + dev if not err else 'ERR ' + str(err)}")
                if det_v2_enabled:
                    _, _, dev, err = load_rtdetr_v2()
                    msgs.append(f"RT-DETRv2: {'OK on ' + dev if not err else 'ERR ' + str(err)}")
                st.success("  |  ".join(msgs))
    elif not DETECTION_AVAILABLE:
        st.caption("Install torch + transformers to enable")

    st.divider()

    # Analysis Settings
    st.markdown("<div class='section-label'>Analysis Settings</div>", unsafe_allow_html=True)
    img_resize     = st.slider("Image Resize (%)",            25, 100, 80, 5)
    proc_interval  = st.slider("VLM Interval (s)",             2,  30,  8)
    playback_dur   = st.slider("Max Analysis Duration (s)",   10, 600, 120)
    max_tokens_val = st.slider("Max Tokens",                 200, 1200, 600, 100)

    st.divider()
    start_btn = st.button("Start Analysis", width='stretch', type="primary")
    stop_btn  = st.button("Stop",           width='stretch')
    if st.button("Clear All Logs", width='stretch'):
        st.session_state.analysis_log            = []
        st.session_state.summary_text            = ""
        st.session_state.search_results          = []
        st.session_state.highlights              = []
        st.session_state.total_frames_analyzed   = 0
        st.rerun()

# ─────────────────────────────────────────────
# メインタブ
# ─────────────────────────────────────────────
tab_live, tab_search, tab_summary, tab_highlight, tab_log = st.tabs([
    "LIVE DETECTION", "VIDEO SEARCH", "SUMMARIZATION", "HIGHLIGHTS", "ANALYSIS LOG",
])

# ════════════════════════════════════════════
# タブ1: LIVE DETECTION（並列処理メインループ）
# ════════════════════════════════════════════
with tab_live:

    with st.expander("Analysis Prompt", expanded=True):
        # ウィジェット生成「前」に pending_prompt を current_prompt に反映する
        # （生成後に key 付き state を書き換えると Streamlit が例外を出すため）
        if st.session_state.get("pending_prompt"):
            st.session_state.current_prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = None

        _pc1, _pc2 = st.columns([3, 1])
        with _pc1:
            edited_prompt = st.text_area(
                "Prompt (editable)",
                value=st.session_state.current_prompt,
                height=110,
                # key を持たせない → value= が毎 rerun で正しく反映される
            )
            if edited_prompt != st.session_state.current_prompt:
                st.session_state.current_prompt = edited_prompt
        with _pc2:
            st.markdown(
                "<div style='font-family:Orbitron,monospace;font-size:0.65rem;color:#1a6a88;"
                "letter-spacing:0.1em;margin-top:4px;margin-bottom:6px;'>PRESETS</div>",
                unsafe_allow_html=True,
            )
            preset_sel = st.selectbox(
                "Prompt Presets",
                options=list(PROMPT_PRESETS.keys()),
                label_visibility="collapsed",
                key="preset_sel",
            )
            if st.button("Apply", width='stretch'):
                # pending_prompt に保存 → rerun → ウィジェット生成「前」に適用
                # （生成後に key 付き state を書き換えると Streamlit が例外を出すため）
                st.session_state.pending_prompt = PROMPT_PRESETS[preset_sel]
                st.rerun()

    # ステータスバー（注釈サイズ）
    _status_ph = st.empty()
    _mdisplay  = ALL_VISION_MODELS.get(st.session_state.selected_model, {}).get(
                     "display", st.session_state.selected_model)
    _status_ph.markdown(
        make_status_bar("STANDBY", "---", st.session_state.total_frames_analyzed,
                        _mdisplay, 0, 0),
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin:4px 0 8px;border-top:1px solid #0a2a3e;'></div>",
                unsafe_allow_html=True)

    # ── 3カラムレイアウト: 映像 | 検知結果 | VLM説明 ──
    vcol, dcol, rcol = st.columns([3, 1.5, 1.5])

    with vcol:
        st.markdown("<div class='section-label'>Video Feed + Detection Overlay</div>",
                    unsafe_allow_html=True)
        video_ph  = st.empty()
        vcap_info = st.empty()

    with dcol:
        # 有効なモデルに応じてヘッダーを動的生成
        _v1_label = (f"<span style='color:#00c8e6;'>RT-DETR</span>"
                     if st.session_state.det_v1_enabled else
                     "<span style='color:#1a4050;'>RT-DETR (off)</span>")
        _v2_label = (f"<span style='color:#50d26e;'>RT-DETRv2</span>"
                     if st.session_state.det_v2_enabled else
                     "<span style='color:#1a4050;'>RT-DETRv2 (off)</span>")
        st.markdown(
            f"<div class='section-label'>{_v1_label} &nbsp;+&nbsp; {_v2_label}</div>",
            unsafe_allow_html=True,
        )
        det_ph = st.empty()

    with rcol:
        st.markdown("<div class='section-label'>AI Scene Analysis (JPN)</div>",
                    unsafe_allow_html=True)
        result_ph = st.empty()

    status_ph = st.empty()

    # ════════════════════════════════════════
    # 並列処理ループ起動
    # ════════════════════════════════════════
    if start_btn:
        use_stream = bool(st.session_state.stream_url)
        use_file   = bool(st.session_state.video_file
                          and Path(st.session_state.video_file).exists())

        if not use_stream and not use_file:
            status_ph.warning("Connect to a stream or prepare a video file first.")
        else:
            st.session_state.processing = True
            cap_src = st.session_state.stream_url if use_stream else st.session_state.video_file
            cap = cv2.VideoCapture(cap_src)
            if use_stream:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                status_ph.error("Cannot open video / stream.")
                st.session_state.processing = False
            else:
                fps          = cap.get(cv2.CAP_PROP_FPS) or 30
                total_fr     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                is_stream    = use_stream or (total_fr <= 0)
                end_frame    = min(int(playback_dur * fps), total_fr) if not is_stream else None

                current_prompt = st.session_state.current_prompt
                current_model  = st.session_state.selected_model
                use_v1         = st.session_state.det_v1_enabled
                use_v2         = st.session_state.det_v2_enabled

                # ── スレッド間共有辞書 ──
                shared = {
                    "running":          True,
                    "vlm_busy":         False, "vlm_frame": None,
                    "vlm_pos":          0.0,   "vlm_ts":    "",
                    "vlm_result":       None,  "vlm_ready": False,
                    "vlm_last_t":       0.0,
                    "det_busy":         False, "det_frame": None,
                    "det_result":       None,  "det_ready": False,
                    "det_last_t":       0.0,
                    "latest_det":       {"v1": [], "v2": []},
                }
                lock = threading.Lock()

                # VLM スレッド
                def vlm_worker():
                    while shared["running"]:
                        time.sleep(0.05)
                        with lock:
                            if shared["vlm_busy"] or shared["vlm_frame"] is None:
                                continue
                            snap = shared["vlm_frame"]
                            shared["vlm_busy"] = True; shared["vlm_frame"] = None
                        result = vlm_analyze(snap, current_prompt, current_model,
                                             resize_pct=img_resize, max_tokens=max_tokens_val)
                        with lock:
                            shared["vlm_result"] = result
                            shared["vlm_ready"]  = True
                            shared["vlm_busy"]   = False

                # 検知スレッド（v1/v2 選択状態を反映）
                def det_worker():
                    while shared["running"]:
                        time.sleep(0.05)
                        with lock:
                            if shared["det_busy"] or shared["det_frame"] is None:
                                continue
                            snap = shared["det_frame"]
                            shared["det_busy"] = True; shared["det_frame"] = None
                        dr = run_detection(snap, use_v1, use_v2, threshold=det_threshold)
                        with lock:
                            shared["det_result"]  = dr
                            shared["det_ready"]   = True
                            shared["det_busy"]    = False
                            shared["latest_det"]  = dr

                # スレッド起動
                threading.Thread(target=vlm_worker, daemon=True).start()
                if (use_v1 or use_v2) and DETECTION_AVAILABLE:
                    threading.Thread(target=det_worker, daemon=True).start()

                frame_count = 0
                start_time  = time.time()
                vlm_res     = None

                # ── メインループ（表示専念）──
                while st.session_state.processing and cap.isOpened():
                    if is_stream:
                        cap.grab(); ret, frame = cap.retrieve()
                    else:
                        ret, frame = cap.read()

                    if not ret:
                        if is_stream:
                            status_ph.warning("Frame read failed — retrying...")
                            time.sleep(1); continue
                        else:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); frame_count = 0
                            status_ph.success("Loop playback"); continue

                    if not is_stream and frame_count >= end_frame:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0); frame_count = 0
                        status_ph.success("Loop playback"); continue

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    elapsed   = time.time() - start_time
                    pos_sec   = frame_count / fps if fps > 0 else elapsed
                    now       = time.time()

                    # VLM 送信トリガー
                    with lock:
                        if (not shared["vlm_busy"] and shared["vlm_frame"] is None
                                and (now - shared["vlm_last_t"]) >= proc_interval):
                            shared["vlm_frame"]  = frame_rgb.copy()
                            shared["vlm_pos"]    = round(pos_sec, 1)
                            shared["vlm_ts"]     = datetime.now().strftime("%H:%M:%S")
                            shared["vlm_last_t"] = now

                    # 検知 送信トリガー
                    if (use_v1 or use_v2) and DETECTION_AVAILABLE:
                        with lock:
                            if (not shared["det_busy"] and shared["det_frame"] is None
                                    and (now - shared["det_last_t"]) >= det_interval):
                                shared["det_frame"]  = frame_rgb.copy()
                                shared["det_last_t"] = now

                    # 最新ボックスをオーバーレイ
                    with lock:
                        latest = dict(shared["latest_det"])
                    if (use_v1 or use_v2) and (latest["v1"] or latest["v2"]):
                        disp = draw_detection_boxes(frame_rgb, latest)
                    else:
                        disp = frame_rgb

                    # フレーム表示（スレッド待ちで止まらない）
                    video_ph.image(disp, channels="RGB", width='stretch')

                    _busy = shared["vlm_busy"]
                    _ann  = "  |  Analyzing..." if _busy else ""
                    if is_stream:
                        vcap_info.caption(
                            f"Elapsed {elapsed:.0f}s  |  Frame {frame_count}  |  "
                            f"{datetime.now().strftime('%H:%M:%S')}{_ann}")
                    else:
                        max_s = min(playback_dur, total_fr / fps) if fps > 0 else playback_dur
                        vcap_info.caption(f"{pos_sec:.1f}s / {max_s:.0f}s  |  Frame {frame_count}{_ann}")

                    # VLM 結果受け取り
                    with lock:
                        has_vlm  = shared["vlm_ready"]
                        vlm_res  = shared["vlm_result"] if has_vlm else vlm_res
                        saved_ts = shared["vlm_ts"]
                        saved_ps = shared["vlm_pos"]
                        if has_vlm:
                            shared["vlm_ready"] = False

                    if has_vlm and vlm_res:
                        if vlm_res["ok"]:
                            st.session_state.total_frames_analyzed += 1
                            st.session_state.analysis_log.append({
                                "ts": saved_ts, "frame_idx": frame_count,
                                "pos_sec": saved_ps, "text": vlm_res["text"],
                                "img_b64": vlm_res.get("img_b64", ""),
                                "latency": vlm_res["latency"],
                            })
                            result_ph.markdown(
                                f"<div class='analysis-card'>"
                                f"<div class='analysis-ts'>{saved_ts}  |  {saved_ps}s</div>"
                                f"<div class='analysis-body'>{vlm_res['text']}</div>"
                                f"</div>", unsafe_allow_html=True)
                        else:
                            result_ph.error(vlm_res["text"])

                    # 検知結果受け取り & 検知パネル更新
                    with lock:
                        has_det = shared["det_ready"]
                        det_res = shared["det_result"] if has_det else None
                        if has_det:
                            shared["det_ready"] = False

                    if has_det and det_res:
                        v1d = det_res.get("v1", [])
                        v2d = det_res.get("v2", [])
                        st.session_state.last_det_count = {"v1": len(v1d), "v2": len(v2d)}

                        det_html = ""
                        # RT-DETR v1 セクション
                        if use_v1:
                            det_html += (
                                f"<div class='det-model-header' style='color:#00c8e6;'>"
                                f"RT-DETR  [{len(v1d)} detections]</div>"
                            )
                            if v1d:
                                for d in v1d[:12]:
                                    det_html += (
                                        f"<div class='detection-card det-v1'>"
                                        f"<span class='det-label'>{d['label']}</span>"
                                        f"<span class='det-score'>{d['score']:.2f}</span>"
                                        f"</div>"
                                    )
                            else:
                                det_html += "<div style='color:#1a4a5a;font-size:0.7rem;padding:4px 8px;'>No detections</div>"

                        # RT-DETRv2 セクション
                        if use_v2:
                            det_html += (
                                f"<div class='det-model-header' style='color:#50d26e;margin-top:8px;'>"
                                f"RT-DETRv2  [{len(v2d)} detections]</div>"
                            )
                            if v2d:
                                for d in v2d[:12]:
                                    det_html += (
                                        f"<div class='detection-card det-v2'>"
                                        f"<span class='det-label'>{d['label']}</span>"
                                        f"<span class='det-score'>{d['score']:.2f}</span>"
                                        f"</div>"
                                    )
                            else:
                                det_html += "<div style='color:#1a4a5a;font-size:0.7rem;padding:4px 8px;'>No detections</div>"

                        if not det_html:
                            det_html = "<div style='color:#1a4050;font-size:0.75rem;padding:8px;'>Detection disabled</div>"
                        det_ph.markdown(det_html, unsafe_allow_html=True)

                    # ステータスバー更新
                    _vlm_st   = ("ANALYZING" if shared["vlm_busy"] else
                                 ("READY" if (vlm_res and vlm_res.get("ok")) else "STANDBY"))
                    _lat_str  = (f"{vlm_res['latency']:.1f}s"
                                 if (vlm_res and vlm_res.get("ok")) else "---")
                    _status_ph.markdown(
                        make_status_bar(
                            _vlm_st, _lat_str,
                            st.session_state.total_frames_analyzed,
                            ALL_VISION_MODELS.get(current_model, {}).get("display", current_model),
                            st.session_state.last_det_count["v1"],
                            st.session_state.last_det_count["v2"],
                        ), unsafe_allow_html=True)

                    frame_count += 1
                    time.sleep(0.067 if is_stream else 0.05)

                    if stop_btn or not st.session_state.processing:
                        break

                with lock:
                    shared["running"] = False
                cap.release()
                st.session_state.processing = False
                status_ph.info("Analysis stopped")

    if stop_btn:
        st.session_state.processing = False

# ════════════════════════════════════════════
# タブ2: VIDEO SEARCH（自動タグ付け＋キーワード検索）
# ════════════════════════════════════════════
with tab_search:
    st.markdown("<div class='panel-title'>Video Search</div>", unsafe_allow_html=True)
    st.caption("Auto-extracted tags from analysis log — click to filter frames")

    if not st.session_state.analysis_log:
        st.info("Run a live analysis first — tags and results will appear here.")
    else:
        st.success(f"Analyzed frames: **{len(st.session_state.analysis_log)}**")

        # ── 自動タグ抽出・表示 ──────────────────────────────────
        auto_tags = extract_tags(st.session_state.analysis_log, top_n=50)
        if auto_tags:
            st.markdown(
                "<div style='font-family:Orbitron,monospace;font-size:0.65rem;"
                "color:#2a6a88;letter-spacing:0.12em;margin-bottom:6px;'>AUTO TAGS</div>",
                unsafe_allow_html=True,
            )
            # タグをボタン風に横並び表示
            # クリックで search_query に反映
            _tag_cols = st.columns(min(len(auto_tags), 8))
            for _i, _tag in enumerate(auto_tags[:40]):
                with _tag_cols[_i % 8]:
                    if st.button(
                        _tag,
                        key=f"tag_{_i}_{_tag}",
                        width='stretch',
                    ):
                        st.session_state.search_query = _tag
                        st.session_state.search_results = search_analysis_log(
                            st.session_state.analysis_log, _tag)
                        st.rerun()

        st.markdown("<div style='margin:8px 0 4px;border-top:1px solid #0a2a3e;'></div>",
                    unsafe_allow_html=True)

        # ── キーワード手入力検索 ────────────────────────────────
        _sc1, _sc2 = st.columns([4, 1])
        with _sc1:
            search_q = st.text_input(
                "Search Query",
                value=st.session_state.search_query,
                placeholder="keyword  (or click a tag above)",
                help="Space / comma separated — OR search",
                key="search_input",
            )
        with _sc2:
            st.markdown("<br>", unsafe_allow_html=True)
            do_search = st.button("Search", width='stretch', type="primary")

        if do_search or (search_q != st.session_state.search_query):
            st.session_state.search_query   = search_q
            st.session_state.search_results = search_analysis_log(
                st.session_state.analysis_log, search_q)

        # ── 検索結果表示 ────────────────────────────────────────
        if st.session_state.search_results:
            st.markdown(f"**{len(st.session_state.search_results)} frames matched**")
            for entry in st.session_state.search_results:
                _e1, _e2 = st.columns([1, 3])
                with _e1:
                    if entry.get("img_b64"):
                        st.image(
                            base64.b64decode(entry["img_b64"]),
                            caption=f"{entry['pos_sec']}s",
                            width='stretch',
                        )
                with _e2:
                    st.markdown(
                        f"<div class='analysis-card'>"
                        f"<div class='analysis-ts'>"
                        f"{entry['ts']}  |  Frame {entry['frame_idx']}  ({entry['pos_sec']}s)"
                        f"</div>"
                        f"<div class='analysis-body'>"
                        f"{highlight_text(entry['text'], st.session_state.search_query)}"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        elif st.session_state.search_query:
            st.warning("No matching frames found.")

# ════════════════════════════════════════════
# タブ3: SUMMARIZATION
# ════════════════════════════════════════════
with tab_summary:
    st.markdown("<div class='panel-title'>Video Summarization</div>", unsafe_allow_html=True)
    st.caption("AI integrates all frame analyses into a comprehensive report")
    if not st.session_state.analysis_log:
        st.info("Run a live analysis first.")
    else:
        st.info(f"Source data: **{len(st.session_state.analysis_log)} frames** analyzed")
        _sm1, _sm2 = st.columns([2, 1])
        with _sm1:
            sum_model = st.selectbox("Summarization Model", options=list(TEXT_MODELS.keys()))
        with _sm2:
            st.markdown("<br>", unsafe_allow_html=True)
            gen_summary = st.button("Generate Summary", width='stretch', type="primary")
        if gen_summary:
            with st.spinner(f"Generating from {len(st.session_state.analysis_log)} frames..."):
                res = nim_text_summarize(st.session_state.analysis_log, sum_model)
                if res["ok"]:
                    st.session_state.summary_text = res["text"]
                    st.success(f"Done ({res['latency']:.1f}s)")
                else:
                    st.error(f"Failed: {res['text']}")
        if st.session_state.summary_text:
            st.markdown("---")
            st.markdown("<div class='panel-title'>Summary Report</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='background:#020f1f;border:1px solid #0a3055;"
                f"border-left:3px solid #00b4d8;border-radius:3px;padding:18px;"
                f"color:#b0cce0;line-height:1.7;'>"
                f"{st.session_state.summary_text.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True)
            st.download_button("Download Summary", data=st.session_state.summary_text,
                               file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                               mime="text/plain")

# ════════════════════════════════════════════
# タブ4: HIGHLIGHTS
# ════════════════════════════════════════════
with tab_highlight:
    st.markdown("<div class='panel-title'>Highlight Detection</div>", unsafe_allow_html=True)
    st.caption("Auto-extracts frames with highest information density")
    if not st.session_state.analysis_log:
        st.info("Run a live analysis first.")
    else:
        top_n = st.slider("Number of highlights", 1, min(10, len(st.session_state.analysis_log)), 5)
        if st.button("Extract Highlights", width='stretch', type="primary"):
            st.session_state.highlights = detect_highlights(st.session_state.analysis_log, top_n)
        if st.session_state.highlights:
            st.markdown(f"**Top {len(st.session_state.highlights)} scenes**")
            for i, entry in enumerate(st.session_state.highlights, 1):
                with st.expander(f"Highlight #{i}  —  {entry['ts']}  ({entry['pos_sec']}s)",
                                 expanded=(i == 1)):
                    _h1, _h2 = st.columns([1, 2])
                    with _h1:
                        if entry.get("img_b64"):
                            st.image(base64.b64decode(entry["img_b64"]),
                                     caption=f"Frame {entry['frame_idx']}  |  {entry['pos_sec']}s",
                                     width='stretch')
                    with _h2:
                        st.markdown(entry["text"])
                        st.caption(f"Latency: {entry.get('latency', 0):.1f}s")

# ════════════════════════════════════════════
# タブ5: ANALYSIS LOG
# ════════════════════════════════════════════
with tab_log:
    st.markdown("<div class='panel-title'>Analysis Log</div>", unsafe_allow_html=True)
    if not st.session_state.analysis_log:
        st.info("No analysis results yet.")
    else:
        st.markdown(f"**{len(st.session_state.analysis_log)} frames analyzed**")
        _csv = "timestamp,frame_idx,pos_sec,latency,text\n"
        for e in st.session_state.analysis_log:
            _safe = e["text"].replace('"', '""').replace("\n", " ")
            _csv += f'"{e["ts"]}",{e["frame_idx"]},{e["pos_sec"]},{e.get("latency",0):.2f},"{_safe}"\n'
        st.download_button("Download CSV Log", data=_csv,
                           file_name=f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")
        show_imgs = st.checkbox("Show frame thumbnails", value=False)
        for entry in reversed(st.session_state.analysis_log):
            with st.expander(
                    f"{entry['ts']}  |  {entry['pos_sec']}s  |  Frame {entry['frame_idx']}",
                    expanded=False):
                if show_imgs and entry.get("img_b64"):
                    st.image(base64.b64decode(entry["img_b64"]), width='stretch')
                st.markdown(entry["text"])
                st.caption(f"Latency: {entry.get('latency', 0):.1f}s")

# ─────────────────────────────────────────────
# フッター
# ─────────────────────────────────────────────
st.markdown(
    "<div class='vit-footer'>"
    "VIDEO INTELLIGENCE TERMINAL"
    " &nbsp;|&nbsp; <span style='color:#005a80;'>NVIDIA NIM</span>"
    " + <span style='color:#005a80;'>HuggingFace</span>"
    " + <span style='color:#005a80;'>RT-DETR v1 / v2</span>"
    " &nbsp;|&nbsp; Parallel Detection Pipeline"
    "</div>",
    unsafe_allow_html=True,
)
