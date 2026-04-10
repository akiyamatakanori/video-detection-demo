"""
Video Intelligence Terminal
============================
リアルタイム物体検知 (RT-DETR / RT-DETRv2) と AI 日本語状況説明を並列処理するアプリ

主な機能:
- CPU / GPU 切り替えラジオボタン（UI左上）
- RT-DETR v1 / RT-DETRv2 を個別ON/OFFで選択してローカル物体検知（バウンディングボックス表示）
- Ollama VLM 推論 → GPU サーバー (192.168.11.111:11434) を使用
- NVIDIA NIM Vision API / HuggingFace 無料モデルによる日本語状況説明（並列処理）
- YouTube 動画 / ライブ配信 / ローカルフォルダ のファイル解析対応
- 社内プロキシ環境対応: SSL 検証を全 import より前に無効化
- 受付大画面向け 24/365 常時表示対応
"""

# ─────────────────────────────────────────────
# 【最優先】社内プロキシ SSL 対策
# ─────────────────────────────────────────────
import os, ssl

os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

_v1_cache = os.path.expanduser("~/.cache/huggingface/hub/models--PekingU--rtdetr_r50vd")
_v2_cache = os.path.expanduser("~/.cache/huggingface/hub/models--PekingU--rtdetr_v2_r50vd")
_RTDETR_CACHED = os.path.isdir(_v1_cache) and os.path.isdir(_v2_cache)

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"

ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests as _req_mod
_OrigSession = _req_mod.Session

class _NoVerifySession(_OrigSession):
    def request(self, method, url, **kwargs):
        kwargs["verify"] = False
        return super().request(method, url, **kwargs)

_req_mod.Session = _NoVerifySession

try:
    import huggingface_hub.utils._http as _hf_http
    if hasattr(_hf_http, "get_session"):
        _hf_http.get_session = lambda: _NoVerifySession()
except Exception:
    pass

# ─────────────────────────────────────────────
# 通常 import
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

# ── Ollama エンドポイント ──────────────────────────────────────────────────
# GPU サーバー: 192.168.11.111:11434
# ローカル CPU: localhost:11434
OLLAMA_GPU_URL  = "http://192.168.11.111:11434"
OLLAMA_CPU_URL  = "http://localhost:11434"

_ca_bundle = os.getenv("REQUESTS_CA_BUNDLE", "")
SSL_VERIFY  = _ca_bundle if _ca_bundle else False

Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# GPU / CPU デバイス判定ユーティリティ
# ─────────────────────────────────────────────
def get_compute_device(prefer_gpu: bool) -> str:
    """
    prefer_gpu=True  → CUDA があれば cuda、なければ cpu
    prefer_gpu=False → 強制的に cpu
    """
    if prefer_gpu and DETECTION_AVAILABLE:
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"

def get_ollama_url(prefer_gpu: bool) -> str:
    return OLLAMA_GPU_URL if prefer_gpu else OLLAMA_CPU_URL

# ─────────────────────────────────────────────
# モデル定義
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

# ローカル Ollama ビジョンモデル（GPUサーバーに搭載済みモデルを反映）
LOCAL_VISION_MODELS = {
    "LOCAL: Qwen2.5-VL 7B": {
        "id": "qwen2.5vl:7b",
        "desc": "完全ローカル推論 / 日本語高精度 / Ollama",
        "backend": "ollama",
        "display": "Qwen2.5-VL 7B [LOCAL / Ollama]",
    },
    "LOCAL: Qwen2.5-VL 32B": {
        "id": "qwen2.5vl:32b",
        "desc": "高精度ローカル推論 / Ollama",
        "backend": "ollama",
        "display": "Qwen2.5-VL 32B [LOCAL / Ollama]",
    },
    "LOCAL: Qwen3-VL 8B": {
        "id": "qwen3-vl:8b",
        "desc": "最新世代VLM / 軽量 / Ollama",
        "backend": "ollama",
        "display": "Qwen3-VL 8B [LOCAL / Ollama]",
    },
    "LOCAL: Qwen3-VL 32B": {
        "id": "qwen3-vl:32b",
        "desc": "最新世代VLM / 高精度 / Ollama",
        "backend": "ollama",
        "display": "Qwen3-VL 32B [LOCAL / Ollama]",
    },
    "LOCAL: Llama3.2-Vision 11B FP16": {
        "id": "llama3.2-vision:11b-instruct-fp16",
        "desc": "FP16高精度 / Meta / Ollama",
        "backend": "ollama",
        "display": "Llama3.2-Vision 11B FP16 [LOCAL / Ollama]",
    },
    "LOCAL: Llama3.2-Vision 90B": {
        "id": "llama3.2-vision:90b",
        "desc": "大規模ローカル推論 / Ollama",
        "backend": "ollama",
        "display": "Llama3.2-Vision 90B [LOCAL / Ollama]",
    },
}

ALL_VISION_MODELS = {**LOCAL_VISION_MODELS, **NIM_VISION_MODELS, **HF_VISION_MODELS}

TEXT_MODELS = {
    "Llama 3.3 70B":      {"id": "meta/llama-3.3-70b-instruct",               "desc": "要約・分析向け"},
    "Llama 3.1 70B":      {"id": "meta/llama-3.1-70b-instruct",               "desc": "バランス型"},
    "Nemotron Super 49B": {"id": "nvidia/llama-3.3-nemotron-super-49b-v1",    "desc": "推論特化・高精度"},
}

DETECTION_MODELS = {
    "v1": {
        "id":        "PekingU/rtdetr_r50vd",
        "label":     "RT-DETR",
        "color":     (0, 200, 230),
        "color_hex": "#00c8e6",
    },
    "v2": {
        "id":        "PekingU/rtdetr_v2_r50vd",
        "label":     "RT-DETRv2",
        "color":     (80, 210, 110),
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
1. 動画全体の概要 2. 主要な出来事・シーンの流れ（時系列）
3. 検出された主要な物体・人物・テキスト 4. 特筆すべき重要な瞬間やハイライト 5. 全体的な考察・結論

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

/* ── CPU/GPU 切り替えバナー ── */
.compute-banner {
    display: flex; align-items: center; gap: 14px;
    padding: 8px 16px; border-radius: 3px; margin-bottom: 10px;
    font-family: 'Orbitron', monospace; font-size: 0.72rem;
    letter-spacing: 0.12em; text-transform: uppercase;
}
.compute-banner.gpu {
    background: linear-gradient(90deg, #001a10 0%, #002d1a 100%);
    border: 1px solid #00e57a; border-left: 4px solid #00e57a;
}
.compute-banner.cpu {
    background: linear-gradient(90deg, #0a0d1a 0%, #121a2e 100%);
    border: 1px solid #4a7aaa; border-left: 4px solid #4a7aaa;
}
.compute-icon { font-size: 1.2rem; }
.compute-label-gpu { color: #00e57a; font-weight: 700; }
.compute-label-cpu { color: #5ab8d0; font-weight: 700; }
.compute-detail { color: #3a6a80; font-size: 0.62rem; margin-top: 2px; }
.compute-perf-badge {
    margin-left: auto; padding: 3px 10px; border-radius: 2px;
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.1em;
}
.compute-perf-badge.gpu { background: #00e57a22; color: #00e57a; border: 1px solid #00e57a44; }
.compute-perf-badge.cpu { background: #4a7aaa22; color: #5ab8d0; border: 1px solid #4a7aaa44; }

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
.status-val { color: #00b4d8; margin-left: 5px; }
.status-val.ok    { color: #00e5a0; }
.status-val.busy  { color: #ffd166; }
.status-val.error { color: #ef476f; }
.status-val.gpu   { color: #00e57a; font-weight: bold; }
.status-val.cpu   { color: #5ab8d0; }

.analysis-card {
    background: #020f1f; border: 1px solid #0a3a55;
    border-left: 3px solid #00b4d8; border-radius: 2px; padding: 10px 14px; margin: 6px 0;
}
.analysis-ts   { font-family: 'Share Tech Mono', monospace; font-size: 0.68rem; color: #00b4d8; }
.analysis-body { color: #b0cce0; font-size: 0.84rem; margin-top: 5px; line-height: 1.6; }

.detection-card {
    background: #020f1f; border: 1px solid #0a3a55; border-radius: 2px;
    padding: 6px 10px; margin: 3px 0;
    font-family: 'Share Tech Mono', monospace; font-size: 0.7rem;
}
.det-v1    { border-left: 3px solid #00c8e6; }
.det-v2    { border-left: 3px solid #50d26e; }
.det-label { color: #d0e8f0; }
.det-score { color: #ffd166; margin-left: 8px; }
.det-tag   { font-size: 0.58rem; color: #2a6070; margin-left: 6px; }

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

/* GPU ON/OFF ボタン */
.gpu-btn-wrap {
    display: flex; flex-direction: column; align-items: flex-end;
    gap: 6px; padding-top: 8px;
}
.gpu-btn-label {
    font-family: 'Orbitron', monospace; font-size: 0.6rem;
    color: #2a6a88; letter-spacing: 0.15em; text-transform: uppercase;
}
/* ON 状態（緑） */
button[data-testid="gpu_on_btn"] {
    background: linear-gradient(135deg, #003a20, #005a30) !important;
    border: 2px solid #00e57a !important; color: #00e57a !important;
    font-family: 'Orbitron', monospace !important; font-size: 0.78rem !important;
    font-weight: 900 !important; letter-spacing: 0.12em !important;
    border-radius: 3px !important; padding: 8px 18px !important;
    box-shadow: 0 0 12px rgba(0,229,122,0.3) !important;
}
/* OFF 状態（グレーブルー） */
button[data-testid="gpu_off_btn"] {
    background: #0a1a2e !important;
    border: 2px solid #1a4a6a !important; color: #2a7a9a !important;
    font-family: 'Orbitron', monospace !important; font-size: 0.78rem !important;
    font-weight: 700 !important; letter-spacing: 0.12em !important;
    border-radius: 3px !important; padding: 8px 18px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# セッション状態 初期化
# ─────────────────────────────────────────────
_defaults = {
    "processing":           False,
    "youtube_url":          "",
    "video_file":           None,
    "stream_url":           None,
    "is_live":              False,
    "stream_title":         "",
    "selected_model":       "LOCAL: Qwen2.5-VL 7B",
    "current_prompt":       PROMPT_PRESETS["General Detection"],
    "pending_prompt":       None,
    "analysis_log":         [],
    "summary_text":         "",
    "search_query":         "",
    "search_results":       [],
    "total_frames_analyzed": 0,
    "mode":                 "YouTube",
    "highlights":           [],
    "local_folder":         DEFAULT_FOLDER,
    "det_v1_enabled":       True,
    "det_v2_enabled":       True,
    "last_det_count":       {"v1": 0, "v2": 0},
    # ── 新規: GPU/CPU 切り替え ──
    "use_gpu":              True,
    # ── パフォーマンス計測 ──
    "perf_history":         [],   # {"mode": "GPU"/"CPU", "latency": float, "ts": str}
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────
# ヘッダー（CPU/GPU バナー含む）
# ─────────────────────────────────────────────
header_col, mode_col = st.columns([3, 1])

with header_col:
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

with mode_col:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='gpu-btn-label'>⚡ COMPUTE MODE</div>", unsafe_allow_html=True)

    # ── GPU ON/OFF トグルボタン（UI左上部に配置） ──
    if st.session_state.use_gpu:
        # GPU ON 状態 → ボタンを押すと CPU へ
        if st.button(
            "⚡ GPU  ON",
            key="gpu_toggle_btn",
            help="クリックすると CPU モードへ切り替え",
            use_container_width=True,
        ):
            st.session_state.use_gpu = False
            st.cache_resource.clear()
            st.rerun()
        st.markdown(
            "<div style='font-family:Share Tech Mono,monospace;font-size:0.6rem;"
            "color:#00e57a;text-align:center;margin-top:2px;'>H100 ACTIVE</div>",
            unsafe_allow_html=True,
        )
    else:
        # GPU OFF 状態 → ボタンを押すと GPU へ
        if st.button(
            "💻 GPU  OFF",
            key="gpu_toggle_btn",
            help="クリックすると GPU モードへ切り替え",
            use_container_width=True,
        ):
            st.session_state.use_gpu = True
            st.cache_resource.clear()
            st.rerun()
        st.markdown(
            "<div style='font-family:Share Tech Mono,monospace;font-size:0.6rem;"
            "color:#2a7a9a;text-align:center;margin-top:2px;'>CPU MODE</div>",
            unsafe_allow_html=True,
        )

# ── CPU/GPU バナー表示 ──
_mode_str = "GPU" if st.session_state.use_gpu else "CPU"
_ollama_endpoint = get_ollama_url(st.session_state.use_gpu)
_rt_device = get_compute_device(st.session_state.use_gpu)

if st.session_state.use_gpu:
    st.markdown(f"""
    <div class="compute-banner gpu">
      <span class="compute-icon">⚡</span>
      <div>
        <div class="compute-label-gpu">GPU MODE — NVIDIA H100 ACTIVE</div>
        <div class="compute-detail">
          RT-DETR Device: <b>CUDA</b> &nbsp;|&nbsp;
          Ollama: <b>{OLLAMA_GPU_URL}</b> &nbsp;|&nbsp;
          VLM: GPU Server (192.168.11.111)
        </div>
      </div>
      <span class="compute-perf-badge gpu">HIGH PERFORMANCE</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="compute-banner cpu">
      <span class="compute-icon">💻</span>
      <div>
        <div class="compute-label-cpu">CPU MODE — LOCAL COMPUTE</div>
        <div class="compute-detail">
          RT-DETR Device: <b>CPU</b> &nbsp;|&nbsp;
          Ollama: <b>{OLLAMA_CPU_URL}</b> &nbsp;|&nbsp;
          VLM: Localhost
        </div>
      </div>
      <span class="compute-perf-badge cpu">STANDARD MODE</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# APIキー確認
# ─────────────────────────────────────────────
if not NVIDIA_API_KEY and not HF_TOKEN:
    st.warning("⚠️ NVIDIA_API_KEY / HF_TOKEN が未設定です。NIM/HF モデルは使用不可ですが、ローカル Ollama は利用可能です。")

# ─────────────────────────────────────────────
# RT-DETR モデルキャッシュロード（GPU/CPU 対応）
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rtdetr_v1(use_gpu: bool):
    if not DETECTION_AVAILABLE:
        return None, None, None, "torch/transformers not installed"
    if not _RTDETR_CACHED:
        return None, None, None, "モデル未キャッシュ。download_models.py を実行してください"
    try:
        device = get_compute_device(use_gpu)
        cfg  = DETECTION_MODELS["v1"]
        proc  = AutoImageProcessor.from_pretrained(cfg["id"], trust_remote_code=False, local_files_only=True)
        model = AutoModelForObjectDetection.from_pretrained(cfg["id"], trust_remote_code=False, local_files_only=True)
        model.to(device)
        model.eval()
        return proc, model, device, None
    except Exception as exc:
        return None, None, None, str(exc)

@st.cache_resource(show_spinner=False)
def load_rtdetr_v2(use_gpu: bool):
    if not DETECTION_AVAILABLE:
        return None, None, None, "torch/transformers not installed"
    if not _RTDETR_CACHED:
        return None, None, None, "モデル未キャッシュ。download_models.py を実行してください"
    try:
        device = get_compute_device(use_gpu)
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
    frame = frame_rgb.copy()
    for ver, dets in detections.items():
        cfg   = DETECTION_MODELS.get(ver, {})
        color = cfg.get("color", (200, 200, 200))
        for det in (dets or []):
            x1, y1, x2, y2 = det["box"]
            score = det["score"]
            lbl   = det["label"]
            text  = f"{lbl} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs, th = 0.44, 1
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
                  threshold: float = 0.45, use_gpu: bool = True) -> dict:
    result = {"v1": [], "v2": [], "error": None}

    if use_v1:
        proc, model, device, err = load_rtdetr_v1(use_gpu)
        if err:
            result["error"] = f"v1: {err}"
        elif proc and model:
            try:
                pil = Image.fromarray(frame_rgb)
                inp = proc(images=pil, return_tensors="pt")
                inp = {k: v.to(device) for k, v in inp.items()}
                with torch.no_grad():
                    out = model(**inp)
                ts    = torch.tensor([[pil.height, pil.width]])
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
        proc, model, device, err = load_rtdetr_v2(use_gpu)
        if err:
            result["error"] = (result["error"] or "") + f" | v2: {err}"
        elif proc and model:
            try:
                pil = Image.fromarray(frame_rgb)
                inp = proc(images=pil, return_tensors="pt")
                inp = {k: v.to(device) for k, v in inp.items()}
                with torch.no_grad():
                    out = model(**inp)
                ts    = torch.tensor([[pil.height, pil.width]])
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

def _ollama_analyze(frame_rgb, prompt, model_id, resize_pct, max_tokens,
                    use_gpu: bool = True) -> dict:
    """
    Ollama ローカル推論。
    use_gpu=True  → GPUサーバー (192.168.11.111:11434)
    use_gpu=False → ローカル    (localhost:11434)
    """
    if resize_pct < 100:
        frame_rgb = resize_frame(frame_rgb, resize_pct)
    img_b64 = frame_to_base64(frame_rgb)
    t0      = time.time()
    base_url = get_ollama_url(use_gpu)

    try:
        payload = {
            "model": model_id,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "num_predict": min(max_tokens, 600 if use_gpu else 400),
                "temperature": 0.2,
            },
        }
        resp = requests.post(
            f"{base_url}/api/generate",
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
                "device":  "GPU" if use_gpu else "CPU",
            }
        elif resp.status_code == 404:
            return {
                "ok":      False,
                "text":    f"Ollamaモデル未インストール。実行: ollama pull {model_id}",
                "latency": latency,
            }
        return {
            "ok":      False,
            "text":    f"Ollama HTTP {resp.status_code}: {resp.text[:200]}",
            "latency": latency,
        }
    except requests.exceptions.ConnectionError:
        target = f"GPUサーバー ({OLLAMA_GPU_URL})" if use_gpu else f"ローカル ({OLLAMA_CPU_URL})"
        return {
            "ok":      False,
            "text":    f"Ollamaに接続できません: {target}",
            "latency": time.time() - t0,
        }
    except Exception as exc:
        return {"ok": False, "text": f"Ollamaエラー: {exc}", "latency": time.time() - t0}

def vlm_analyze(frame_rgb, prompt, model_name, resize_pct=100,
                max_tokens=600, use_gpu=True) -> dict:
    cfg = ALL_VISION_MODELS.get(model_name)
    if not cfg:
        return {"ok": False, "text": f"Unknown model: {model_name}", "latency": 0.0}
    if cfg["backend"] == "ollama":
        return _ollama_analyze(frame_rgb, prompt, cfg["id"],
                               resize_pct, max_tokens, use_gpu)
    if cfg["backend"] == "nim":
        return _nim_analyze(frame_rgb, prompt, cfg["id"], resize_pct, max_tokens)
    return _hf_analyze(frame_rgb, prompt, cfg["id"], resize_pct, max_tokens)

def _nim_analyze(frame_rgb, prompt, model_id, resize_pct, max_tokens) -> dict:
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
            return {"ok": True,  "text": resp.json()["choices"][0]["message"]["content"],
                    "latency": latency, "img_b64": img_b64, "device": "NIM API"}
        return {"ok": False, "text": f"NIM {resp.status_code}: {resp.text[:300]}", "latency": latency}
    except requests.Timeout:
        return {"ok": False, "text": "Timeout (90s)", "latency": 90.0}
    except Exception as exc:
        return {"ok": False, "text": f"NIM error: {exc}", "latency": 0.0}

def _hf_analyze(frame_rgb, prompt, model_id, resize_pct, max_tokens) -> dict:
    if not HF_HUB_AVAILABLE:
        return {"ok": False, "text": "huggingface_hub not installed", "latency": 0.0}
    if not HF_TOKEN:
        return {"ok": False, "text": "HF_TOKEN not set in .env", "latency": 0.0}
    if resize_pct < 100:
        frame_rgb = resize_frame(frame_rgb, resize_pct)
    img_b64 = frame_to_base64(frame_rgb)
    t0 = time.time()
    try:
        client   = InferenceClient(token=HF_TOKEN)
        response = client.chat_completion(
            model=model_id,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text",      "text": prompt},
            ]}],
            max_tokens=max_tokens,
        )
        return {"ok": True, "text": response.choices[0].message.content,
                "latency": time.time() - t0, "img_b64": img_b64, "device": "HF API"}
    except Exception as exc:
        return {"ok": False, "text": f"HF error: {exc}", "latency": time.time() - t0}

def nim_text_summarize(analysis_log: list, model_name: str = "Llama 3.3 70B") -> dict:
    model_id = TEXT_MODELS.get(model_name, TEXT_MODELS["Llama 3.3 70B"])["id"]
    log_str  = "\n\n".join([f"[{e['ts']}] Frame {e['frame_idx']}: {e['text']}" for e in analysis_log])
    payload  = {
        "model":    model_id,
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
    import re
    from collections import Counter
    STOPWORDS = {
        "の","に","は","を","が","で","と","た","し","て","い","な","こ","れ","も","から",
        "です","ます","いる","ある","この","その","また","など","よう","これ","それ",
        "a","an","the","is","in","on","at","to","of","and","or","with","this","that",
        "are","was","were","be","been","has","have","it","its","for","as","by","from",
        "画像","写真","フレーム","場面","状況","内容","様子","見え","られ","なっ","おり",
    }
    counter = Counter()
    for entry in log:
        text     = entry.get("text", "")
        jp_words = re.findall(r'[一-龯ぁ-んァ-ン]{2,}', text)
        en_words = re.findall(r'[A-Za-z]{3,}', text)
        for w in jp_words + en_words:
            wl = w.lower()
            if wl not in STOPWORDS and len(wl) >= 2:
                counter[w] += 1
    return [w for w, c in counter.most_common(top_n) if c >= 1]

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

def make_status_bar(status, latency, frames, model_full,
                    det_v1, det_v2, compute_mode) -> str:
    sc       = "ok" if status in ("READY", "OK") else ("busy" if status == "ANALYZING" else "")
    mode_cls = "gpu" if compute_mode == "GPU" else "cpu"
    return (
        f"<div class='status-bar'>"
        f"<span class='status-item'>STATUS <span class='status-val {sc}'>{status}</span></span>"
        f"<span class='status-item'>COMPUTE <span class='status-val {mode_cls}'>{compute_mode}</span></span>"
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
            st.success(f"NVIDIA API Key set `...{NVIDIA_API_KEY[-8:]}`")
        else:
            st.warning("NVIDIA_API_KEY not set")
        if HF_TOKEN:
            st.success(f"HuggingFace Token set `...{HF_TOKEN[-8:]}`")
        else:
            st.info("HF_TOKEN not set")
        ssl_mode = "CA Bundle: " + _ca_bundle if _ca_bundle else "SSL verify: OFF (proxy mode)"
        st.caption(ssl_mode)

        # Ollama 接続テスト（現在のモードに応じて接続先を変える）
        if st.button("Test Ollama Connection", key="test_ollama"):
            with st.spinner("Testing Ollama..."):
                try:
                    r = requests.get(f"{get_ollama_url(st.session_state.use_gpu)}/api/tags",
                                     timeout=10, verify=False)
                    if r.status_code == 200:
                        models_list = r.json().get("models", [])
                        st.success(f"OK — Ollama 接続成功 ({len(models_list)} models)")
                    else:
                        st.error(f"HTTP {r.status_code}")
                except Exception as exc:
                    st.error(str(exc))

        if st.button("Test NIM Connection", key="test_nim"):
            with st.spinner("Testing NIM..."):
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

    # ── RT-DETR 設定 ──────────────────────────────────────────
    st.markdown("<div class='section-label'>Object Detection</div>", unsafe_allow_html=True)

    _device_label = f"CUDA ({_rt_device})" if st.session_state.use_gpu and _rt_device == "cuda" else f"CPU"
    st.caption(f"Device: **{_device_label}**")

    det_v1_col, det_v2_col = st.columns(2)
    with det_v1_col:
        st.session_state.det_v1_enabled = st.toggle(
            "RT-DETR v1", value=st.session_state.det_v1_enabled, key="tog_v1")
    with det_v2_col:
        st.session_state.det_v2_enabled = st.toggle(
            "RT-DETRv2",  value=st.session_state.det_v2_enabled, key="tog_v2")

    det_threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.45, 0.05)

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

        if st.button("Download Video", type="primary"):
            if st.session_state.youtube_url:
                with st.spinner("Downloading..."):
                    path, title = download_youtube(st.session_state.youtube_url)
                    if path:
                        st.session_state.video_file  = path
                        st.session_state.stream_title = title
                        st.success(f"Downloaded: {title}")
                    else:
                        st.error(f"Error: {title}")

    elif st.session_state.mode == "Live":
        live_url = st.text_input("Live Stream URL", value=st.session_state.youtube_url,
                                 placeholder="https://youtu.be/live_xxxx")
        if live_url != st.session_state.youtube_url:
            st.session_state.youtube_url = live_url
            st.session_state.stream_url  = None

        if st.button("Connect Live", type="primary"):
            if st.session_state.youtube_url:
                with st.spinner("Connecting..."):
                    url, title, is_live, err = get_stream_url(st.session_state.youtube_url)
                    if url:
                        st.session_state.stream_url   = url
                        st.session_state.stream_title  = title
                        st.session_state.is_live       = is_live
                        st.success(f"{'LIVE' if is_live else 'VOD'}: {title}")
                    else:
                        st.error(f"Error: {err}")

    else:  # Local
        folder = st.text_input("Video Folder", value=st.session_state.local_folder)
        if folder != st.session_state.local_folder:
            st.session_state.local_folder = folder
        files = get_video_files(folder)
        if files:
            sel = st.selectbox("Select Video", [f.name for f in files])
            if sel:
                st.session_state.video_file = str(Path(folder) / sel)
        else:
            st.info("No video files found")

    st.divider()

    # VLM モデル選択
    st.markdown("<div class='section-label'>VLM Model</div>", unsafe_allow_html=True)
    _ollama_url_label = f"Ollama → {'GPU Server' if st.session_state.use_gpu else 'Localhost'}"
    st.caption(_ollama_url_label)

    model_options = list(ALL_VISION_MODELS.keys())
    cur_idx       = model_options.index(st.session_state.selected_model) \
                    if st.session_state.selected_model in model_options else 0
    sel_model     = st.selectbox("VLM Model", model_options,
                                 index=cur_idx, label_visibility="collapsed")
    if sel_model != st.session_state.selected_model:
        st.session_state.selected_model = sel_model

    st.divider()

    # Analysis Settings
    st.markdown("<div class='section-label'>Analysis Settings</div>", unsafe_allow_html=True)
    vlm_interval  = st.slider("VLM Interval (s)",  1, 30, 10, 1)
    max_tokens    = st.slider("Max Tokens",        100, 1000, 600 if st.session_state.use_gpu else 300, 50)
    resize_pct    = st.slider("Image Resize (%)",   20, 100,  80 if st.session_state.use_gpu else 60,  10)

    st.divider()

    # Generation Parameters
    st.markdown("<div class='section-label'>Generation Parameters</div>", unsafe_allow_html=True)
    top_k = st.slider("Top-k", 1, 100, 40, 5)

    st.divider()

    # Prompt Preset
    st.markdown("<div class='section-label'>Prompt Preset</div>", unsafe_allow_html=True)
    preset_name = st.selectbox("Preset", list(PROMPT_PRESETS.keys()),
                               label_visibility="collapsed")
    if st.button("Apply Preset"):
        st.session_state.pending_prompt = PROMPT_PRESETS[preset_name]

    custom_prompt = st.text_area("Custom Prompt", value=st.session_state.current_prompt,
                                 height=120, key="prompt_input")
    if custom_prompt != st.session_state.current_prompt:
        st.session_state.current_prompt = custom_prompt
    if st.session_state.pending_prompt:
        st.session_state.current_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        st.rerun()

# ─────────────────────────────────────────────
# ステータスバー
# ─────────────────────────────────────────────
_compute_mode_str = "GPU" if st.session_state.use_gpu else "CPU"
_latest_latency   = "—"
if st.session_state.perf_history:
    _latest_latency = f"{st.session_state.perf_history[-1]['latency']:.2f}s"

status_ph = st.empty()
status_ph.markdown(
    make_status_bar(
        "READY" if not st.session_state.processing else "ANALYZING",
        _latest_latency,
        st.session_state.total_frames_analyzed,
        ALL_VISION_MODELS.get(st.session_state.selected_model, {}).get("display", "—"),
        "ON" if st.session_state.det_v1_enabled else "OFF",
        "ON" if st.session_state.det_v2_enabled else "OFF",
        _compute_mode_str,
    ),
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# メインタブ
# ─────────────────────────────────────────────
tab_live, tab_search, tab_summary, tab_highlights, tab_log, tab_perf = st.tabs([
    "LIVE DETECTION", "VIDEO SEARCH", "SUMMARIZATION", "HIGHLIGHTS", "ANALYSIS LOG", "PERFORMANCE"
])

# ────────────────────────────────────────────────────────────
# TAB: LIVE DETECTION
# ────────────────────────────────────────────────────────────
with tab_live:
    ctrl_col, info_col = st.columns([1, 2])

    with ctrl_col:
        st.markdown("<div class='panel-title'>Controls</div>", unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            start_btn = st.button("▶ START", type="primary",
                                  disabled=st.session_state.processing)
        with btn_col2:
            stop_btn = st.button("■ STOP",
                                 disabled=not st.session_state.processing)

        if stop_btn:
            st.session_state.processing = False

        # 入力ソースチェック
        _source_ready = False
        if st.session_state.mode == "YouTube" and st.session_state.video_file:
            st.info(f"📹 {Path(st.session_state.video_file).name}")
            _source_ready = True
        elif st.session_state.mode == "Live" and st.session_state.stream_url:
            st.info(f"📡 LIVE: {st.session_state.stream_title[:30]}")
            _source_ready = True
        elif st.session_state.mode == "Local" and st.session_state.video_file:
            st.info(f"📁 {Path(st.session_state.video_file).name}")
            _source_ready = True
        else:
            st.warning("⚠️ Input source not ready")

        if st.button("🗑 Clear Log"):
            st.session_state.analysis_log         = []
            st.session_state.total_frames_analyzed = 0
            st.session_state.highlights            = []
            st.session_state.perf_history          = []
            st.rerun()

    with info_col:
        st.markdown("<div class='panel-title'>Live Feed</div>", unsafe_allow_html=True)
        frame_ph   = st.empty()
        det_info_ph = st.empty()

    analysis_ph = st.empty()

    # ── メイン処理ループ ──────────────────────────────────────
    if start_btn and _source_ready and not st.session_state.processing:
        st.session_state.processing = True

        video_source = (st.session_state.stream_url
                        if st.session_state.mode == "Live"
                        else st.session_state.video_file)
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            st.error("Cannot open video source")
            st.session_state.processing = False
        else:
            last_vlm_time = 0.0
            frame_idx     = 0
            _use_gpu      = st.session_state.use_gpu

            while st.session_state.processing:
                ret, frame = cap.read()
                if not ret:
                    if st.session_state.mode != "Live":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_idx += 1

                # RT-DETR 物体検知
                det_result = run_detection(
                    frame_rgb,
                    use_v1=st.session_state.det_v1_enabled,
                    use_v2=st.session_state.det_v2_enabled,
                    threshold=det_threshold,
                    use_gpu=_use_gpu,
                )

                st.session_state.last_det_count["v1"] = len(det_result.get("v1", []))
                st.session_state.last_det_count["v2"] = len(det_result.get("v2", []))

                # ボックス描画
                annotated = draw_detection_boxes(frame_rgb, {
                    k: v for k, v in det_result.items() if k in ("v1", "v2")
                })
                frame_ph.image(annotated, channels="RGB", use_container_width=True)

                # 検知結果表示
                det_html = ""
                for ver in ("v1", "v2"):
                    dets = det_result.get(ver, [])
                    if dets:
                        cfg = DETECTION_MODELS[ver]
                        det_html += (f"<div class='det-model-header' style='color:{cfg['color_hex']}'>"
                                     f"{cfg['label']} — {len(dets)} objects</div>")
                        for d in dets[:8]:
                            det_html += (f"<div class='detection-card det-{ver}'>"
                                         f"<span class='det-label'>{d['label']}</span>"
                                         f"<span class='det-score'>{d['score']:.2f}</span>"
                                         f"<span class='det-tag'>[{ver.upper()}]</span>"
                                         f"</div>")
                if det_html:
                    det_info_ph.markdown(det_html, unsafe_allow_html=True)

                # VLM 分析（インターバル制御）
                now = time.time()
                if now - last_vlm_time >= vlm_interval:
                    last_vlm_time = now
                    ts_str = datetime.now().strftime("%H:%M:%S")

                    status_ph.markdown(
                        make_status_bar("ANALYZING", "...", st.session_state.total_frames_analyzed,
                                        ALL_VISION_MODELS.get(st.session_state.selected_model, {}).get("display", "—"),
                                        "ON" if st.session_state.det_v1_enabled else "OFF",
                                        "ON" if st.session_state.det_v2_enabled else "OFF",
                                        _compute_mode_str),
                        unsafe_allow_html=True,
                    )

                    result = vlm_analyze(
                        frame_rgb,
                        st.session_state.current_prompt,
                        st.session_state.selected_model,
                        resize_pct=resize_pct,
                        max_tokens=max_tokens,
                        use_gpu=_use_gpu,
                    )

                    lat_str = f"{result['latency']:.2f}s"

                    # パフォーマンス履歴記録
                    st.session_state.perf_history.append({
                        "mode":    "GPU" if _use_gpu else "CPU",
                        "latency": result["latency"],
                        "ts":      ts_str,
                        "model":   st.session_state.selected_model,
                    })
                    # 直近100件のみ保持
                    if len(st.session_state.perf_history) > 100:
                        st.session_state.perf_history = st.session_state.perf_history[-100:]

                    if result["ok"]:
                        entry = {
                            "ts":        ts_str,
                            "frame_idx": frame_idx,
                            "text":      result["text"],
                            "img_b64":   result.get("img_b64", ""),
                            "latency":   result["latency"],
                            "model":     st.session_state.selected_model,
                            "device":    result.get("device", _compute_mode_str),
                        }
                        st.session_state.analysis_log.append(entry)
                        st.session_state.total_frames_analyzed += 1
                        st.session_state.highlights = detect_highlights(st.session_state.analysis_log)

                        analysis_ph.markdown(
                            f"<div class='analysis-card'>"
                            f"<div class='analysis-ts'>[{ts_str}] Frame {frame_idx} "
                            f"| {lat_str} | {result.get('device', _compute_mode_str)}</div>"
                            f"<div class='analysis-body'>{result['text']}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        analysis_ph.error(result["text"])

                    status_ph.markdown(
                        make_status_bar("OK", lat_str, st.session_state.total_frames_analyzed,
                                        ALL_VISION_MODELS.get(st.session_state.selected_model, {}).get("display", "—"),
                                        "ON" if st.session_state.det_v1_enabled else "OFF",
                                        "ON" if st.session_state.det_v2_enabled else "OFF",
                                        _compute_mode_str),
                        unsafe_allow_html=True,
                    )

                time.sleep(0.03)  # ~30fps 表示レート

            cap.release()
            st.session_state.processing = False

# ────────────────────────────────────────────────────────────
# TAB: VIDEO SEARCH
# ────────────────────────────────────────────────────────────
with tab_search:
    st.markdown("<div class='panel-title'>Search Analysis Log</div>", unsafe_allow_html=True)

    tags = extract_tags(st.session_state.analysis_log)
    if tags:
        st.markdown("**Auto Tags:**")
        tag_cols = st.columns(min(len(tags), 8))
        for i, tag in enumerate(tags[:8]):
            with tag_cols[i % 8]:
                if st.button(tag, key=f"tag_{i}"):
                    st.session_state.search_query   = tag
                    st.session_state.search_results = search_analysis_log(
                        st.session_state.analysis_log, tag)

    query = st.text_input("Search Query", value=st.session_state.search_query,
                          placeholder="キーワードを入力 (スペース区切りでAND検索)")
    if query != st.session_state.search_query:
        st.session_state.search_query   = query
        st.session_state.search_results = search_analysis_log(
            st.session_state.analysis_log, query)

    if st.session_state.search_results:
        st.info(f"{len(st.session_state.search_results)} results found")
        for entry in st.session_state.search_results:
            st.markdown(
                f"<div class='analysis-card'>"
                f"<div class='analysis-ts'>[{entry['ts']}] Frame {entry['frame_idx']}"
                f" | {entry.get('device', '—')}</div>"
                f"<div class='analysis-body'>"
                f"{highlight_text(entry['text'], st.session_state.search_query)}"
                f"</div></div>",
                unsafe_allow_html=True,
            )
    elif st.session_state.search_query:
        st.info("No results found")

# ────────────────────────────────────────────────────────────
# TAB: SUMMARIZATION
# ────────────────────────────────────────────────────────────
with tab_summary:
    st.markdown("<div class='panel-title'>Video Summarization</div>", unsafe_allow_html=True)

    sum_model = st.selectbox("Text Model", list(TEXT_MODELS.keys()))
    if st.button("Generate Summary", type="primary"):
        if st.session_state.analysis_log:
            with st.spinner("Generating summary..."):
                result = nim_text_summarize(st.session_state.analysis_log, sum_model)
                if result["ok"]:
                    st.session_state.summary_text = result["text"]
                else:
                    st.error(result["text"])
        else:
            st.warning("No analysis log to summarize")

    if st.session_state.summary_text:
        st.markdown(
            f"<div class='analysis-card'>"
            f"<div class='analysis-body'>{st.session_state.summary_text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.download_button("📥 Download Summary",
                           data=st.session_state.summary_text,
                           file_name="summary.txt", mime="text/plain")

# ────────────────────────────────────────────────────────────
# TAB: HIGHLIGHTS
# ────────────────────────────────────────────────────────────
with tab_highlights:
    st.markdown("<div class='panel-title'>Auto Highlights</div>", unsafe_allow_html=True)

    if st.session_state.highlights:
        for i, entry in enumerate(st.session_state.highlights):
            cols = st.columns([1, 2])
            with cols[0]:
                if entry.get("img_b64"):
                    try:
                        img_data = base64.b64decode(entry["img_b64"])
                        img      = Image.open(io.BytesIO(img_data))
                        st.image(img, use_container_width=True)
                    except Exception:
                        st.info("No image")
            with cols[1]:
                st.markdown(
                    f"<div class='analysis-card'>"
                    f"<div class='analysis-ts'>#{i+1} [{entry['ts']}] Frame {entry['frame_idx']}"
                    f" | {entry.get('device', '—')}</div>"
                    f"<div class='analysis-body'>{entry['text'][:300]}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No highlights yet. Start analysis to generate highlights.")

# ────────────────────────────────────────────────────────────
# TAB: ANALYSIS LOG
# ────────────────────────────────────────────────────────────
with tab_log:
    st.markdown("<div class='panel-title'>Full Analysis Log</div>", unsafe_allow_html=True)

    if st.session_state.analysis_log:
        st.info(f"Total: {len(st.session_state.analysis_log)} entries")

        if st.button("📥 Export JSON"):
            json_str = json.dumps(
                [{k: v for k, v in e.items() if k != "img_b64"}
                 for e in st.session_state.analysis_log],
                ensure_ascii=False, indent=2
            )
            st.download_button("Download JSON", data=json_str,
                               file_name="analysis_log.json", mime="application/json")

        for entry in reversed(st.session_state.analysis_log[-50:]):
            st.markdown(
                f"<div class='analysis-card'>"
                f"<div class='analysis-ts'>[{entry['ts']}] Frame {entry['frame_idx']}"
                f" | {entry.get('latency', 0):.2f}s | {entry.get('device', '—')}</div>"
                f"<div class='analysis-body'>{entry['text']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No analysis log yet.")

# ────────────────────────────────────────────────────────────
# TAB: PERFORMANCE（CPU vs GPU 比較）
# ────────────────────────────────────────────────────────────
with tab_perf:
    st.markdown("<div class='panel-title'>CPU vs GPU Performance Comparison</div>", unsafe_allow_html=True)

    if st.session_state.perf_history:
        gpu_records = [r for r in st.session_state.perf_history if r["mode"] == "GPU"]
        cpu_records = [r for r in st.session_state.perf_history if r["mode"] == "CPU"]

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            if gpu_records:
                avg_gpu = sum(r["latency"] for r in gpu_records) / len(gpu_records)
                st.metric("GPU Avg Latency", f"{avg_gpu:.2f}s", delta=None)
            else:
                st.metric("GPU Avg Latency", "—")
        with m_col2:
            if cpu_records:
                avg_cpu = sum(r["latency"] for r in cpu_records) / len(cpu_records)
                st.metric("CPU Avg Latency", f"{avg_cpu:.2f}s")
            else:
                st.metric("CPU Avg Latency", "—")
        with m_col3:
            if gpu_records and cpu_records:
                speedup = avg_cpu / avg_gpu if avg_gpu > 0 else 0
                st.metric("GPU Speedup", f"{speedup:.1f}×", delta=f"+{speedup-1:.1f}×" if speedup > 1 else None)
            else:
                st.metric("GPU Speedup", "—")
        with m_col4:
            st.metric("Total Samples", len(st.session_state.perf_history))

        # レイテンシ推移グラフ
        st.markdown("#### Latency Timeline")
        import json as _json
        chart_data_gpu = [[r["ts"], r["latency"]] for r in gpu_records]
        chart_data_cpu = [[r["ts"], r["latency"]] for r in cpu_records]

        # Streamlit ネイティブチャート用データ
        import pandas as pd
        if gpu_records or cpu_records:
            all_records = []
            for r in st.session_state.perf_history:
                all_records.append({
                    "index": st.session_state.perf_history.index(r),
                    "latency": r["latency"],
                    "mode": r["mode"],
                })
            df = pd.DataFrame(all_records)
            gpu_df = df[df["mode"] == "GPU"][["index", "latency"]].rename(columns={"latency": "GPU Latency (s)"})
            cpu_df = df[df["mode"] == "CPU"][["index", "latency"]].rename(columns={"latency": "CPU Latency (s)"})

            if not gpu_df.empty and not cpu_df.empty:
                merged = pd.merge(gpu_df, cpu_df, on="index", how="outer").set_index("index")
                st.line_chart(merged)
            elif not gpu_df.empty:
                st.line_chart(gpu_df.set_index("index"))
            elif not cpu_df.empty:
                st.line_chart(cpu_df.set_index("index"))

        # 詳細テーブル
        with st.expander("Raw Performance Data"):
            st.dataframe(
                pd.DataFrame(st.session_state.perf_history)[["ts", "mode", "latency", "model"]],
                use_container_width=True,
            )

        if st.button("🗑 Clear Performance Data"):
            st.session_state.perf_history = []
            st.rerun()
    else:
        st.info("パフォーマンスデータがありません。GPU/CPU モードを切り替えながら分析を実行してください。")
        st.markdown("""
        **使い方:**
        1. 右上の **🟢 GPU** モードで分析を開始
        2. 数フレーム分析したら **■ STOP**
        3. **🔵 CPU** モードに切り替えて再度 **▶ START**
        4. このタブでレイテンシを比較
        """)

# ─────────────────────────────────────────────
# フッター
# ─────────────────────────────────────────────
st.markdown(
    "<div class='vit-footer'>"
    "VIDEO INTELLIGENCE TERMINAL &nbsp;|&nbsp; RT-DETR v1+v2 + Ollama VLM "
    "&nbsp;|&nbsp; GPU: 192.168.11.111 &nbsp;|&nbsp; CISCO AI LAB"
    "</div>",
    unsafe_allow_html=True,
)
