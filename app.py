"""
Video Intelligence Terminal
============================
アーキテクチャ（GPUサーバー 192.168.11.111 に一本化）:
  - RT-DETR   : GPU ON → CUDA (H100)  / GPU OFF → CPU
  - Ollama VLM: GPU ON → 大型モデル   / GPU OFF → 軽量モデル
  - Ollama URL: localhost:11434 (同一サーバー)
"""

# ─────────────────────────────────────────────
# 【最優先】SSL 対策
# ─────────────────────────────────────────────
import os, ssl

os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
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
import base64, io, requests, time, json, re
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

# ─────────────────────────────────────────────
# 設定定数
# ─────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
HF_TOKEN       = os.getenv("HF_TOKEN", "")
NIM_BASE_URL   = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Ollama: GPUサーバー (192.168.11.111) に接続
OLLAMA_URL = "http://192.168.11.111:11434"

_ca_bundle = os.getenv("REQUESTS_CA_BUNDLE", "")
SSL_VERIFY = _ca_bundle if _ca_bundle else False

# ─────────────────────────────────────────────
# ディレクトリ設定
# .env で DATA_DIR を指定すれば任意パスに変更可能
# 例: DATA_DIR=/mnt/storage/vit-data
# ─────────────────────────────────────────────
_BASE        = Path(__file__).parent
DATA_DIR     = Path(os.getenv("DATA_DIR",     str(_BASE / "data")))
DOWNLOAD_DIR = Path(os.getenv("DOWNLOAD_DIR", str(DATA_DIR / "downloads")))
LOG_DIR      = Path(os.getenv("LOG_DIR",      str(DATA_DIR / "logs")))
SUMMARY_DIR  = Path(os.getenv("SUMMARY_DIR",  str(DATA_DIR / "summaries")))
EXPORT_DIR   = Path(os.getenv("EXPORT_DIR",   str(DATA_DIR / "exports")))

# Local File のデフォルト参照フォルダ
# .env で DEFAULT_VIDEO_FOLDER を指定すれば変更可能
DEFAULT_FOLDER = os.getenv("DEFAULT_VIDEO_FOLDER", "/home/ailab/videos")

# フォルダを全て自動生成
for _d in [DOWNLOAD_DIR, LOG_DIR, SUMMARY_DIR, EXPORT_DIR, Path(DEFAULT_FOLDER)]:
    _d.mkdir(parents=True, exist_ok=True)

# RT-DETR モデルキャッシュ確認
_v1_cache     = os.path.expanduser("~/.cache/huggingface/hub/models--PekingU--rtdetr_r50vd")
_v2_cache     = os.path.expanduser("~/.cache/huggingface/hub/models--PekingU--rtdetr_v2_r50vd")
_RTDETR_CACHED = os.path.isdir(_v1_cache) and os.path.isdir(_v2_cache)

# GPU/CPU デフォルトモデル
GPU_DEFAULT_MODEL = "Qwen2.5-VL 32B (GPU Server)"
CPU_DEFAULT_MODEL = "Qwen2.5-VL 7B (GPU Server)"

# ─────────────────────────────────────────────
# モデル定義（命名規則: "モデル名 (バックエンド)"）
# ─────────────────────────────────────────────

# ── GPU サーバー Ollama モデル ─────────────────────────────
LOCAL_VISION_MODELS = {
    # Qwen2.5-VL シリーズ
    "Qwen2.5-VL 7B (GPU Server)":            {"id": "qwen2.5vl:7b",                       "desc": "軽量・高速",          "backend": "ollama"},
    "Qwen2.5-VL 32B (GPU Server)":           {"id": "qwen2.5vl:32b",                      "desc": "高精度",              "backend": "ollama"},
    # Qwen3-VL シリーズ
    "Qwen3-VL 8B (GPU Server)":              {"id": "qwen3-vl:8b",                         "desc": "最新世代・軽量",      "backend": "ollama"},
    "Qwen3-VL 32B (GPU Server)":             {"id": "qwen3-vl:32b",                        "desc": "最新世代・高精度",    "backend": "ollama"},
    # Llama3.2-Vision シリーズ
    "Llama3.2-Vision 11B FP16 (GPU Server)": {"id": "llama3.2-vision:11b-instruct-fp16",  "desc": "FP16高精度",          "backend": "ollama"},
    "Llama3.2-Vision 11B (GPU Server)":      {"id": "x/llama3.2-vision:latest",            "desc": "標準版",              "backend": "ollama"},
    "Llama3.2-Vision 90B (GPU Server)":      {"id": "llama3.2-vision:90b",                 "desc": "最大規模",            "backend": "ollama"},
    # Llama4
    "Llama4 Scout 108B (GPU Server)":        {"id": "llama4:scout",                        "desc": "最新世代・大規模",    "backend": "ollama"},
    # Gemma シリーズ
    "Gemma4 31B (GPU Server)":               {"id": "gemma4:31b",                          "desc": "Google最新世代",      "backend": "ollama"},
    "Gemma3 27B (GPU Server)":               {"id": "gemma3:27b",                          "desc": "マルチモーダル",      "backend": "ollama"},
    "Gemma3 12B FP16 (GPU Server)":          {"id": "gemma3:12b-it-fp16",                  "desc": "FP16・高精度",        "backend": "ollama"},
    "Gemma3n E4B (GPU Server)":              {"id": "gemma3n:e4b",                         "desc": "軽量・効率型",        "backend": "ollama"},
    # GLM シリーズ
    "GLM-4.7 Flash BF16 (GPU Server)":       {"id": "glm-4.7-flash:bf16",                  "desc": "中国語強化",          "backend": "ollama"},
    "GLM-4.7 Flash BF16 192K (GPU Server)":  {"id": "glm-4.7-flash:bf16-192k",             "desc": "超長文コンテキスト",  "backend": "ollama"},
}

# ── NVIDIA NIM API モデル ──────────────────────────────────
NIM_VISION_MODELS = {
    "Llama 3.2 11B Vision (NIM)":  {"id": "meta/llama-3.2-11b-vision-instruct",         "desc": "バランス型",    "backend": "nim"},
    "Llama 3.2 90B Vision (NIM)":  {"id": "meta/llama-3.2-90b-vision-instruct",         "desc": "高精度",        "backend": "nim"},
    "Llama 4 Maverick 17B (NIM)":  {"id": "meta/llama-4-maverick-17b-128e-instruct",    "desc": "最新世代",      "backend": "nim"},
    "Llama 4 Scout 17B (NIM)":     {"id": "meta/llama-4-scout-17b-16e-instruct",        "desc": "効率型",        "backend": "nim"},
    "Nemotron Nano VL 8B (NIM)":   {"id": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",   "desc": "物体検出強化",  "backend": "nim"},
    "Phi-4 Multimodal (NIM)":      {"id": "microsoft/phi-4-multimodal-instruct",        "desc": "最新型",        "backend": "nim"},
    "Phi-3.5 Vision (NIM)":        {"id": "microsoft/phi-3.5-vision-instruct",          "desc": "小型・高速",    "backend": "nim"},
}

# ── HuggingFace 無料枠モデル ───────────────────────────────
HF_VISION_MODELS = {
    "Qwen2-VL 7B (HuggingFace)":      {"id": "Qwen/Qwen2-VL-7B-Instruct",                "desc": "無料枠", "backend": "hf"},
    "Llama 3.2 11B Vision (HuggingFace)": {"id": "meta-llama/Llama-3.2-11B-Vision-Instruct", "desc": "無料枠", "backend": "hf"},
    "Pixtral 12B (HuggingFace)":       {"id": "mistralai/Pixtral-12B-2409",               "desc": "無料枠", "backend": "hf"},
}

ALL_VISION_MODELS = {**LOCAL_VISION_MODELS, **NIM_VISION_MODELS, **HF_VISION_MODELS}

TEXT_MODELS = {
    "Llama 3.3 70B":      {"id": "meta/llama-3.3-70b-instruct",            "desc": "要約向け"},
    "Llama 3.1 70B":      {"id": "meta/llama-3.1-70b-instruct",            "desc": "バランス型"},
    "Nemotron Super 49B": {"id": "nvidia/llama-3.3-nemotron-super-49b-v1", "desc": "推論特化"},
}

DETECTION_MODELS = {
    "v1": {"id": "PekingU/rtdetr_r50vd",    "label": "RT-DETR",   "color": (0,200,230),  "color_hex": "#00c8e6"},
    "v2": {"id": "PekingU/rtdetr_v2_r50vd", "label": "RT-DETRv2", "color": (80,210,110), "color_hex": "#50d26e"},
}

PROMPT_PRESETS = {
    "General Detection": (
        "この画像に写っているすべての要素を日本語で詳しく報告してください。"
        "【検出対象】人物・動物・車両・建物・自然物・テキスト・看板・製品・その他。"
        "【報告形式】①主要な物体一覧（種類・数・位置）②人物の行動 ③背景の特徴 ④注目点"
    ),
    "Traffic / Vehicle": (
        "交通・車両分析の専門家として日本語で報告。"
        "①車両：種類・台数・色 ②信号の色 ③歩行者：人数・行動 ④道路状況 ⑤混雑度判定"
    ),
    "Congestion Monitor": (
        "渋滞判定。【この形式のみで回答】\n"
        "上り車線：渋滞 or 順調（車両数：XX台）\n"
        "下り車線：渋滞 or 順調（車両数：XX台）\n"
        "根拠：（1行）"
    ),
    "Person / Action": (
        "人物行動分析。全員を検出し日本語で報告。"
        "①外見・服装 ②動作 ③位置関係 ④異常があれば優先報告"
    ),
    "Security Monitor": (
        "セキュリティ監視。"
        "①不審者 ②不正侵入 ③不審物 ④緊急度：異常なし/要注意/要即時対応"
    ),
    "Text / Signage OCR": (
        "画像内のすべてのテキストをOCRして日本語で報告。"
        "看板・標識・ラベル・ナンバープレート等すべての文字を抽出。"
    ),
    "Equipment / Product": (
        "設備・製品検査。機器・設備を技術的に分析。"
        "①機器の特定 ②状態評価（正常/異常）③ケーブル・配線 ④ランプ・パネルの状態 ⑤保守推奨"
    ),
}

SUMMARIZE_PROMPT = """あなたは動画解析の専門家です。
以下の時系列フレーム分析を日本語で包括的に要約してください。
1.概要 2.主要な出来事の流れ 3.検出された主要物体 4.ハイライト 5.考察
---
{analysis_data}
---"""

# ─────────────────────────────────────────────
# Streamlit 設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Video Intelligence Terminal",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS（近未来風 + トグルボタンスタイル）
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600&family=Share+Tech+Mono&display=swap');

/* ── ベース ── */
.stApp { background-color: #010d1a; color: #b8d8e8; }
.stApp > header { background: transparent !important; }
div[data-testid="stSidebar"] { background: #020f1f; border-right: 1px solid #0a2a45; }
div[data-testid="stSidebar"] * { color: #8ab8cc !important; }

/* ── ヘッダー ── */
.vit-header {
    background: linear-gradient(135deg, #010d1a 0%, #021a35 40%, #02244a 100%);
    border: 1px solid #0a3a60; border-top: 2px solid #00b4d8;
    border-radius: 3px; padding: 12px 20px 10px; margin-bottom: 8px;
    position: relative; overflow: hidden;
}
.vit-header::before {
    content: ''; position: absolute; inset: 0;
    background: repeating-linear-gradient(0deg,transparent,transparent 3px,
        rgba(0,180,216,0.02) 3px,rgba(0,180,216,0.02) 4px);
    pointer-events: none;
}
.vit-header h1 {
    font-family: 'Orbitron', monospace; font-size: 1.3rem; font-weight: 900;
    color: #00d4f5; margin: 0; letter-spacing: 0.08em;
    text-shadow: 0 0 20px rgba(0,212,245,0.4);
}
.vit-header .subtitle {
    font-family: 'Rajdhani', sans-serif; font-size: 0.72rem; color: #2a6a88;
    margin: 2px 0 0; letter-spacing: 0.1em; text-transform: uppercase;
}
.vit-badge {
    display: inline-block; background: #00b4d8; color: #010d1a;
    font-family: 'Orbitron', monospace; font-size: 0.52rem; font-weight: 700;
    padding: 2px 6px; border-radius: 2px; margin-left: 8px; vertical-align: middle;
}

/* ── トグルボタン共通 ── */
.toggle-wrap {
    display: flex; flex-direction: column; align-items: center; gap: 4px;
}
.toggle-label {
    font-family: 'Orbitron', monospace; font-size: 0.52rem;
    color: #2a5a78; letter-spacing: 0.18em; text-transform: uppercase;
}


/* ── Streamlit プライマリカラーをシアンに上書き（トグル・スライダー等） ── */
:root {
    --primary-color: #00b4d8 !important;
}
/* input range (スライダー) */
input[type="range"]::-webkit-slider-thumb { background: #00b4d8 !important; }
input[type="range"]::-webkit-slider-runnable-track { background: #00b4d8 !important; }
/* トグルスイッチ全般 */
[role="switch"] { background-color: #00b4d8 !important; }
[role="switch"][aria-checked="false"] { background-color: #0a2a40 !important; }

/* ── トグルスイッチ カラー統一（シアン／近未来風）── */
div[data-testid="column"]:nth-child(2) [role="switch"][aria-checked="true"],
div[data-testid="column"]:nth-child(3) [role="switch"][aria-checked="true"] {
    background-color: #00b4d8 !important;
    border-color: #00b4d8 !important;
}
div[data-testid="column"]:nth-child(2) [role="switch"][aria-checked="false"],
div[data-testid="column"]:nth-child(3) [role="switch"][aria-checked="false"] {
    background-color: #0a2a40 !important;
    border-color: #0a3a55 !important;
}
div[data-testid="column"]:nth-child(2) [role="switch"],
div[data-testid="column"]:nth-child(3) [role="switch"] {
    background-color: #00b4d8 !important;
}
/* ラベル文字 → シアン・Orbitron */
div[data-testid="column"]:nth-child(2) div[data-testid="stToggle"] p,
div[data-testid="column"]:nth-child(3) div[data-testid="stToggle"] p {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    color: #00b4d8 !important;
    text-transform: uppercase !important;
}
.compute-banner {
    display: flex; align-items: center; gap: 12px;
    padding: 5px 14px; border-radius: 3px; margin-bottom: 6px;
    font-family: 'Orbitron', monospace; font-size: 0.62rem;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.compute-banner.gpu { background:linear-gradient(90deg,#001a10,#002d1a); border:1px solid #00e57a; border-left:4px solid #00e57a; }
.compute-banner.cpu { background:linear-gradient(90deg,#0a0d1a,#121a2e); border:1px solid #1e4a6a; border-left:4px solid #1e4a6a; }
.compute-label-gpu { color:#00e57a; font-weight:700; }
.compute-label-cpu { color:#3a8aaa; font-weight:700; }
.compute-detail    { color:#2a5a70; font-size:0.56rem; margin-top:1px; }
.compute-badge { margin-left:auto; padding:2px 8px; border-radius:2px; font-size:0.54rem; font-weight:700; }
.compute-badge.gpu { background:#00e57a15; color:#00e57a; border:1px solid #00e57a30; }
.compute-badge.cpu { background:#1e4a6a15; color:#3a8aaa; border:1px solid #1e4a6a30; }

/* ── ステータスバー ── */
.status-bar {
    display: flex; flex-wrap: wrap; gap: 10px; padding: 4px 12px;
    background: #020f1e; border: 1px solid #0a2a3e; border-radius: 2px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.62rem;
    margin-bottom: 6px;
}
.status-item { color: #1e5570; text-transform: uppercase; }
.status-val  { color: #00b4d8; margin-left: 4px; }
.status-val.ok   { color: #00e5a0; }
.status-val.busy { color: #ffd166; }
.status-val.run  { color: #ff3060; animation: blink 1s step-start infinite; }
.status-val.gpu  { color: #00e57a; font-weight: bold; }
.status-val.cpu  { color: #3a8aaa; }
@keyframes blink { 50% { opacity: 0; } }

/* ── パネル ── */
.panel-title {
    font-family: 'Orbitron', monospace; font-size: 0.62rem; font-weight: 600;
    color: #2a7a9a; letter-spacing: 0.14em; text-transform: uppercase;
    margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid #0a2a40;
}
.analysis-card {
    background: #020f1f; border: 1px solid #0a3a55;
    border-left: 3px solid #00b4d8; border-radius: 2px; padding: 8px 10px; margin: 3px 0;
}
.analysis-ts   { font-family:'Share Tech Mono',monospace; font-size:0.62rem; color:#00b4d8; }
.analysis-body { color:#b0cce0; font-size:0.8rem; margin-top:3px; line-height:1.6; }
.detection-card {
    background: #020f1f; border: 1px solid #0a3a55; border-radius: 2px;
    padding: 3px 7px; margin: 2px 0;
    font-family: 'Share Tech Mono', monospace; font-size: 0.64rem;
}
.det-v1  { border-left: 3px solid #00c8e6; }
.det-v2  { border-left: 3px solid #50d26e; }
.det-label { color: #d0e8f0; }
.det-score { color: #ffd166; margin-left: 5px; }
.det-tag   { font-size: 0.52rem; color: #2a6070; margin-left: 4px; }
.det-model-header {
    font-family: 'Orbitron', monospace; font-size: 0.56rem;
    letter-spacing: 0.08em; padding: 2px 0 3px; margin-bottom: 2px;
    border-bottom: 1px solid #0a2a3e; text-transform: uppercase;
}

/* ── 汎用ボタン ── */
.stButton > button {
    background: transparent; border: 1px solid #0a3a55; color: #5ab8d0;
    font-family: 'Rajdhani', sans-serif; font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; border-radius: 2px;
    transition: all 0.15s;
}
.stButton > button:hover { background: #0a2a40; border-color: #00b4d8; color: #00d4f5; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #003d5c, #00567a);
    border-color: #00b4d8; color: #00d4f5; font-weight: 700;
}

/* ── タブ ── */
div[data-testid="stTabs"] button {
    font-family: 'Orbitron', monospace !important; font-size: 0.6rem !important;
    letter-spacing: 0.08em !important; color: #2a6a88 !important;
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
.search-highlight { background:#004d66; color:#00e5f0; padding:0 2px; border-radius:2px; }
.vit-footer {
    text-align: center; padding: 6px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: #1a4055;
    border-top: 1px solid #0a2a3e; margin-top: 10px;
}
.section-label {
    font-family: 'Orbitron', monospace; font-size: 0.62rem;
    color: #2a6a88; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# セッション状態
# ─────────────────────────────────────────────
_defaults = {
    "processing":            False,
    "youtube_url":           "",
    "video_file":            None,
    "stream_url":            None,
    "is_live":               False,
    "stream_title":          "",
    "selected_model":        GPU_DEFAULT_MODEL,
    "current_prompt":        PROMPT_PRESETS["General Detection"],
    "analysis_log":          [],
    "summary_text":          "",
    "search_query":          "",
    "search_results":        [],
    "total_frames_analyzed": 0,
    "mode":                  "YouTube",
    "highlights":            [],
    "local_folder":          DEFAULT_FOLDER,
    "det_v1_enabled":        True,
    "det_v2_enabled":        True,
    "use_gpu":               True,
    "perf_history":          [],
    "latest_analysis":       "",
    "latest_latency":        "—",
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────
# デバイス判定
# ─────────────────────────────────────────────
def get_device(use_gpu: bool) -> str:
    # video-ai-demo にはGPUがないため常にCPU
    # OllamaのVLMはGPUサーバー(192.168.11.111)で処理
    return "cpu"

_mode_str   = "GPU" if st.session_state.use_gpu else "CPU"
_cur_device = get_device(st.session_state.use_gpu)

# ─────────────────────────────────────────────
# ヘッダー行
# ─────────────────────────────────────────────
hdr_col, tog_start_col, tog_gpu_col = st.columns([5, 1, 1])

with hdr_col:
    st.markdown("""
    <div class="vit-header">
      <h1>VIDEO INTELLIGENCE TERMINAL
        <span class="vit-badge">LIVE</span>
        <span style="font-family:'Rajdhani',sans-serif;font-size:0.82rem;color:#1a6080;
              font-weight:300;letter-spacing:0.04em;margin-left:10px;">
          RT-DETR + Ollama VLM · GPU Server 192.168.11.111
        </span>
      </h1>
      <div class="subtitle">NVIDIA H100 · CUDA · Parallel Processing · RT-DETR v1+v2 · Ollama Local</div>
    </div>
    """, unsafe_allow_html=True)

# ── START / STOP トグル（RT-DETRと同形式）──
with tog_start_col:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _new_processing = st.toggle("START", value=st.session_state.processing, key="start_stop_toggle")
    if _new_processing != st.session_state.processing:
        st.session_state.processing = _new_processing
        st.rerun()

# ── GPU ON/OFF トグル（RT-DETRと同形式）──
with tog_gpu_col:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _new_gpu = st.toggle("GPU", value=st.session_state.use_gpu, key="gpu_toggle")
    if _new_gpu != st.session_state.use_gpu:
        st.session_state.use_gpu = _new_gpu
        st.session_state.selected_model = GPU_DEFAULT_MODEL if _new_gpu else CPU_DEFAULT_MODEL
        st.cache_resource.clear()
        st.rerun()

# ── バナー ──
_model_size = "LARGE MODEL (32B+) · CUDA" if st.session_state.use_gpu else "SMALL MODEL (7B) · CPU"
_device_label = f"CUDA ({_cur_device.upper()})" if _cur_device == "cuda" else "CPU"

if st.session_state.use_gpu:
    st.markdown(f"""
    <div class="compute-banner gpu">
      <span style='font-size:1rem'>⚡</span>
      <div>
        <div class="compute-label-gpu">GPU MODE — OLLAMA GPU SERVER ACTIVE</div>
        <div class="compute-detail">
          RT-DETR: <b>CPU (video-ai-demo)</b> &nbsp;|&nbsp;
          Ollama VLM: <b>{OLLAMA_URL}</b> &nbsp;|&nbsp;
          VLM Model: <b>{_model_size}</b>
        </div>
      </div>
      <span class="compute-badge gpu">HIGH PERFORMANCE</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="compute-banner cpu">
      <span style='font-size:1rem'>💻</span>
      <div>
        <div class="compute-label-cpu">STANDARD MODE — LIGHTWEIGHT MODEL</div>
        <div class="compute-detail">
          RT-DETR: <b>CPU (video-ai-demo)</b> &nbsp;|&nbsp;
          Ollama VLM: <b>{OLLAMA_URL}</b> &nbsp;|&nbsp;
          VLM Model: <b>{_model_size}</b>
        </div>
      </div>
      <span class="compute-badge cpu">STANDARD</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RT-DETR ロード（GPU/CPU 切替対応）
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rtdetr_v1(device: str):
    if not DETECTION_AVAILABLE:
        return None, None, None, "torch/transformers not installed"
    if not _RTDETR_CACHED:
        return None, None, None, "モデル未キャッシュ (download_models.py を実行)"
    try:
        cfg   = DETECTION_MODELS["v1"]
        proc  = AutoImageProcessor.from_pretrained(cfg["id"], trust_remote_code=False, local_files_only=True)
        model = AutoModelForObjectDetection.from_pretrained(cfg["id"], trust_remote_code=False, local_files_only=True)
        model.to(device).eval()
        return proc, model, device, None
    except Exception as exc:
        return None, None, None, str(exc)

@st.cache_resource(show_spinner=False)
def load_rtdetr_v2(device: str):
    if not DETECTION_AVAILABLE:
        return None, None, None, "torch/transformers not installed"
    if not _RTDETR_CACHED:
        return None, None, None, "モデル未キャッシュ (download_models.py を実行)"
    try:
        cfg   = DETECTION_MODELS["v2"]
        proc  = AutoImageProcessor.from_pretrained(cfg["id"], trust_remote_code=False, local_files_only=True)
        model = AutoModelForObjectDetection.from_pretrained(cfg["id"], trust_remote_code=False, local_files_only=True)
        model.to(device).eval()
        return proc, model, device, None
    except Exception as exc:
        return None, None, None, str(exc)

# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────
def frame_to_base64(frame_rgb, quality=95):
    buf = io.BytesIO()
    Image.fromarray(frame_rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def resize_frame(frame, pct):
    if pct >= 100: return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w*pct/100), int(h*pct/100)), interpolation=cv2.INTER_AREA)

def draw_boxes(frame_rgb, detections):
    frame = frame_rgb.copy()
    for ver, dets in detections.items():
        cfg   = DETECTION_MODELS.get(ver, {})
        color = cfg.get("color", (200,200,200))
        for d in (dets or []):
            x1,y1,x2,y2 = d["box"]
            text = f"{d['label']} {d['score']:.2f}"
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            font=cv2.FONT_HERSHEY_SIMPLEX; fs,th=0.44,1
            (tw,th_px),bl=cv2.getTextSize(text,font,fs,th); pad=3
            cv2.rectangle(frame,(x1,max(0,y1-th_px-pad*2-bl)),(x1+tw+pad*2,y1),color,-1)
            cv2.putText(frame,text,(x1+pad,max(th_px+pad,y1-bl-pad)),font,fs,(5,5,5),th,cv2.LINE_AA)
    return frame

def run_detection(frame_rgb, use_v1, use_v2, threshold=0.45, device="cpu"):
    result = {"v1":[], "v2":[], "error": None}
    for ver, loader in [("v1", load_rtdetr_v1), ("v2", load_rtdetr_v2)]:
        if (ver=="v1" and not use_v1) or (ver=="v2" and not use_v2):
            continue
        proc, model, dev, err = loader(device)
        if err:
            result["error"] = (result["error"] or "") + f" [{ver}:{err}]"
            continue
        if not proc or not model:
            continue
        try:
            pil = Image.fromarray(frame_rgb)
            inp = proc(images=pil, return_tensors="pt")
            inp = {k: v.to(dev) for k, v in inp.items()}
            with torch.no_grad():
                out = model(**inp)
            ts    = torch.tensor([[pil.height, pil.width]])
            preds = proc.post_process_object_detection(out, threshold=threshold, target_sizes=ts)[0]
            for sc, lid, box in zip(preds["scores"], preds["labels"], preds["boxes"]):
                result[ver].append({
                    "label": model.config.id2label[lid.item()],
                    "score": round(float(sc.item()), 3),
                    "box":   [round(float(v)) for v in box.tolist()],
                })
        except Exception as exc:
            result["error"] = (result["error"] or "") + f" [{ver} inference:{exc}]"
    return result

def ollama_analyze(frame_rgb, prompt, model_id, resize_pct, max_tokens,
                   temperature=0.2, top_k=40, top_p=0.9):
    """
    Ollama /api/chat 形式で推論。
    content は string、images は別フィールドで渡す（Ollama互換形式）。
    """
    if resize_pct < 100:
        frame_rgb = resize_frame(frame_rgb, resize_pct)
    img_b64 = frame_to_base64(frame_rgb)
    t0 = time.time()
    try:
        payload = {
            "model": model_id,
            "messages": [{
                "role":    "user",
                "content": prompt,
                "images":  [img_b64],
            }],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature,
                        "top_k": top_k, "top_p": top_p},
        }
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload, timeout=300, verify=False)
        lat = time.time() - t0
        if resp.status_code == 200:
            data = resp.json()
            text = (data.get("message", {}).get("content", "")
                    or data.get("response", ""))
            return {"ok": True, "text": text, "latency": lat, "img_b64": img_b64}
        if resp.status_code == 404:
            return {"ok": False, "text": f"モデル未インストール: ollama pull {model_id}", "latency": lat}
        return {"ok": False, "text": f"Ollama HTTP {resp.status_code}: {resp.text[:200]}", "latency": lat}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "text": f"Ollama 接続失敗 ({OLLAMA_URL})\nollama serve を確認してください",
                "latency": time.time()-t0}
    except requests.exceptions.Timeout:
        return {"ok": False, "text": f"Ollama タイムアウト (300s) — モデルが大きすぎる可能性があります",
                "latency": time.time()-t0}
    except Exception as exc:
        return {"ok": False, "text": f"Ollama エラー: {exc}", "latency": time.time()-t0}

def nim_analyze(frame_rgb, prompt, model_id, resize_pct, max_tokens):
    if resize_pct < 100: frame_rgb = resize_frame(frame_rgb, resize_pct)
    img_b64 = frame_to_base64(frame_rgb)
    t0 = time.time()
    try:
        resp = requests.post(
            f"{NIM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"},
            json={"model": model_id,
                  "messages": [{"role":"user","content":[
                      {"type":"text","text":prompt},
                      {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}},
                  ]}],
                  "max_tokens": max_tokens, "temperature": 0.2, "stream": False},
            timeout=90, verify=SSL_VERIFY)
        lat = time.time()-t0
        if resp.status_code == 200:
            return {"ok":True,"text":resp.json()["choices"][0]["message"]["content"],"latency":lat,"img_b64":img_b64}
        return {"ok":False,"text":f"NIM {resp.status_code}: {resp.text[:200]}","latency":lat}
    except Exception as exc:
        return {"ok":False,"text":f"NIM error: {exc}","latency":0.0}

def hf_analyze(frame_rgb, prompt, model_id, resize_pct, max_tokens):
    if not HF_HUB_AVAILABLE or not HF_TOKEN:
        return {"ok":False,"text":"HF_TOKEN not set","latency":0.0}
    if resize_pct < 100: frame_rgb = resize_frame(frame_rgb, resize_pct)
    img_b64 = frame_to_base64(frame_rgb)
    t0 = time.time()
    try:
        resp = InferenceClient(token=HF_TOKEN).chat_completion(
            model=model_id,
            messages=[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}},
                {"type":"text","text":prompt},
            ]}], max_tokens=max_tokens)
        return {"ok":True,"text":resp.choices[0].message.content,"latency":time.time()-t0,"img_b64":img_b64}
    except Exception as exc:
        return {"ok":False,"text":f"HF error: {exc}","latency":time.time()-t0}

def vlm_analyze(frame_rgb, prompt, model_name, resize_pct=80, max_tokens=400,
                temperature=0.2, top_k=40, top_p=0.9):
    cfg = ALL_VISION_MODELS.get(model_name)
    if not cfg: return {"ok":False,"text":f"Unknown model: {model_name}","latency":0.0}
    if cfg["backend"] == "ollama":
        return ollama_analyze(frame_rgb, prompt, cfg["id"], resize_pct, max_tokens,
                              temperature, top_k, top_p)
    if cfg["backend"] == "nim":    return nim_analyze(frame_rgb, prompt, cfg["id"], resize_pct, max_tokens)
    return hf_analyze(frame_rgb, prompt, cfg["id"], resize_pct, max_tokens)

def nim_text_summarize(log, model_name="Llama 3.3 70B"):
    mid     = TEXT_MODELS.get(model_name, TEXT_MODELS["Llama 3.3 70B"])["id"]
    log_str = "\n\n".join([f"[{e['ts']}] Frame {e['frame_idx']}: {e['text']}" for e in log])
    t0 = time.time()
    try:
        resp = requests.post(
            f"{NIM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type":"application/json"},
            json={"model":mid, "messages":[{"role":"user","content":SUMMARIZE_PROMPT.format(analysis_data=log_str)}],
                  "max_tokens":1500,"temperature":0.3,"stream":False},
            timeout=120, verify=SSL_VERIFY)
        if resp.status_code==200:
            return {"ok":True,"text":resp.json()["choices"][0]["message"]["content"],"latency":time.time()-t0}
        return {"ok":False,"text":f"API {resp.status_code}: {resp.text[:200]}"}
    except Exception as exc:
        return {"ok":False,"text":str(exc)}

def search_log(log, query):
    if not query.strip(): return []
    kws = [k.strip().lower() for k in re.split(r"[\s\u3000,\u3001]", query) if k.strip()]
    return [e for e in log if any(kw in e["text"].lower() for kw in kws)]

def highlight_text(text, query):
    if not query: return text
    for kw in re.split(r"[\s\u3000,\u3001]", query):
        if kw.strip():
            text = re.sub(re.escape(kw.strip()),
                          f'<span class="search-highlight">{kw.strip()}</span>',
                          text, flags=re.IGNORECASE)
    return text

def extract_tags(log, top_n=30):
    from collections import Counter
    STOP = {"の","に","は","を","が","で","と","た","し","て","い","な","も","から","です","ます",
            "a","an","the","is","in","on","at","to","of","and","or","with","are","was",
            "画像","写真","フレーム","場面","状況","内容","様子"}
    c = Counter()
    for e in log:
        for w in re.findall(r'[一-龯ぁ-んァ-ン]{2,}',e["text"])+re.findall(r'[A-Za-z]{3,}',e["text"]):
            if w.lower() not in STOP: c[w] += 1
    return [w for w,n in c.most_common(top_n) if n>=1]

def detect_highlights(log, top_n=5):
    return sorted(log, key=lambda x:len(x.get("text","")), reverse=True)[:top_n] if log else []

def download_youtube(url):
    if not YT_DLP_AVAILABLE: return None,"yt-dlp not installed"
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = str(Path(DOWNLOAD_DIR)/f"video_{ts}.%(ext)s")
    try:
        with yt_dlp.YoutubeDL({"format":"best[height>=1080][ext=mp4]/best[height>=1080]/best","outtmpl":out,"quiet":True}) as ydl:
            info  = ydl.extract_info(url, download=True)
            title = info.get("title","Unknown")
            for ext in [".mp4",".webm",".mkv"]:
                p = Path(DOWNLOAD_DIR)/f"video_{ts}{ext}"
                if p.exists(): return str(p), title
            for f in Path(DOWNLOAD_DIR).iterdir():
                if f.name.startswith(f"video_{ts}"): return str(f), f.stem
            return None,"File not found"
    except Exception as exc:
        return None, str(exc)

def get_stream_url(url):
    if not YT_DLP_AVAILABLE: return None,"",False,"yt-dlp not installed"
    try:
        with yt_dlp.YoutubeDL({"format":"best[height>=1080]/best","quiet":True,
                                "no_warnings":True,"skip_download":True}) as ydl:
            info       = ydl.extract_info(url, download=False)
            title      = info.get("title","Unknown")
            is_live    = info.get("is_live",False)
            stream_url = info.get("url")
            if not stream_url and "formats" in info:
                for fmt in reversed(info["formats"]):
                    if (fmt.get("height") or 0)>=1080 and fmt.get("url"):
                        stream_url=fmt["url"]; break
            return (stream_url,title,is_live,None) if stream_url else (None,title,is_live,"No URL")
    except Exception as exc:
        return None,"",False,str(exc)

def get_video_files(folder):
    exts={".mp4",".webm",".mkv",".mov",".avi",".m4v",".ts"}
    try:
        return sorted([f for f in Path(folder).iterdir() if f.suffix.lower() in exts], reverse=True)
    except Exception:
        return []

# ─────────────────────────────────────────────
# ステータスバー
# ─────────────────────────────────────────────
status_ph = st.empty()

def render_status(status="READY"):
    sc  = "ok" if status in ("READY","OK") else ("run" if status=="RUNNING" else "busy")
    mc  = "gpu" if st.session_state.use_gpu else "cpu"
    lat = st.session_state.latest_latency
    frm = st.session_state.total_frames_analyzed
    mdl = st.session_state.selected_model
    v1s = "ON" if st.session_state.det_v1_enabled else "OFF"
    v2s = "ON" if st.session_state.det_v2_enabled else "OFF"
    dev = _cur_device.upper()
    status_ph.markdown(
        f"<div class='status-bar'>"
        f"<span class='status-item'>STATUS <span class='status-val {sc}'>{status}</span></span>"
        f"<span class='status-item'>COMPUTE <span class='status-val {mc}'>{_mode_str} / {dev}</span></span>"
        f"<span class='status-item'>LATENCY <span class='status-val'>{lat}</span></span>"
        f"<span class='status-item'>FRAMES <span class='status-val'>{frm}</span></span>"
        f"<span class='status-item'>DETR <span class='status-val'>{v1s}</span>/<span class='status-val'>{v2s}</span></span>"
        f"<span class='status-item'>MODEL <span class='status-val'>{mdl}</span></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

render_status("RUNNING" if st.session_state.processing else "READY")

# ─────────────────────────────────────────────
# サイドバー
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='section-label'>Input Source</div>", unsafe_allow_html=True)
    source_mode = st.radio("", ["YouTube (Download)","YouTube (Live)","Local Folder"])
    st.session_state.mode = ("Live" if "Live" in source_mode
                              else "YouTube" if "YouTube" in source_mode else "Local")

    if st.session_state.mode == "YouTube":
        # key指定でURL保持
        yt_url = st.text_input("YouTube URL", key="yt_url_input",
                               value=st.session_state.youtube_url,
                               placeholder="https://youtu.be/xxxx")
        if yt_url != st.session_state.youtube_url:
            st.session_state.youtube_url = yt_url
            st.session_state.video_file  = None
        if st.button("Download", type="primary"):
            with st.spinner("Downloading..."):
                path, title = download_youtube(st.session_state.youtube_url)
                if path:
                    st.session_state.video_file   = path
                    st.session_state.stream_title = title
                    st.success(f"OK: {title}")
                else:
                    st.error(title)

    elif st.session_state.mode == "Live":
        # key指定でURL保持・rerun後も消えない
        live_url = st.text_input("Live URL", key="live_url_input",
                                 value=st.session_state.youtube_url,
                                 placeholder="https://youtu.be/xxxx")
        if live_url != st.session_state.youtube_url:
            st.session_state.youtube_url = live_url
            st.session_state.stream_url  = None
        if st.button("Connect", type="primary"):
            with st.spinner("Connecting..."):
                url,title,is_live,err = get_stream_url(st.session_state.youtube_url)
                if url:
                    st.session_state.stream_url   = url
                    st.session_state.stream_title = title
                    st.session_state.is_live      = is_live
                    st.success(f"{'LIVE' if is_live else 'VOD'}: {title}")
                else:
                    st.error(str(err))
    else:
        folder = st.text_input("Folder", value=st.session_state.local_folder)
        if folder != st.session_state.local_folder:
            st.session_state.local_folder = folder
        files = get_video_files(folder)
        if files:
            sel = st.selectbox("File", [f.name for f in files])
            if sel: st.session_state.video_file = str(Path(folder)/sel)
        else:
            st.info("No video files")

    st.divider()
    st.markdown("<div class='section-label'>Detection</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.session_state.det_v1_enabled = st.toggle("RT-DETR v1", value=st.session_state.det_v1_enabled)
    with c2: st.session_state.det_v2_enabled = st.toggle("RT-DETRv2",  value=st.session_state.det_v2_enabled)
    det_threshold = st.slider("Threshold", 0.1, 0.9, 0.45, 0.05)

    st.divider()
    st.markdown("<div class='section-label'>VLM Model</div>", unsafe_allow_html=True)
    st.caption(f"Ollama: {OLLAMA_URL}")
    model_opts = list(ALL_VISION_MODELS.keys())
    cur_idx    = model_opts.index(st.session_state.selected_model) \
                 if st.session_state.selected_model in model_opts else 0
    sel_model  = st.selectbox("Model", model_opts, index=cur_idx)
    if sel_model != st.session_state.selected_model:
        st.session_state.selected_model = sel_model

    st.divider()
    st.markdown("<div class='section-label'>Analysis Settings</div>", unsafe_allow_html=True)
    vlm_interval = st.slider("VLM Interval (s)", 1, 30, 5, 1)
    max_tokens   = st.slider("Max Tokens", 100, 800, 600 if st.session_state.use_gpu else 300, 50)
    resize_pct   = st.slider("Image Resize (%)", 20, 100, 80 if st.session_state.use_gpu else 60, 10)

    st.markdown("<div class='section-label' style='margin-top:8px'>Generation Parameters</div>",
                unsafe_allow_html=True)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    top_k       = st.slider("Top-K",       1,   100,  40,  1)
    top_p       = st.slider("Top-P",       0.0, 1.0,  0.9, 0.05)

    st.divider()
    st.markdown("<div class='section-label'>Prompt</div>", unsafe_allow_html=True)
    preset_name = st.selectbox("Preset", list(PROMPT_PRESETS.keys()))
    if st.button("Apply"):
        st.session_state.current_prompt = PROMPT_PRESETS[preset_name]; st.rerun()
    custom_p = st.text_area("Custom", value=st.session_state.current_prompt, height=100)
    if custom_p != st.session_state.current_prompt:
        st.session_state.current_prompt = custom_p

    st.divider()
    # ── Ollama 接続テスト（全モデル一括確認）──
    if st.button("🔌 Test Ollama Connection"):
        with st.spinner("Testing..."):
            try:
                r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=8, verify=False)
                if r.status_code == 200:
                    installed = {m["name"] for m in r.json().get("models",[])}
                    st.success(f"接続OK — {len(installed)} models installed")
                    st.markdown("**モデル対応状況:**")
                    for name, cfg in LOCAL_VISION_MODELS.items():
                        mid = cfg["id"]
                        # モデルIDのベース名で照合
                        ok = any(mid.split(":")[0] in m for m in installed)
                        icon = "✅" if ok else "❌"
                        st.caption(f"{icon} {name.replace(' (GPU Server)','')}")
                else:
                    st.error(f"HTTP {r.status_code}")
            except Exception as e:
                st.error(str(e))

    if st.button("🗑 Clear All"):
        st.session_state.analysis_log          = []
        st.session_state.total_frames_analyzed = 0
        st.session_state.highlights            = []
        st.session_state.perf_history          = []
        st.session_state.latest_analysis       = ""
        st.session_state.latest_latency        = "—"
        st.rerun()

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

    # ── 入力ソース表示バー ──
    _src_ready = False
    if st.session_state.mode == "Live" and st.session_state.stream_url:
        st.info(f"LIVE: {st.session_state.stream_title[:60]}")
        _src_ready = True
    elif st.session_state.video_file:
        st.info(f"{Path(st.session_state.video_file).name}")
        _src_ready = True
    else:
        st.warning("⚠️ 左上 ≡ サイドバーで Input Source を設定してください")

    # ── ファーストビュー 3カラム ──
    col_feed, col_det, col_ai = st.columns([5, 3, 4])

    with col_feed:
        st.markdown("<div class='panel-title'>LIVE FEED</div>", unsafe_allow_html=True)
        frame_ph = st.empty()

    with col_det:
        st.markdown("<div class='panel-title'>RT-DETR DETECTION</div>", unsafe_allow_html=True)
        det_ph = st.empty()

    with col_ai:
        st.markdown("<div class='panel-title'>AI ANALYSIS JPN</div>", unsafe_allow_html=True)
        ai_ph = st.empty()
        if st.session_state.latest_analysis:
            ai_ph.markdown(
                f"<div class='analysis-card'>"
                f"<div class='analysis-body'>{st.session_state.latest_analysis}</div>"
                f"</div>", unsafe_allow_html=True)

    # ── 処理ループ ──
    if st.session_state.processing and _src_ready:
        video_source = (st.session_state.stream_url
                        if st.session_state.mode == "Live"
                        else st.session_state.video_file)
        cap = cv2.VideoCapture(video_source)

        # 高解像度設定（1920x1080を要求）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファ最小化で最新フレームを取得

        if not cap.isOpened():
            st.error("動画ソースを開けません")
            st.session_state.processing = False
        else:
            render_status("RUNNING")
            last_vlm_time = 0.0
            frame_idx     = 0
            _device       = get_device(st.session_state.use_gpu)

            while st.session_state.processing:
                ret, frame = cap.read()
                if not ret:
                    if st.session_state.mode != "Live":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                    break

                frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_idx += 1

                # RT-DETR（GPU/CPU 切替）
                det_result = run_detection(
                    frame_rgb,
                    use_v1=st.session_state.det_v1_enabled,
                    use_v2=st.session_state.det_v2_enabled,
                    threshold=det_threshold,
                    device=_device,
                )

                # ① 映像
                annotated = draw_boxes(frame_rgb,
                    {k:v for k,v in det_result.items() if k in ("v1","v2")})
                frame_ph.image(annotated, channels="RGB", use_container_width=True)

                # ② 検知リスト
                det_html = ""
                for ver in ("v1","v2"):
                    dets = det_result.get(ver,[])
                    if dets:
                        cfg = DETECTION_MODELS[ver]
                        det_html += (f"<div class='det-model-header' style='color:{cfg['color_hex']}'>"
                                     f"{cfg['label']} [{_device.upper()}] — {len(dets)} obj</div>")
                        for d in dets[:12]:
                            det_html += (f"<div class='detection-card det-{ver}'>"
                                         f"<span class='det-label'>{d['label']}</span>"
                                         f"<span class='det-score'>{d['score']:.2f}</span>"
                                         f"<span class='det-tag'>[{ver.upper()}]</span></div>")
                if det_result.get("error"):
                    det_html += f"<div style='color:#ef476f;font-size:0.6rem'>{det_result['error']}</div>"
                if det_html:
                    det_ph.markdown(det_html, unsafe_allow_html=True)

                # ③ VLM 分析
                now = time.time()
                if now - last_vlm_time >= vlm_interval:
                    last_vlm_time = now
                    ts_str = datetime.now().strftime("%H:%M:%S")
                    render_status("ANALYZING")

                    result = vlm_analyze(
                        frame_rgb,
                        st.session_state.current_prompt,
                        st.session_state.selected_model,
                        resize_pct=resize_pct,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )

                    lat_str = f"{result['latency']:.2f}s"
                    st.session_state.latest_latency = lat_str

                    if result["ok"]:
                        st.session_state.latest_analysis = result["text"]
                        entry = {
                            "ts": ts_str, "frame_idx": frame_idx,
                            "text": result["text"], "img_b64": result.get("img_b64",""),
                            "latency": result["latency"],
                            "model": st.session_state.selected_model,
                            "mode": _mode_str, "device": _device,
                        }
                        st.session_state.analysis_log.append(entry)
                        st.session_state.total_frames_analyzed += 1
                        st.session_state.highlights = detect_highlights(st.session_state.analysis_log)
                        st.session_state.perf_history.append({
                            "mode": _mode_str, "latency": result["latency"],
                            "ts": ts_str, "model": st.session_state.selected_model,
                        })
                        if len(st.session_state.perf_history) > 100:
                            st.session_state.perf_history = st.session_state.perf_history[-100:]

                        # ── ログを LOG_DIR に自動保存（img_b64除く） ──
                        try:
                            _log_file = LOG_DIR / f"log_{datetime.now().strftime('%Y%m%d')}.jsonl"
                            with open(_log_file, "a", encoding="utf-8") as _lf:
                                _lf.write(json.dumps(
                                    {k:v for k,v in entry.items() if k != "img_b64"},
                                    ensure_ascii=False) + "\n")
                        except Exception:
                            pass

                        ai_ph.markdown(
                            f"<div class='analysis-card'>"
                            f"<div class='analysis-ts'>[{ts_str}] Frame {frame_idx} | {lat_str} | {_mode_str}/{_device.upper()}</div>"
                            f"<div class='analysis-body'>{result['text']}</div>"
                            f"</div>", unsafe_allow_html=True)
                    else:
                        ai_ph.error(result["text"])

                    render_status("RUNNING")

                time.sleep(0.03)

            cap.release()
            st.session_state.processing = False
            render_status("READY")

# ────────────────────────────────────────────────────────────
# TAB: VIDEO SEARCH
# ────────────────────────────────────────────────────────────
with tab_search:
    st.markdown("<div class='panel-title'>Search Analysis Log</div>", unsafe_allow_html=True)
    tags = extract_tags(st.session_state.analysis_log)
    if tags:
        tcols = st.columns(min(len(tags),8))
        for i,tag in enumerate(tags[:8]):
            with tcols[i%8]:
                if st.button(tag, key=f"tag_{i}"):
                    st.session_state.search_query   = tag
                    st.session_state.search_results = search_log(st.session_state.analysis_log, tag)
    query = st.text_input("Search", value=st.session_state.search_query,
                          placeholder="キーワード（スペース区切りでAND）")
    if query != st.session_state.search_query:
        st.session_state.search_query   = query
        st.session_state.search_results = search_log(st.session_state.analysis_log, query)
    if st.session_state.search_results:
        st.info(f"{len(st.session_state.search_results)} 件")
        for e in st.session_state.search_results:
            st.markdown(
                f"<div class='analysis-card'>"
                f"<div class='analysis-ts'>[{e['ts']}] Frame {e['frame_idx']} | {e.get('mode','—')}</div>"
                f"<div class='analysis-body'>{highlight_text(e['text'],query)}</div>"
                f"</div>", unsafe_allow_html=True)
    elif query:
        st.info("該当なし")

# ────────────────────────────────────────────────────────────
# TAB: SUMMARIZATION
# ────────────────────────────────────────────────────────────
with tab_summary:
    st.markdown("<div class='panel-title'>Video Summarization</div>", unsafe_allow_html=True)
    sum_model = st.selectbox("Text Model", list(TEXT_MODELS.keys()))
    if st.button("Generate Summary", type="primary"):
        if st.session_state.analysis_log:
            with st.spinner("Generating..."):
                r = nim_text_summarize(st.session_state.analysis_log, sum_model)
                st.session_state.summary_text = r["text"]
                # SUMMARY_DIR に自動保存
                try:
                    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    _sf = SUMMARY_DIR / f"summary_{_ts}.txt"
                    _sf.write_text(st.session_state.summary_text, encoding="utf-8")
                except Exception:
                    pass
        else:
            st.warning("分析ログがありません")
    if st.session_state.summary_text:
        st.markdown(f"<div class='analysis-card'><div class='analysis-body'>"
                    f"{st.session_state.summary_text}</div></div>", unsafe_allow_html=True)
        _dl_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("📥 Download", data=st.session_state.summary_text,
                           file_name=f"summary_{_dl_ts}.txt", mime="text/plain")

# ────────────────────────────────────────────────────────────
# TAB: HIGHLIGHTS
# ────────────────────────────────────────────────────────────
with tab_highlights:
    st.markdown("<div class='panel-title'>Auto Highlights</div>", unsafe_allow_html=True)
    if st.session_state.highlights:
        for i,e in enumerate(st.session_state.highlights):
            c1,c2 = st.columns([1,2])
            with c1:
                if e.get("img_b64"):
                    try: st.image(Image.open(io.BytesIO(base64.b64decode(e["img_b64"]))), use_container_width=True)
                    except: pass
            with c2:
                st.markdown(
                    f"<div class='analysis-card'>"
                    f"<div class='analysis-ts'>#{i+1} [{e['ts']}] | {e.get('mode','—')}</div>"
                    f"<div class='analysis-body'>{e['text'][:300]}</div>"
                    f"</div>", unsafe_allow_html=True)
    else:
        st.info("ハイライトなし。分析を実行してください。")

# ────────────────────────────────────────────────────────────
# TAB: ANALYSIS LOG
# ────────────────────────────────────────────────────────────
with tab_log:
    st.markdown("<div class='panel-title'>Full Analysis Log</div>", unsafe_allow_html=True)
    if st.session_state.analysis_log:
        st.info(f"Total: {len(st.session_state.analysis_log)} entries")
        if st.button("📥 Export JSON"):
            _export_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            _export_data = json.dumps(
                [{k:v for k,v in e.items() if k!="img_b64"}
                  for e in st.session_state.analysis_log],
                ensure_ascii=False, indent=2)
            # EXPORT_DIR に自動保存
            try:
                _ef = EXPORT_DIR / f"log_{_export_ts}.json"
                _ef.write_text(_export_data, encoding="utf-8")
            except Exception:
                pass
            st.download_button("Download",
                data=_export_data,
                file_name=f"log_{_export_ts}.json", mime="application/json")
        for e in reversed(st.session_state.analysis_log[-50:]):
            st.markdown(
                f"<div class='analysis-card'>"
                f"<div class='analysis-ts'>[{e['ts']}] Frame {e['frame_idx']}"
                f" | {e.get('latency',0):.2f}s | {e.get('mode','—')}/{e.get('device','—')}</div>"
                f"<div class='analysis-body'>{e['text']}</div>"
                f"</div>", unsafe_allow_html=True)
    else:
        st.info("ログなし")

# ────────────────────────────────────────────────────────────
# TAB: PERFORMANCE
# ────────────────────────────────────────────────────────────
with tab_perf:
    st.markdown("<div class='panel-title'>GPU vs CPU Performance</div>", unsafe_allow_html=True)
    if st.session_state.perf_history:
        import pandas as pd
        gpu_r = [r for r in st.session_state.perf_history if r["mode"]=="GPU"]
        cpu_r = [r for r in st.session_state.perf_history if r["mode"]=="CPU"]
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.metric("GPU Avg Latency",
                            f"{sum(r['latency'] for r in gpu_r)/len(gpu_r):.2f}s" if gpu_r else "—")
        with m2: st.metric("CPU Avg Latency",
                            f"{sum(r['latency'] for r in cpu_r)/len(cpu_r):.2f}s" if cpu_r else "—")
        with m3:
            if gpu_r and cpu_r:
                ag=sum(r["latency"] for r in gpu_r)/len(gpu_r)
                ac=sum(r["latency"] for r in cpu_r)/len(cpu_r)
                st.metric("Speedup", f"{ac/ag:.1f}×" if ag>0 else "—")
            else: st.metric("Speedup","—")
        with m4: st.metric("Samples", len(st.session_state.perf_history))
        df  = pd.DataFrame(st.session_state.perf_history)
        df["idx"] = range(len(df))
        gdf = df[df["mode"]=="GPU"][["idx","latency"]].rename(columns={"latency":"GPU(s)"})
        cdf = df[df["mode"]=="CPU"][["idx","latency"]].rename(columns={"latency":"CPU(s)"})
        merged = pd.merge(gdf, cdf, on="idx", how="outer").set_index("idx")
        if not merged.empty: st.line_chart(merged)
        if st.button("🗑 Clear"): st.session_state.perf_history=[]; st.rerun()
    else:
        st.info("GPU/CPU モードを切り替えながら分析を実行するとグラフが表示されます。")

# ─────────────────────────────────────────────
# フッター
# ─────────────────────────────────────────────
st.markdown(
    "<div class='vit-footer'>"
    "VIDEO INTELLIGENCE TERMINAL &nbsp;|&nbsp;"
    " RT-DETR CUDA/CPU · Ollama VLM · GPU Server 192.168.11.111 &nbsp;|&nbsp; CISCO AI LAB"
    "</div>",
    unsafe_allow_html=True,
)
