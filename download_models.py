"""
モデル事前ダウンロードスクリプト
=================================
個人PC（社外ネットワーク）で一度だけ実行してください。
必要なパッケージの自動インストールから行います。

ダウンロードされるモデル（合計 約10GB）:
  - RT-DETR v1        (~150MB)
  - RT-DETRv2         (~150MB)
  - Qwen2-VL 7B       (~9GB)  ← ローカルVLM（完全オフライン日本語説明）

実行方法:
    python3 download_models.py
"""

import sys, subprocess, os, ssl

# ── 必要パッケージを自動インストール ──
REQUIRED = [
    "urllib3", "requests", "huggingface-hub",
    "transformers", "torch", "torchvision",
    "accelerate", "qwen-vl-utils",
]
print("[1/3] 必要なパッケージを確認・インストールします...")
for pkg in REQUIRED:
    mod = pkg.replace("-", "_")
    try:
        __import__(mod)
    except ImportError:
        print(f"    インストール中: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])
print("    パッケージ OK")

# ── SSL 無効化 ──
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"]     = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
import urllib3; urllib3.disable_warnings()
import requests
_Orig = requests.Session
class _NoVerify(_Orig):
    def request(self, method, url, **kwargs):
        kwargs["verify"] = False
        return super().request(method, url, **kwargs)
requests.Session = _NoVerify

# ── モデルダウンロード ──
from transformers import (
    AutoImageProcessor, AutoModelForObjectDetection,
    Qwen2VLForConditionalGeneration, AutoProcessor,
)

MODELS = [
    ("RT-DETR v1  (~150MB)",  "detection",  "PekingU/rtdetr_r50vd"),
    ("RT-DETRv2   (~150MB)",  "detection",  "PekingU/rtdetr_v2_r50vd"),
    ("Qwen2-VL 7B (~9GB)  ",  "qwen2vl",    "Qwen/Qwen2-VL-7B-Instruct"),
]

print("\n[2/3] モデルをダウンロードします（合計 約10GB・時間がかかります）...")
all_ok = True
for label, kind, model_id in MODELS:
    print(f"\n    [{label}] {model_id}")
    try:
        if kind == "detection":
            AutoImageProcessor.from_pretrained(model_id)
            AutoModelForObjectDetection.from_pretrained(model_id)
        elif kind == "qwen2vl":
            AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype="auto", trust_remote_code=True
            )
        print(f"    OK")
    except Exception as e:
        print(f"    ERROR: {e}")
        all_ok = False

# ── 結果確認 ──
print("\n[3/3] キャッシュ確認...")
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
check_dirs = [
    "models--PekingU--rtdetr_r50vd",
    "models--PekingU--rtdetr_v2_r50vd",
    "models--Qwen--Qwen2-VL-7B-Instruct",
]
for d in check_dirs:
    path = os.path.join(cache_dir, d)
    status = "OK     " if os.path.isdir(path) else "MISSING"
    print(f"    [{status}] {d}")

print()
if all_ok:
    print("=" * 55)
    print(" ダウンロード完了！")
    print(f" キャッシュ場所: {cache_dir}")
    print(" 会社MacにAirDrop/USBでコピーしてください:")
    for d in check_dirs:
        print(f"   {d}")
    print("=" * 55)
else:
    print("一部エラーがありました。ネットワーク接続を確認して再実行してください。")
