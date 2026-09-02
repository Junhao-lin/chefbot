"""
方案 B: 多模態視覺語言大模型 (VLM) 輔助打標工具 (VLM Auto Annotator)

原理：
1. 提取關鍵影格並進行 Base64 編碼
2. 調用多模態大模型 (Qwen2.5-VL / GPT-4o) 的 Zero-shot 物件偵測與邊界框提取能力
3. 提示詞要求輸出 3 類別 (wok, egg, container) 之正規化坐標
4. 自動轉換為 YOLO-Seg 多邊形標籤格式

評測結論：
語意識別能力極強 (能精確定位生蛋與熟蛋)，但輸出之多邊形頂點數較少 (多為 4 頂點 Bounding Box)，
無法貼合非剛性液態邊緣，幾何擬合精度不足。
"""

import sys
import os
import json
import base64
import argparse
from pathlib import Path
import cv2
import requests

# 取得專案根目錄 (位於 scripts/label/ 的上兩層)
project_root = Path(__file__).resolve().parent.parent.parent

def encode_image_to_base64(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_vlm_for_annotation(img_path: Path, api_key: str, model: str = "qwen/qwen-2.5-vl-72b-instruct") -> str:
    """呼叫 OpenRouter / OpenAI API 進行 VLM 物件定位"""
    base64_img = encode_image_to_base64(img_path)
    
    prompt = """You are an expert annotator for cooking robot perception.
Detect and segment the following 3 classes:
- 0: wok (the circular cooking surface of the black wok)
- 1: egg (the fried egg or raw egg inside wok or container)
- 2: container (the small bowl at top-right)

Return ONLY a valid JSON list of objects:
[
  {"class_id": 0, "polygon": [x1, y1, x2, y2, x3, y3, x4, y4, ...]},
  {"class_id": 1, "polygon": [x1, y1, x2, y2, x3, y3, ...]}
]
All coordinates MUST be normalized between 0.0 and 1.0. Do NOT output markdown code fences (like ```json), return raw JSON only."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }
        ],
        "temperature": 0.1
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
        res_json = response.json()
        if "choices" in res_json and len(res_json["choices"]) > 0:
            return res_json["choices"][0]["message"]["content"]
        else:
            print(f"[WARN] API 回傳無效格式: {res_json}")
            return ""
    except Exception as e:
        print(f"[WARN] VLM 請求失敗 ({img_path.name}): {e}")
        return ""

def parse_vlm_response_to_yolo(vlm_text: str, out_label_path: Path):
    """解析 VLM 返回之 JSON 並寫入標準 YOLO-Seg 標籤格式"""
    if not vlm_text:
        return 0
    try:
        clean_text = vlm_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        data = json.loads(clean_text)
        lines = []
        for obj in data:
            cls_id = int(obj.get("class_id", 0))
            poly = obj.get("polygon", [])
            if len(poly) >= 6: # 至少 3 個點 (6 個座標)
                coords_str = " ".join([f"{float(c):.5f}" for c in poly])
                lines.append(f"{cls_id} {coords_str}\n")

        with open(out_label_path, "w") as f:
            f.writelines(lines)
        return len(lines)
    except Exception as e:
        print(f"[ERROR] 解析 VLM JSON 失敗: {e}\n原始輸出: {vlm_text}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="方案 B: VLM 多模態大模型輔助打標工具")
    parser.add_argument("--images", type=str, default="../data/dataset_seg_vlm/images", help="圖片目錄")
    parser.add_argument("--labels", type=str, default="../data/dataset_seg_vlm/labels", help="標籤輸出目錄")
    parser.add_argument("--api_key", type=str, default="", help="OpenRouter API Key (或設定環境變數 OPENROUTER_API_KEY)")
    parser.add_argument("--model", type=str, default="qwen/qwen-2.5-vl-72b-instruct", help="VLM 模型名稱")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")

    # 解析路徑
    img_dir = Path(args.images)
    if not img_dir.is_absolute():
        img_dir = (Path.cwd() / args.images).resolve()

    lbl_dir = Path(args.labels)
    if not lbl_dir.is_absolute():
        lbl_dir = (Path.cwd() / args.labels).resolve()

    lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not img_files:
        print(f"[ERROR] 在 {img_dir} 找不到任何圖片！")
        return

    print(f"\n================ 啟動方案 B: VLM 多模態自動打標 ================")
    print(f"圖片目錄: {img_dir}")
    print(f"標籤目錄: {lbl_dir}")
    print(f"待處理數: {len(img_files)} 張影格")
    print(f"VLM 模型: {args.model}")

    if not api_key:
        print("\n[WARN] 未提供 --api_key 且未檢測到 OPENROUTER_API_KEY 環境變數！")
        print("[INFO] 範例指令: python scripts/label/annotate_with_vlm.py --api_key sk-or-v1-xxx")
        return

    success_count = 0
    for idx, img_p in enumerate(img_files):
        out_lbl = lbl_dir / f"{img_p.stem}.txt"
        print(f"[{idx+1:02d}/{len(img_files)}] 正在請求 VLM 標註: {img_p.name}...")
        
        vlm_resp = call_vlm_for_annotation(img_p, api_key, args.model)
        num_objs = parse_vlm_response_to_yolo(vlm_resp, out_lbl)
        
        if num_objs > 0:
            print(f" -> 已儲存標籤: {out_lbl.name} (含 {num_objs} 個物件)")
            success_count += 1
        else:
            print(f" -> 標註失敗或未偵測到物件")

    print(f"\n================ VLM 標註完成！成功標註 {success_count}/{len(img_files)} 張 ================\n")

if __name__ == "__main__":
    main()
