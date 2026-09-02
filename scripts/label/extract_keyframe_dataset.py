"""
方案 C: 分層時間段關鍵影格提取器 (Stratified Keyframe Extractor)

功能：
從 ROS2 Bag (data/*.db3) 全程 4400 幀 (223秒) 中，
依照 4 大關鍵料理時間段分層提取 36 張代表性關鍵影格，供手動多邊形標註工具使用：
1. 階段一：空鍋預熱與熱油潤鍋 (0s ~ 60s)    -> 採樣 6 幀
2. 階段二：生蛋入鍋與蛋白白化 (60s ~ 115s)  -> 採樣 12 幀
3. 階段三：翻面定型與顛鍋動態 (115s ~ 160s) -> 採樣 10 幀
4. 階段四：雙面熟化與起鍋裝盤 (160s ~ 223s) -> 採樣 8 幀
"""

import sys
import argparse
import sqlite3
from pathlib import Path
import numpy as np
import cv2

project_root = Path(__file__).resolve().parent.parent

def extract_png_from_cdr(raw_bytes: bytes) -> np.ndarray:
    png_magic = b"\x89PNG\r\n\x1a\n"
    idx = raw_bytes.find(png_magic)
    if idx != -1:
        img_bytes = np.frombuffer(raw_bytes[idx:], dtype=np.uint8)
        return cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
    return None

def extract_stratified_keyframes(bag_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    db_files = list(bag_path.glob("*.db3")) if bag_path.is_dir() else [bag_path]
    if not db_files:
        print(f"[ERROR] 找不到 .db3: {bag_path}")
        return

    db_path = db_files[0]
    print(f"\n================ 啟動分層時間段關鍵影格提取 ================")
    print(f"來源 Bag : {db_path.name}")
    print(f"輸出目錄 : {output_dir}")

    # 4 大階段的精準採樣時間戳 (秒)
    # 階段 1 (0~60s): 6 張
    stage1_t = np.linspace(0.0, 58.0, 6)
    # 階段 2 (60~115s): 12 張 (密集涵蓋 62s 倒蛋、擴散與白化)
    stage2_t = np.linspace(61.0, 114.0, 12)
    # 階段 3 (115~160s): 10 張 (密集涵蓋 118s 翻面、顛鍋與空中姿態)
    stage3_t = np.linspace(116.0, 158.0, 10)
    # 階段 4 (160~223s): 8 張 (涵蓋熟化與起鍋)
    stage4_t = np.linspace(162.0, 220.0, 8)

    all_target_seconds = np.concatenate([stage1_t, stage2_t, stage3_t, stage4_t])
    # 轉換為 20 FPS 下的幀序號 (共 36 幀)
    target_frame_indices = set([int(round(s * 20.0)) for s in all_target_seconds])

    print(f"總計規劃提取: {len(target_frame_indices)} 張分層關鍵影格")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM topics WHERE name LIKE '%compressed%'")
    target_id = cursor.fetchone()[0]
    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC", (target_id,))

    frame_idx = 0
    saved_count = 0

    while True:
        row = cursor.fetchone()
        if row is None:
            break

        if frame_idx in target_frame_indices:
            timestamp, rawdata = row
            rgbt_bgra = extract_png_from_cdr(rawdata)
            if rgbt_bgra is not None:
                rgbt_rgba = cv2.cvtColor(rgbt_bgra, cv2.COLOR_BGRA2RGBA)
                rgb_raw = rgbt_rgba[:, :, :3]
                bgr = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)

                out_name = f"frame_{frame_idx:05d}.jpg"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), bgr)
                t_sec = frame_idx / 20.0
                print(f"[{saved_count+1:02d}/36] 已儲存關鍵幀: {out_name} (t = {t_sec:5.1f}s)")
                saved_count += 1

        frame_idx += 1

    conn.close()
    print(f"\n================ 關鍵影格提取完畢！共儲存 {saved_count} 張 ================\n")

def main():
    parser = argparse.ArgumentParser(description="提取分層時間段關鍵影格")
    parser.add_argument("--bag", type=str, default="../data", help="Bag 路徑")
    parser.add_argument("--output", type=str, default="../data/dataset_seg_manual/images", help="圖片輸出目錄")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    bag_path = project_root / args.bag
    out_dir = project_root / args.output

    extract_stratified_keyframes(bag_path, out_dir)

if __name__ == "__main__":
    main()
