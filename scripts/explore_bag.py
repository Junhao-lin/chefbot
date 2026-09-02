"""
ROS2 RGBT Bag 結構探索與影格提取驗證腳本
功能：
1. 檢視 ROS2 Bag 的 topics、訊息型態、訊息數量與時長
2. 解碼 RGBT CompressedImage (4-channel PNG / BGRA to RGBA)
3. 拆分 RGB (前 3 channels) 與 Thermal (第 4 channel)
4. 儲存範例影格並輸出統計資訊（通道值範圍、解析度等）
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import sqlite3
def extract_png_from_cdr(raw_bytes: bytes) -> np.ndarray:
    """從 CDR 二進位資料中定位 PNG 標頭並解碼"""
    png_magic = b"\x89PNG\r\n\x1a\n"
    idx = raw_bytes.find(png_magic)
    if idx != -1:
        img_bytes = np.frombuffer(raw_bytes[idx:], dtype=np.uint8)
        return cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
    return None

def explore_bag(bag_path: Path, output_dir: Path, max_frames: int = 10):
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_out = output_dir / "rgb"
    thermal_out = output_dir / "thermal"
    rgb_out.mkdir(exist_ok=True)
    thermal_out.mkdir(exist_ok=True)

    db_files = list(bag_path.glob("*.db3")) if bag_path.is_dir() else [bag_path]
    if not db_files:
        print(f"[ERROR] 找不到 .db3 檔案：{bag_path}")
        return

    db_path = db_files[0]
    print(f"\n================ 探索 ROS2 Bag: {db_path.name} ================")


    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, type FROM topics")
    topics = cursor.fetchall()
    print("\n--- 包含的 Topics 清單 ---")
    target_id = None
    for tid, name, msg_type in topics:
        cursor.execute("SELECT COUNT(*) FROM messages WHERE topic_id = ?", (tid,))
        count = cursor.fetchone()[0]
        print(f"Topic: {name:<35} | Type: {msg_type} | Count: {count}")
        if "compressed" in name or "compressed" in msg_type.lower():
            target_id = tid

    if target_id is None:
        target_id = topics[0][0]

    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC", (target_id,))

    frame_count = 0
    thermal_stats = []

    while True:
        row = cursor.fetchone()
        if row is None:
            break

        timestamp, rawdata = row
        rgbt_bgra = extract_png_from_cdr(rawdata)
        if rgbt_bgra is None:
            continue

        # BGRA -> RGBA
        rgbt_rgba = cv2.cvtColor(rgbt_bgra, cv2.COLOR_BGRA2RGBA)
        rgb = rgbt_rgba[:, :, :3]
        thermal = rgbt_rgba[:, :, 3]

        t_min, t_max, t_mean = thermal.min(), thermal.max(), thermal.mean()
        thermal_stats.append((t_min, t_max, t_mean))

        if frame_count < max_frames:
            bgr_save = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(rgb_out / f"frame_{frame_count:05d}.jpg"), bgr_save)
            cv2.imwrite(str(thermal_out / f"thermal_raw_{frame_count:05d}.png"), thermal)
            thermal_colored = cv2.applyColorMap(thermal, cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(thermal_out / f"thermal_inferno_{frame_count:05d}.jpg"), thermal_colored)

        frame_count += 1
        if frame_count % 200 == 0:
            print(f"[進度] 已分析 {frame_count} 幀影像...")

    conn.close()
    print(f"\n[完成] 成功解析 {frame_count} 幀影像！")
    print(f"[輸出] 範例影像已儲存至：{output_dir}")
    if thermal_stats:
        overall_min = min(s[0] for s in thermal_stats)
        overall_max = max(s[1] for s in thermal_stats)
        overall_mean = np.mean([s[2] for s in thermal_stats])
        print(f"[Thermal 統計] 全局最小值: {overall_min}, 全局最大值: {overall_max}, 平均值: {overall_mean:.2f}")

def main():
    parser = argparse.ArgumentParser(description="探索與解析 RGBT ROS2 Bag")
    parser.add_argument("--bag", type=str, default="data", help="ROS2 Bag 資料夾路徑")
    parser.add_argument("--output", type=str, default="data/extracted_frames", help="抽幀輸出路徑")
    parser.add_argument("--max_frames", type=int, default=20000, help="最多儲存抽樣幀數")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    bag_path = project_root / args.bag
    output_dir = project_root / args.output

    # 自動尋找 bag 檔案或包含 metadata.yaml 的資料夾
    if not bag_path.exists():
        print(f"[ERROR] 找不到路徑：{bag_path}")
        return

    # 若傳入的是包含 bag 的資料夾，尋找第一個有效 bag
    candidates = list(bag_path.glob("**/metadata.yaml"))
    if candidates:
        target_bag = candidates[0].parent
    else:
        target_bag = bag_path

    explore_bag(target_bag, output_dir, args.max_frames)

if __name__ == "__main__":
    main()
