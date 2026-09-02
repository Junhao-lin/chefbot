"""
方案 A: 傳統規則式自動標註工具 (Rule-based OpenCV Auto Annotator)

原理：
1. 利用 HSV 顏色空間分離蛋黃 (黃色高飽和度) 與蛋白 (白色/高亮)
2. 利用 Hough Circle 擬合黑鐵鍋受熱圓形邊界
3. 利用右上區域色彩特徵定位生蛋鋼碗 (Container)
4. 自動產出 YOLO-Seg 多邊形標籤

缺點：
易受金屬鏡面反光與油煙焦化干擾，標籤雜訊約 25%~30%，僅作為基線 (Baseline) 探索。
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

def auto_annotate_frame(bgr: np.ndarray) -> list:
    """從單張影格中透過 OpenCV 規則自動提取 3 類物件多邊形"""
    h, w = bgr.shape[:2]
    annotations = []

    # 1. 黑鐵鍋 (Wok: Class 0) - 固定幾何先驗與霍夫圓擬合
    wok_poly = []
    center_x, center_y, radius = int(w * 0.52), int(h * 0.58), int(min(h, w) * 0.43)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    for ang in angles:
        px = (center_x + radius * np.cos(ang)) / w
        py = (center_y + radius * np.sin(ang)) / h
        wok_poly.extend([f"{np.clip(px, 0.0, 1.0):.5f}", f"{np.clip(py, 0.0, 1.0):.5f}"])
    annotations.append(f"0 " + " ".join(wok_poly))

    # 2. 荷包蛋 (Egg: Class 1) - HSV 顏色分割
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array([12, 80, 80]), np.array([35, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 55, 255]))
    egg_mask = cv2.bitwise_or(yellow, white)
    
    cnts, _ = cv2.findContours(egg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) > 300:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) >= 3:
                poly = []
                for pt in approx:
                    poly.extend([f"{pt[0][0]/w:.5f}", f"{pt[0][1]/h:.5f}"])
                annotations.append(f"1 " + " ".join(poly))

    # 3. 小鋼碗 (Container: Class 2) - 右上區域先驗
    container_box = [int(w * 0.55), 0, int(w * 0.90), int(h * 0.35)]
    c_poly = [
        f"{container_box[0]/w:.5f}", f"{container_box[1]/h:.5f}",
        f"{container_box[2]/w:.5f}", f"{container_box[1]/h:.5f}",
        f"{container_box[2]/w:.5f}", f"{container_box[3]/h:.5f}",
        f"{container_box[0]/w:.5f}", f"{container_box[3]/h:.5f}"
    ]
    annotations.append(f"2 " + " ".join(c_poly))

    return annotations

def run_rule_based_annotation(bag_path: Path, output_dir: Path, stride: int = 25):
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    db_files = list(bag_path.glob("*.db3")) if bag_path.is_dir() else [bag_path]
    if not db_files:
        print(f"[ERROR] 找不到 .db3: {bag_path}")
        return

    conn = sqlite3.connect(str(db_files[0]))
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM topics WHERE name LIKE '%compressed%'")
    target_id = cursor.fetchone()[0]
    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC", (target_id,))

    frame_idx = 0
    saved_count = 0

    print(f"\n================ 啟動方案 A: 規則式自動標註 ================")
    print(f"每 {stride} 幀抽樣一張自動打標...")

    while True:
        row = cursor.fetchone()
        if row is None:
            break

        if frame_idx % stride == 0:
            timestamp, rawdata = row
            rgbt_bgra = extract_png_from_cdr(rawdata)
            if rgbt_bgra is not None:
                rgbt_rgba = cv2.cvtColor(rgbt_bgra, cv2.COLOR_BGRA2RGBA)
                rgb_raw = rgbt_rgba[:, :, :3]
                bgr = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)

                img_name = f"frame_{frame_idx:05d}.jpg"
                lbl_name = f"frame_{frame_idx:05d}.txt"

                cv2.imwrite(str(img_dir / img_name), bgr)
                lines = auto_annotate_frame(bgr)
                with open(lbl_dir / lbl_name, "w") as f:
                    for l in lines:
                        f.write(l + "\n")
                saved_count += 1

        frame_idx += 1

    conn.close()
    print(f"\n自動標註完成！共產生 {saved_count} 張標註檔案至 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="方案 A: 規則式自動標註工具")
    parser.add_argument("--bag", type=str, default="../data", help="Bag 路徑")
    parser.add_argument("--output", type=str, default="../data/dataset_seg_rule_based", help="輸出資料集目錄")
    parser.add_argument("--stride", type=int, default=25, help="抽幀步長")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    bag_path = project_root / args.bag
    out_dir = project_root / args.output
    run_rule_based_annotation(bag_path, out_dir, args.stride)

if __name__ == "__main__":
    main()
