"""
3 類別 YOLO-Seg 即時推論與展示影片生成器 (3-Class YOLO-Seg Inference Video Generator)

類別定義：
- 0: Wok (藍色) - 鍋具工作受熱面
- 1: Egg (黃色) - 荷包蛋 (生蛋、蛋白白化、翻面焦黃)
- 2: Container (洋紅色) - 右上生蛋小鋼碗

功能：
1. 讀取 ROS2 Bag 原始影格
2. 載入微調後的 YOLO-Seg 最佳權重
3. 繪製半透明 Mask、多邊形輪廓線與置信度標籤
4. 即時統計推論延遲 (Latency) 與 FPS
5. 輸出展示影片至 data/seg_inference_3class_expert.mp4
"""

import sys
import time
import argparse
import sqlite3
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

def get_project_root() -> Path:
    """自動定位 Final 專案根目錄"""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "README.md").exists() and (parent / "requirements.txt").exists():
            return parent
    return Path.cwd()

project_root = get_project_root()

CLASS_COLORS = {
    0: (255, 0, 0),     # Wok (藍)
    1: (0, 255, 255),   # Egg (黃)
    2: (255, 0, 255)    # Container (洋紅)
}
CLASS_NAMES = {0: "Wok", 1: "Egg", 2: "Container"}

def extract_png_from_cdr(raw_bytes: bytes) -> np.ndarray:
    png_magic = b"\x89PNG\r\n\x1a\n"
    idx = raw_bytes.find(png_magic)
    if idx != -1:
        img_bytes = np.frombuffer(raw_bytes[idx:], dtype=np.uint8)
        return cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
    return None

def run_3class_inference(bag_path: Path, model_path: Path, output_video: Path, conf_thresh: float = 0.30, max_frames: int = 0):
    output_video.parent.mkdir(parents=True, exist_ok=True)
    db_files = list(bag_path.glob("*.db3")) if bag_path.is_dir() else [bag_path]
    if not db_files:
        print(f"[ERROR] 找不到 .db3 資料庫: {bag_path}")
        return

    db_path = db_files[0]
    print(f"\n================ 啟動 3 類別 YOLO-Seg 模型推論 ================")
    print(f"模型權重: {model_path}")
    print(f"來源 Bag : {db_path.name}")
    print(f"輸出影片: {output_video.name}")
    print(f"核心類別: 0: Wok, 1: Egg, 2: Container")

    model = YOLO(str(model_path))

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM topics WHERE name LIKE '%compressed%'")
    target_id = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM messages WHERE topic_id = ?", (target_id,))
    total_msgs = cursor.fetchone()[0]
    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC", (target_id,))

    video_writer = None
    frame_idx = 0
    t0 = time.time()
    latencies = []

    while True:
        row = cursor.fetchone()
        if row is None or (max_frames > 0 and frame_idx >= max_frames):
            break

        timestamp, rawdata = row
        t_start = time.perf_counter()

        rgbt_bgra = extract_png_from_cdr(rawdata)
        if rgbt_bgra is None:
            continue

        rgbt_rgba = cv2.cvtColor(rgbt_bgra, cv2.COLOR_BGRA2RGBA)
        rgb_raw = rgbt_rgba[:, :, :3]
        bgr_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
        h, w = bgr_raw.shape[:2]

        canvas = bgr_raw.copy()

        # 模型推論
        results = model.predict(bgr_raw, conf=conf_thresh, verbose=False, device=0)[0]
        
        t_end = time.perf_counter()
        lat_ms = (t_end - t_start) * 1000.0
        latencies.append(lat_ms)

        # 繪製推論 Mask 與標籤
        if results.masks is not None:
            for box, poly in zip(results.boxes, results.masks.xy):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                pts = poly.astype(np.int32)
                color = CLASS_COLORS.get(cls_id, (0, 255, 0))
                cname = CLASS_NAMES.get(cls_id, f"Class {cls_id}")

                overlay = canvas.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.40, canvas, 0.60, 0, canvas)
                cv2.polylines(canvas, [pts], True, color, 2)

                if len(pts) > 0:
                    top_pt = pts[np.argmin(pts[:, 1])]
                    label_str = f"{cname} {conf:.2f}"
                    cv2.putText(canvas, label_str, (top_pt[0], max(top_pt[1]-6, 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(canvas, label_str, (top_pt[0], max(top_pt[1]-6, 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # 頂部 HUD
        t_sec = frame_idx / 20.0
        hud_bg = canvas.copy()
        cv2.rectangle(hud_bg, (0, 0), (w, 40), (20, 20, 20), -1)
        canvas = cv2.addWeighted(hud_bg, 0.75, canvas, 0.25, 0)

        cur_fps = 1000.0 / lat_ms if lat_ms > 0 else 0
        info_left = f"YOLO-Seg Expert | Frame: {frame_idx:04d} ({t_sec:.1f}s)"
        info_right = f"Latency: {lat_ms:.1f}ms ({cur_fps:.0f} FPS)"
        cv2.putText(canvas, info_left, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.putText(canvas, info_right, (w - 230, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)

        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_video), fourcc, 20.0, (w, h))

        video_writer.write(canvas)
        frame_idx += 1

        if frame_idx % 300 == 0:
            percent = (frame_idx / total_msgs) * 100.0 if total_msgs > 0 else 0
            avg_lat = np.mean(latencies[-300:])
            print(f"推論進度: [{frame_idx:>4}/{total_msgs}] ({percent:5.1f}%) | 平均延遲: {avg_lat:.1f}ms ({1000.0/avg_lat:.0f} FPS)")

    conn.close()
    if video_writer is not None:
        video_writer.release()

    overall_lat = np.mean(latencies) if latencies else 0
    print(f"\n推論完成！共 {frame_idx} 幀，耗時: {time.time()-t0:.1f} 秒")
    print(f"全域平均延遲: {overall_lat:.2f} ms ({1000.0/overall_lat:.1f} FPS)")
    print(f"成果影片路徑: {output_video.resolve()}\n")

def main():
    parser = argparse.ArgumentParser(description="3 類別 YOLO-Seg 推論展示")
    parser.add_argument("--bag", type=str, default="data", help="Bag 目錄")
    parser.add_argument("--model", type=str, default="runs/segment/cooking_seg_3class_expert/weights/best.pt", help="權重路徑")
    parser.add_argument("--output", type=str, default="data/seg_inference_3class_expert.mp4", help="輸出影片路徑")
    parser.add_argument("--conf", type=float, default=0.30, help="置信度")
    parser.add_argument("--max_frames", type=int, default=0, help="最大幀數")
    args = parser.parse_args()

    bag_p = Path(args.bag)
    bag_path = bag_p if bag_p.is_absolute() else (project_root / args.bag).resolve()

    model_p = Path(args.model)
    model_path = model_p if model_p.is_absolute() else (project_root / args.model).resolve()

    out_p = Path(args.output)
    out_video = out_p if out_p.is_absolute() else (project_root / args.output).resolve()

    run_3class_inference(bag_path, model_path, out_video, args.conf, args.max_frames)

if __name__ == "__main__":
    main()
