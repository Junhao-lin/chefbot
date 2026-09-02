"""
鍋具受熱面混合融合遮罩生成器 (Wok Masking: Deep Learning + Geometric Prior Fusion)

演算法架構：
1. 深度學習分支 (Deep Learning YOLO-Seg):
   - 載入微調後的 3 類別 YOLO-Seg 權重
   - 即時預測黑鐵鍋工作受熱面之精細動態幾何輪廓 (適應晃鍋、視角傾斜、手部局部遮擋)
2. 傳統幾何先驗分支 (Classical Geometric Prior):
   - 霍夫圓變換 (Hough Circle) / 幾何先驗受熱圓形基準
3. 混合融合與時序濾波器 (Hybrid Fusion & Temporal EMA Smoother):
   - 融合策略: 當 YOLO-Seg 輸出高置信度多邊形時，與幾何圓進行形態學平滑聯集 (Morphological Union & Close)
   - 兜底保護: 若劇烈顛鍋或大面積工具遮擋導致置信度下降，先驗幾何圓自動平滑兜底
   - 輸出即時極致平滑、無噪訊跳閃的鍋具受熱工作區遮罩 (/wok/mask)
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
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "README.md").exists() and (parent / "requirements.txt").exists():
            return parent
    return Path.cwd()

project_root = get_project_root()

def extract_png_from_cdr(raw_bytes: bytes) -> np.ndarray:
    png_magic = b"\x89PNG\r\n\x1a\n"
    idx = raw_bytes.find(png_magic)
    if idx != -1:
        img_bytes = np.frombuffer(raw_bytes[idx:], dtype=np.uint8)
        return cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
    return None

class HybridWokMaskEstimator:
    def __init__(self, model_path: Path, conf_thresh: float = 0.30):
        self.model = YOLO(str(model_path))
        self.conf_thresh = conf_thresh
        self.ema_mask = None
        self.alpha = 0.25 # 時序指數平滑因子

    def get_geometric_prior_circle(self, h: int, w: int) -> np.ndarray:
        """傳統演算法先驗幾何圓 (Prior Geometric Circle)"""
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        center = (int(w * 0.52), int(h * 0.58))
        radius = int(min(h, w) * 0.43)
        cv2.circle(circle_mask, center, radius, 255, -1)
        return circle_mask

    def process_frame(self, bgr: np.ndarray) -> tuple:
        """
        融合深度學習預測與幾何先驗
        回傳: (fused_mask_u8, seg_mask_u8, prior_circle_u8)
        """
        h, w = bgr.shape[:2]
        prior_circle = self.get_geometric_prior_circle(h, w)

        # 1. 深度學習 YOLO-Seg 預測
        results = self.model.predict(bgr, conf=self.conf_thresh, verbose=False, device=0)[0]
        seg_mask = np.zeros((h, w), dtype=np.uint8)

        if results.masks is not None:
            for box, poly in zip(results.boxes, results.masks.xy):
                cls_id = int(box.cls[0])
                if cls_id == 0: # 0: Wok
                    pts = poly.astype(np.int32)
                    if len(pts) >= 3:
                        cv2.fillPoly(seg_mask, [pts], 255)

        # 2. 混合融合演算法 (Hybrid Ensemble)
        seg_area = np.sum(seg_mask > 0)
        if seg_area > (h * w * 0.15):
            # 模型檢測良好：以模型動態多邊形為主，與先驗圓形做形態學聯集與平滑閉合
            raw_fused = cv2.bitwise_or(seg_mask, cv2.bitwise_and(prior_circle, cv2.dilate(seg_mask, np.ones((7, 7), np.uint8))))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            fused_mask = cv2.morphologyEx(raw_fused, cv2.MORPH_CLOSE, kernel)
        else:
            # 遮擋或顛鍋漏檢：先驗幾何圓平滑兜底
            fused_mask = prior_circle.copy()

        # 3. 時序 EMA 平滑濾波 (Temporal Exponential Moving Average)
        if self.ema_mask is None:
            self.ema_mask = fused_mask.astype(np.float32)
        else:
            self.ema_mask = (1.0 - self.alpha) * self.ema_mask + self.alpha * fused_mask.astype(np.float32)

        final_mask = (self.ema_mask > 127).astype(np.uint8) * 255
        return final_mask, seg_mask, prior_circle

def generate_wok_mask_demo_video(bag_path: Path, model_path: Path, output_video: Path, max_frames: int = 0):
    output_video.parent.mkdir(parents=True, exist_ok=True)
    db_files = list(bag_path.glob("*.db3")) if bag_path.is_dir() else [bag_path]
    if not db_files:
        print(f"[ERROR] 找不到 .db3: {bag_path}")
        return

    db_path = db_files[0]
    print(f"\n================ 啟動鍋具 Mask 融合 (YOLO-Seg + 幾何圓) 展示 ================")
    print(f"來源 Bag : {db_path.name}")
    print(f"分割權重 : {model_path}")
    print(f"輸出影片 : {output_video.name}")

    estimator = HybridWokMaskEstimator(model_path)

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

    while True:
        row = cursor.fetchone()
        if row is None or (max_frames > 0 and frame_idx >= max_frames):
            break

        timestamp, rawdata = row
        rgbt_bgra = extract_png_from_cdr(rawdata)
        if rgbt_bgra is None:
            continue

        rgbt_rgba = cv2.cvtColor(rgbt_bgra, cv2.COLOR_BGRA2RGBA)
        rgb_raw = rgbt_rgba[:, :, :3]
        bgr = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        t_sec = frame_idx / 20.0

        # 計算融合 Mask
        final_fused_mask, raw_seg_mask, prior_circle = estimator.process_frame(bgr)

        # 建立 3 視窗橫向對比畫面 (1. 原始可見光, 2. 深度學習Seg + 幾何圓, 3. 最終融合 /wok/mask)
        panel1 = bgr.copy()
        
        panel2 = bgr.copy()
        # 繪製幾何先驗圓 (綠色虛線輪廓)
        prior_cnts, _ = cv2.findContours(prior_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel2, prior_cnts, -1, (0, 255, 0), 2)
        # 繪製 YOLO-Seg (藍色半透明)
        seg_overlay = panel2.copy()
        seg_overlay[raw_seg_mask > 0] = (255, 0, 0)
        panel2 = cv2.addWeighted(seg_overlay, 0.45, panel2, 0.55, 0)

        panel3 = bgr.copy()
        fused_overlay = panel3.copy()
        fused_overlay[final_fused_mask > 0] = (255, 200, 0)
        panel3 = cv2.addWeighted(fused_overlay, 0.40, panel3, 0.60, 0)
        fused_cnts, _ = cv2.findContours(final_fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel3, fused_cnts, -1, (255, 255, 0), 2)

        # Header 標籤
        headers = [
            f"1. Raw RGB Stream (t={t_sec:.1f}s)",
            "2. YOLO-Seg (Blue) + Prior Circle (Green)",
            "3. Final Fused /wok/mask (Smooth EMA)"
        ]
        panels = [panel1, panel2, panel3]
        for p, title in zip(panels, headers):
            cv2.rectangle(p, (0, 0), (w, 35), (20, 20, 20), -1)
            cv2.putText(p, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        combined = np.hstack(panels)

        if video_writer is None:
            gh, gw = combined.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_video), fourcc, 20.0, (gw, gh))
            print(f"[INIT] 建立橫向三分割展示畫布: {gw}x{gh} @ 20 FPS")

        video_writer.write(combined)
        frame_idx += 1

        if frame_idx % 300 == 0:
            percent = (frame_idx / total_msgs) * 100.0 if total_msgs > 0 else 0
            print(f"進度: [{frame_idx:>4}/{total_msgs}] ({percent:5.1f}%) | 耗時: {time.time()-t0:.1f}s")

    conn.close()
    if video_writer is not None:
        video_writer.release()

    print(f"\n================ 鍋具 Mask 融合影片生成完畢！ ================")
    print(f"展示影片路徑: {output_video.resolve()}\n")

def main():
    parser = argparse.ArgumentParser(description="鍋具 Masking: 深度學習 + 幾何先驗融合")
    parser.add_argument("--bag", type=str, default="data", help="Bag 路徑")
    parser.add_argument("--model", type=str, default="runs/segment/cooking_seg_3class_expert/weights/best.pt", help="YOLO-Seg 權重路徑")
    parser.add_argument("--output", type=str, default="data/wok_mask_fusion_demo.mp4", help="輸出影片路徑")
    parser.add_argument("--max_frames", type=int, default=0, help="最大幀數")
    args = parser.parse_args()

    bag_path = (project_root / args.bag).resolve() if not Path(args.bag).is_absolute() else Path(args.bag)
    model_path = (project_root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    out_video = (project_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    generate_wok_mask_demo_video(bag_path, model_path, out_video, args.max_frames)

if __name__ == "__main__":
    main()
