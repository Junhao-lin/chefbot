"""
RGBT + Wok Mask 綜合多模態 Dashboard 展示影片生成器 (RGB + Thermal + Mask 4-in-1 Generator)

功能：
生成包含 4 大畫面的出版級 2x2 綜合監控面板影片 (data/rgbt_mask_dashboard.mp4)：
1. 左上: 原始可見光 RGB (/camera/rgb/image_raw)
2. 右上: Method 3 熱梯度等溫線 HUD 注入影像 (/camera/thermal/enhanced) - 最佳選用方案
3. 左下: 即時鍋具工作受熱面遮罩 (/wok/mask)
4. 右下: 綜合多模態融合畫面 (RGB + 等溫線 HUD + /wok/mask 輪廓)
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

def apply_method_3_thermal_gradient_hud(bgr: np.ndarray, thermal_raw: np.ndarray, wok_mask: np.ndarray) -> np.ndarray:
    """Method 3: 熱梯度等溫線 HUD 注入 (鍋蛋反差最鮮明、零破壞 RGB 自然色澤)"""
    t_blur = cv2.GaussianBlur(thermal_raw, (5, 5), 0)
    gx = cv2.Sobel(t_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(t_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_norm = np.clip(mag / 25.0, 0.0, 1.0)
    
    edge_color = cv2.applyColorMap((mag_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = bgr.copy()
    edge_mask = (mag_norm > 0.15) & (wok_mask > 0)
    out[edge_mask] = cv2.addWeighted(bgr[edge_mask], 0.35, edge_color[edge_mask], 0.65, 0)
    return out

def generate_dashboard_video(bag_path: Path, model_path: Path, output_video: Path, max_frames: int = 0):
    output_video.parent.mkdir(parents=True, exist_ok=True)
    db_files = list(bag_path.glob("*.db3")) if bag_path.is_dir() else [bag_path]
    if not db_files:
        print(f"[ERROR] 找不到 .db3: {bag_path}")
        return

    db_path = db_files[0]
    print(f"\n================ 啟動 RGBT + Mask 綜合多模態影片生成 ================")
    print(f"來源 Bag : {db_path.name}")
    print(f"模型權重 : {model_path}")
    print(f"輸出影片 : {output_video.name}")

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
    ema_mask = None
    alpha = 0.25

    while True:
        row = cursor.fetchone()
        if row is None or (max_frames > 0 and frame_idx >= max_frames):
            break

        timestamp, rawdata = row
        t_sec = frame_idx / 20.0

        rgbt_bgra = extract_png_from_cdr(rawdata)
        if rgbt_bgra is None:
            continue

        rgbt_rgba = cv2.cvtColor(rgbt_bgra, cv2.COLOR_BGRA2RGBA)
        rgb_raw = rgbt_rgba[:, :, :3]
        thermal_raw = rgbt_rgba[:, :, 3]
        bgr = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        # 1. 計算 YOLO-Seg 與融合 Mask
        results = model.predict(bgr, conf=0.30, verbose=False, device=0)[0]
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        if results.masks is not None:
            for box, poly in zip(results.boxes, results.masks.xy):
                if int(box.cls[0]) == 0:
                    pts = poly.astype(np.int32)
                    if len(pts) >= 3:
                        cv2.fillPoly(seg_mask, [pts], 255)

        prior_circle = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(prior_circle, (int(w * 0.52), int(h * 0.58)), int(min(h, w) * 0.43), 255, -1)

        if np.sum(seg_mask > 0) > (h * w * 0.15):
            raw_fused = cv2.bitwise_or(seg_mask, cv2.bitwise_and(prior_circle, cv2.dilate(seg_mask, np.ones((7, 7), np.uint8))))
            fused_mask = cv2.morphologyEx(raw_fused, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        else:
            fused_mask = prior_circle.copy()

        if ema_mask is None:
            ema_mask = fused_mask.astype(np.float32)
        else:
            ema_mask = (1.0 - alpha) * ema_mask + alpha * fused_mask.astype(np.float32)
        wok_mask = (ema_mask > 127).astype(np.uint8) * 255

        # 2. 生成 4 大面板
        # 面板 1: 原圖 RGB
        p1 = bgr.copy()

        # 面板 2: Method 3 熱梯度等溫線 HUD 注入 (替換原先的 Turbo 全域覆蓋)
        p2 = apply_method_3_thermal_gradient_hud(bgr, thermal_raw, wok_mask)

        # 面板 3: 純黑底綠色 /wok/mask
        p3 = np.zeros((h, w, 3), dtype=np.uint8)
        p3[wok_mask > 0] = (0, 220, 0)

        # 面板 4: 綜合多模態融合畫面 (RGB + Thermal HUD + Mask 輪廓)
        p4 = bgr.copy()
        mask_layer = p4.copy()
        mask_layer[wok_mask > 0] = (255, 200, 0)
        p4 = cv2.addWeighted(mask_layer, 0.35, p4, 0.65, 0)
        cnts, _ = cv2.findContours(wok_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p4, cnts, -1, (0, 255, 255), 2)

        # 等溫線梯度 HUD 注入
        t_blur = cv2.GaussianBlur(thermal_raw, (5, 5), 0)
        gx = cv2.Sobel(t_blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(t_blur, cv2.CV_32F, 0, 1, ksize=3)
        mag_norm = np.clip(np.sqrt(gx**2 + gy**2) / 25.0, 0.0, 1.0)
        edge_color = cv2.applyColorMap((mag_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        edge_mask = (mag_norm > 0.18) & (wok_mask > 0)
        p4[edge_mask] = cv2.addWeighted(p4[edge_mask], 0.30, edge_color[edge_mask], 0.70, 0)

        # 面板 Header
        titles = [
            f"1. RGB Stream (/camera/rgb/image_raw) | t={t_sec:.1f}s",
            "2. Thermal Gradient Edge HUD (/camera/thermal/enhanced)",
            "3. Realtime Wok Mask (/wok/mask)",
            "4. Multi-modal Fusion Dashboard"
        ]
        panels = [p1, p2, p3, p4]
        for p, title in zip(panels, titles):
            cv2.rectangle(p, (0, 0), (w, 35), (20, 20, 20), -1)
            cv2.putText(p, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

        # 拼裝 2x2 畫布
        row1 = np.hstack([panels[0], panels[1]])
        row2 = np.hstack([panels[2], panels[3]])
        grid = np.vstack([row1, row2])

        if video_writer is None:
            gh, gw = grid.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_video), fourcc, 20.0, (gw, gh))
            print(f"[INIT] 建立 2x2 多模態綜合畫布: {gw}x{gh} @ 20 FPS")

        video_writer.write(grid)
        frame_idx += 1

        if frame_idx % 300 == 0:
            percent = (frame_idx / total_msgs) * 100.0 if total_msgs > 0 else 0
            print(f"進度: [{frame_idx:>4}/{total_msgs}] ({percent:5.1f}%) | 耗時: {time.time()-t0:.1f}s")

    conn.close()
    if video_writer is not None:
        video_writer.release()

    print(f"\n================ 綜合多模態展示影片生成完畢！ ================")
    print(f"影片路徑: {output_video.resolve()}\n")

def main():
    parser = argparse.ArgumentParser(description="RGBT + Wok Mask 綜合多模態 Dashboard 生成器")
    parser.add_argument("--bag", type=str, default="data", help="Bag 路徑")
    parser.add_argument("--model", type=str, default="runs/segment/cooking_seg_3class_expert/weights/best.pt", help="YOLO-Seg 權重路徑")
    parser.add_argument("--output", type=str, default="data/rgbt_mask_dashboard.mp4", help="輸出影片路徑")
    parser.add_argument("--max_frames", type=int, default=0, help="最大幀數")
    args = parser.parse_args()

    bag_path = (project_root / args.bag).resolve() if not Path(args.bag).is_absolute() else Path(args.bag)
    model_path = (project_root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    out_video = (project_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    generate_dashboard_video(bag_path, model_path, out_video, args.max_frames)

if __name__ == "__main__":
    main()
