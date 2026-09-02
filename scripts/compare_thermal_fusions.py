"""
Thermal 影像處理與 RGBT 多模態融合雙影片生成器 (Pure Thermal & RGBT Fusion Benchmark)

功能：
生成兩部獨立的 2x3 六分割對比展示影片：
1. 【影片一：純 Thermal 處理對比】(data/thermal_pure_benchmark.mp4)
   - 6 種不同的熱成像正規化、偽彩色與梯度場處理（不與 RGB 疊合）
2. 【影片二：RGBT 多模態融合對比】(data/thermal_fusion_benchmark.mp4)
   - 6 種熱成像校正結果與可見光 RGB 的多模態疊合效果（含最佳的 Method 3: Thermal Gradient HUD）
"""

import sys
import time
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

# ==========================================
# 6 種純 Thermal 影像處理演算法 (Pure Thermal)
# ==========================================
def pure_method_1_wok_masked(thermal_raw: np.ndarray, wok_mask: np.ndarray) -> np.ndarray:
    """1. 鍋內局部量程 (Wok-Only Masked Inferno)"""
    norm = np.clip((thermal_raw.astype(np.float32) - 120.0) / (220.0 - 120.0), 0.0, 1.0)
    t_color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    out = np.zeros_like(t_color)
    out[wok_mask > 0] = t_color[wok_mask > 0]
    return out

def pure_method_2_calibrated_turbo(thermal_raw: np.ndarray) -> np.ndarray:
    """2. 物理量程校準 (Calibrated Turbo Range: 130~215)"""
    norm = np.clip((thermal_raw.astype(np.float32) - 130.0) / (215.0 - 130.0), 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

def pure_method_3_gradient_field(thermal_raw: np.ndarray, wok_mask: np.ndarray) -> np.ndarray:
    """3. 熱梯度等溫線場 (Thermal Gradient Field)"""
    t_blur = cv2.GaussianBlur(thermal_raw, (5, 5), 0)
    gx = cv2.Sobel(t_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(t_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_norm = np.clip(mag / 25.0, 0.0, 1.0)
    edge_color = cv2.applyColorMap((mag_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = np.zeros_like(edge_color)
    out[wok_mask > 0] = edge_color[wok_mask > 0]
    return out

def pure_method_4_bilateral_smooth(thermal_raw: np.ndarray) -> np.ndarray:
    """4. 雙邊濾波保邊平滑 (Bilateral Smoothed Magma)"""
    norm = np.clip((thermal_raw.astype(np.float32) - 120.0) / (220.0 - 120.0), 0.0, 1.0)
    t_norm = (norm * 255).astype(np.uint8)
    t_smooth = cv2.bilateralFilter(t_norm, d=9, sigmaColor=75, sigmaSpace=75)
    return cv2.applyColorMap(t_smooth, cv2.COLORMAP_MAGMA)

def pure_method_5_clahe_magma(thermal_raw: np.ndarray) -> np.ndarray:
    """5. 自適應局部直方圖增強 (Adaptive CLAHE + Magma)"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    t_clahe = clahe.apply(thermal_raw)
    return cv2.applyColorMap(t_clahe, cv2.COLORMAP_MAGMA)

def pure_method_6_baseline(thermal_raw: np.ndarray) -> np.ndarray:
    """6. 全域基準量程 (Baseline Fixed Range Inferno)"""
    return cv2.applyColorMap(thermal_raw, cv2.COLORMAP_INFERNO)


# ==========================================
# 6 種 RGBT 多模態融合演算法 (RGB + Thermal)
# ==========================================
def fusion_method_1_wok_masked(bgr: np.ndarray, t_pure1: np.ndarray, wok_mask: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    fused = cv2.addWeighted(bgr, 0.60, t_pure1, 0.40, 0)
    out[wok_mask > 0] = fused[wok_mask > 0]
    return out

def fusion_method_2_calibrated_turbo(bgr: np.ndarray, t_pure2: np.ndarray, wok_mask: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    fused = cv2.addWeighted(bgr, 0.65, t_pure2, 0.35, 0)
    out[wok_mask > 0] = fused[wok_mask > 0]
    return out

def fusion_method_3_gradient_hud(bgr: np.ndarray, thermal_raw: np.ndarray, wok_mask: np.ndarray) -> np.ndarray:
    """3. 熱梯度邊界 HUD 注入 (最佳選用)"""
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

def fusion_method_4_guided(bgr: np.ndarray, t_pure4: np.ndarray, wok_mask: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    fused = cv2.addWeighted(bgr, 0.65, t_pure4, 0.35, 0)
    out[wok_mask > 0] = fused[wok_mask > 0]
    return out

def fusion_method_5_clahe(bgr: np.ndarray, t_pure5: np.ndarray, wok_mask: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    fused = cv2.addWeighted(bgr, 0.60, t_pure5, 0.40, 0)
    out[wok_mask > 0] = fused[wok_mask > 0]
    return out

def fusion_method_6_baseline(bgr: np.ndarray, t_pure6: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(bgr, 0.65, t_pure6, 0.35, 0)


def generate_benchmark_videos(bag_path: Path, out_pure_video: Path, out_fusion_video: Path, max_frames: int = 0):
    out_pure_video.parent.mkdir(parents=True, exist_ok=True)
    out_fusion_video.parent.mkdir(parents=True, exist_ok=True)

    db_files = list(bag_path.glob("*.db3")) if bag_path.is_dir() else [bag_path]
    if not db_files:
        print(f"[ERROR] 找不到 .db3: {bag_path}")
        return

    db_path = db_files[0]
    print(f"\n========================================================")
    print(f"同步生成 2 部 Thermal 對比展示影片")
    print(f"來源 Bag      : {db_path.name}")
    print(f"影片 1 (純 Thermal) : {out_pure_video.name}")
    print(f"影片 2 (RGBT 融合)  : {out_fusion_video.name}")
    print(f"========================================================\n")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM topics WHERE name LIKE '%compressed%'")
    target_id = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM messages WHERE topic_id = ?", (target_id,))
    total_msgs = cursor.fetchone()[0]
    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC", (target_id,))

    writer_pure = None
    writer_fusion = None
    frame_idx = 0
    t0 = time.time()

    pure_titles = [
        "1. Wok-Only Masked (Inferno)",
        "2. Calibrated Turbo Range (60-210 C)",
        "3. Thermal Gradient Field (Sobel)",
        "4. Bilateral Edge-Preserving (Magma)",
        "5. Adaptive CLAHE (Magma)",
        "6. Baseline Fixed Range (0-255)"
    ]

    fusion_titles = [
        "1. Wok-Only Masked Fusion",
        "2. Calibrated Turbo Fusion",
        "3. Thermal Gradient Edge HUD [Selected]",
        "4. Guided Bilateral Fusion",
        "5. Adaptive CLAHE Fusion",
        "6. Baseline Alpha Blend"
    ]

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
        thermal_raw = rgbt_rgba[:, :, 3]
        bgr_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
        h, w = bgr_raw.shape[:2]

        t_sec = frame_idx / 20.0

        wok_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(wok_mask, (int(w * 0.52), int(h * 0.58)), int(min(h, w) * 0.43), 255, -1)

        # 1. 計算 6 種純 Thermal 畫面
        p1 = pure_method_1_wok_masked(thermal_raw, wok_mask)
        p2 = pure_method_2_calibrated_turbo(thermal_raw)
        p3 = pure_method_3_gradient_field(thermal_raw, wok_mask)
        p4 = pure_method_4_bilateral_smooth(thermal_raw)
        p5 = pure_method_5_clahe_magma(thermal_raw)
        p6 = pure_method_6_baseline(thermal_raw)
        pure_panels = [p1, p2, p3, p4, p5, p6]

        # 2. 計算 6 種 RGBT 融合畫面
        f1 = fusion_method_1_wok_masked(bgr_raw, p1, wok_mask)
        f2 = fusion_method_2_calibrated_turbo(bgr_raw, p2, wok_mask)
        f3 = fusion_method_3_gradient_hud(bgr_raw, thermal_raw, wok_mask)
        f4 = fusion_method_4_guided(bgr_raw, p4, wok_mask)
        f5 = fusion_method_5_clahe(bgr_raw, p5, wok_mask)
        f6 = fusion_method_6_baseline(bgr_raw, p6)
        fusion_panels = [f1, f2, f3, f4, f5, f6]

        # 為面板添加 Header 標籤
        for p, title in zip(pure_panels, pure_titles):
            cv2.rectangle(p, (0, 0), (w, 36), (20, 20, 20), -1)
            cv2.putText(p, f"{title} | t={t_sec:.1f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2)

        for f, title in zip(fusion_panels, fusion_titles):
            cv2.rectangle(f, (0, 0), (w, 36), (20, 20, 20), -1)
            cv2.putText(f, f"{title} | t={t_sec:.1f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2)

        # 拼接 2x3 畫布
        grid_pure = np.vstack([np.hstack([pure_panels[0], pure_panels[1], pure_panels[2]]),
                               np.hstack([pure_panels[3], pure_panels[4], pure_panels[5]])])
        grid_fusion = np.vstack([np.hstack([fusion_panels[0], fusion_panels[1], fusion_panels[2]]),
                                 np.hstack([fusion_panels[3], fusion_panels[4], fusion_panels[5]])])

        if writer_pure is None:
            gh, gw = grid_pure.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer_pure = cv2.VideoWriter(str(out_pure_video), fourcc, 20.0, (gw, gh))
            writer_fusion = cv2.VideoWriter(str(out_fusion_video), fourcc, 20.0, (gw, gh))
            print(f"[INIT] 建立 2 部 2x3 六分割畫布: {gw}x{gh} @ 20 FPS")

        writer_pure.write(grid_pure)
        writer_fusion.write(grid_fusion)
        frame_idx += 1

        if frame_idx % 300 == 0:
            percent = (frame_idx / total_msgs) * 100.0 if total_msgs > 0 else 0
            print(f"進度: [{frame_idx:>4}/{total_msgs}] ({percent:5.1f}%) | 耗時: {time.time()-t0:.1f}s")

    conn.close()
    if writer_pure is not None:
        writer_pure.release()
    if writer_fusion is not None:
        writer_fusion.release()

    print(f"\n========================================================")
    print(f"2 部對比影片生成完畢！共 {frame_idx} 幀")
    print(f"1. 純 Thermal 影片: {out_pure_video.resolve()}")
    print(f"2. RGBT 融合影片  : {out_fusion_video.resolve()}")
    print(f"========================================================\n")

def main():
    parser = argparse.ArgumentParser(description="生成純 Thermal 與 RGBT 融合 2 部對比影片")
    parser.add_argument("--bag", type=str, default="data", help="Bag 路徑")
    parser.add_argument("--out_pure", type=str, default="data/thermal_pure_benchmark.mp4", help="純 Thermal 輸出影片路徑")
    parser.add_argument("--out_fusion", type=str, default="data/thermal_fusion_benchmark.mp4", help="RGBT 融合輸出影片路徑")
    parser.add_argument("--max_frames", type=int, default=0, help="最大幀數 (0 為全部)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    bag_path = project_root / args.bag
    out_pure = project_root / args.out_pure
    out_fusion = project_root / args.out_fusion
    generate_benchmark_videos(bag_path, out_pure, out_fusion, args.max_frames)

if __name__ == "__main__":
    main()
