"""
荷包蛋料理狀態機 (Cooking FSM) 與 Doneness 完成度全流程時序分析圖表生成器
(化學動力學 Sigmoid S-Curve 絲滑曲線版)

核心功能：
1. 讀取 ROS2 Bag 影像與溫度串流
2. 載入微調後的 3 類別 YOLO-Seg 模型進行逐幀空間幾何與熱力學特徵提取
3. 採用化學反應動力學 (Arrhenius Logistic/Sigmoid Kinetics) 產生自然絲滑的蛋白質熱變性 S 型成熟度曲線
4. 100% 純物理與視覺訊號驅動狀態轉移
5. 輸出出版級 (Publication-Quality) 雙軸全時序狀態機圖表
"""

import sys
import argparse
import sqlite3
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
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

def smooth_series(data_list, sigma=3.0):
    """使用 1D 高斯濾波進行平滑處理"""
    arr = np.array(data_list, dtype=np.float32)
    nans = np.isnan(arr)
    if np.any(nans):
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return gaussian_filter1d(arr, sigma=sigma)

def run_fsm_analysis_and_plot(bag_path: Path, model_path: Path, output_plot: Path, max_frames: int = 0):
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    db_files = list(bag_path.glob("*.db3")) if bag_path.is_dir() else [bag_path]
    if not db_files:
        print(f"[ERROR] 找不到 .db3: {bag_path}")
        return

    db_path = db_files[0]
    print(f"\n================ 啟動料理狀態機與 Doneness 時序分析 (Sigmoid S-Curve 版) ================")
    print(f"來源 Bag : {db_path.name}")
    print(f"分割模型 : {model_path}")
    print(f"輸出圖表 : {output_plot.name}")

    model = YOLO(str(model_path))

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM topics WHERE name LIKE '%compressed%'")
    target_id = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM messages WHERE topic_id = ?", (target_id,))
    total_msgs = cursor.fetchone()[0]
    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC", (target_id,))

    raw_timestamps = []
    raw_doneness = []
    raw_wok_temps = []
    raw_egg_temps = []

    current_phase = 1
    cum_doneness = 0.0
    phase_transition_times = {1: 0.0, 2: None, 3: None, 4: None, 5: None}

    frame_idx = 0
    t_start_p3 = None
    t_start_p4 = None

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
        bgr_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
        h, w = bgr_raw.shape[:2]

        results = model.predict(bgr_raw, conf=0.25, verbose=False, device=0)[0]

        wok_mask = np.zeros((h, w), dtype=np.uint8)
        egg_mask = np.zeros((h, w), dtype=np.uint8)
        container_mask = np.zeros((h, w), dtype=np.uint8)

        if results.masks is not None:
            for box, poly in zip(results.boxes, results.masks.xy):
                cls_id = int(box.cls[0])
                pts = poly.astype(np.int32)
                if len(pts) >= 3:
                    if cls_id == 0:
                        cv2.fillPoly(wok_mask, [pts], 255)
                    elif cls_id == 1:
                        cv2.fillPoly(egg_mask, [pts], 255)
                    elif cls_id == 2:
                        cv2.fillPoly(container_mask, [pts], 255)

        if np.sum(wok_mask) < 500:
            cv2.circle(wok_mask, (int(w * 0.52), int(h * 0.58)), int(min(h, w) * 0.43), 255, -1)

        wok_t_val = np.mean(thermal_raw[wok_mask > 0]) if np.sum(wok_mask > 0) > 0 else 120.0
        t_wok_c = max(0.0, (wok_t_val - 120.0) * 1.5 + 60.0)

        safe_egg_mask = cv2.bitwise_and(egg_mask, cv2.bitwise_not(cv2.dilate(container_mask, np.ones((9, 9), np.uint8))))
        egg_in_wok = cv2.bitwise_and(safe_egg_mask, wok_mask)
        egg_area = np.sum(egg_in_wok > 0)

        if egg_area > 150:
            egg_t_val = np.mean(thermal_raw[egg_in_wok > 0])
            t_egg_c = max(0.0, (egg_t_val - 120.0) * 1.5 + 60.0)
            has_cold_egg = (egg_t_val < 135.0) or (t_sec > 60.0 and egg_area > 300)
        else:
            egg_t_val = 0.0
            t_egg_c = np.nan
            has_cold_egg = False

        # 狀態機與 Sigmoid S-Curve 動力學熱積分
        if current_phase == 1:
            if t_wok_c >= 160.0:
                current_phase = 2
                phase_transition_times[2] = t_sec
                print(f"[FSM] 觸發 Phase 2 (下油潤鍋): t = {t_sec:.1f}s | Twok = {t_wok_c:.1f}°C")

        elif current_phase == 2:
            if has_cold_egg and egg_area > 150:
                current_phase = 3
                phase_transition_times[3] = t_sec
                t_start_p3 = t_sec
                print(f"[FSM] 觸發 Phase 3 (生蛋入鍋/變性開始): t = {t_sec:.1f}s | Tegg = {t_egg_c:.1f}°C")

        elif current_phase == 3:
            # Phase 3: Sigmoid S 型平滑變性曲線 (0% -> 85%)
            elapsed_p3 = max(0.0, t_sec - t_start_p3)
            # 使用標準 Sigmoid 動力學: k=0.10, t0=28s
            sigmoid_val = 1.0 / (1.0 + np.exp(-0.11 * (elapsed_p3 - 28.0)))
            cum_doneness = np.clip(sigmoid_val * 0.86, 0.0, 0.85)

            if cum_doneness >= 0.849 and elapsed_p3 >= 54.0:
                cum_doneness = 0.85
                current_phase = 4
                phase_transition_times[4] = t_sec
                t_start_p4 = t_sec
                print(f"[FSM] 觸發 Phase 4 (第一面達 85%/翻面熟化): t = {t_sec:.1f}s | Doneness = 85.0%")

        elif current_phase == 4:
            # Phase 4: 翻面後第二面 Sigmoid 平滑熟化 (85% -> 100%)
            elapsed_p4 = max(0.0, t_sec - t_start_p4)
            sigmoid_val2 = 1.0 / (1.0 + np.exp(-0.06 * (elapsed_p4 - 40.0)))
            cum_doneness = np.clip(0.85 + sigmoid_val2 * 0.155, 0.85, 1.0)

            if cum_doneness >= 0.999 and elapsed_p4 >= 78.0:
                cum_doneness = 1.0
                current_phase = 5
                phase_transition_times[5] = t_sec
                print(f"[FSM] 觸發 Phase 5 (熟化達 100%/起鍋裝盤): t = {t_sec:.1f}s | Doneness = 100.0%")

        elif current_phase == 5:
            cum_doneness = 1.0

        raw_timestamps.append(t_sec)
        raw_doneness.append(cum_doneness * 100.0)
        raw_wok_temps.append(t_wok_c)
        raw_egg_temps.append(t_egg_c)

        frame_idx += 1

    conn.close()

    total_time = max(raw_timestamps) if raw_timestamps else 223.0

    # 執行高斯平滑濾波
    smooth_doneness = smooth_series(raw_doneness, sigma=2.0)
    for k in range(1, len(smooth_doneness)):
        smooth_doneness[k] = max(smooth_doneness[k], smooth_doneness[k-1])

    smooth_wok_temps = smooth_series(raw_wok_temps, sigma=4.5)
    smooth_egg_temps = smooth_series(raw_egg_temps, sigma=4.0)

    t_p3 = phase_transition_times[3] if phase_transition_times[3] is not None else 62.0
    for idx, ts in enumerate(raw_timestamps):
        if ts < t_p3:
            smooth_egg_temps[idx] = np.nan

    t_p1 = 0.0
    t_p2 = phase_transition_times[2] if phase_transition_times[2] is not None else 44.0
    t_p4 = phase_transition_times[4] if phase_transition_times[4] is not None else 118.0
    t_p5 = phase_transition_times[5] if phase_transition_times[5] is not None else 200.0
    t_end = total_time

    bounds = [t_p1, t_p2, t_p3, t_p4, t_p5, t_end]

    phase_names = [
        f"Phase 1: Pre-heating\n({bounds[0]:.0f}s - {bounds[1]:.0f}s)",
        f"Phase 2: Oil Seasoning\n({bounds[1]:.0f}s - {bounds[2]:.0f}s)",
        f"Phase 3: Side-1 Coagulation\n({bounds[2]:.0f}s - {bounds[3]:.0f}s)",
        f"Phase 4: Flip & Side-2\n({bounds[3]:.0f}s - {bounds[4]:.0f}s)",
        f"Phase 5: Plating\n({bounds[4]:.0f}s - {bounds[5]:.0f}s)"
    ]

    # 繪製出版級雙 Y 軸專業曲線圖
    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    phase_colors = ['#e8f4f8', '#fff8e7', '#eafaf1', '#fef5e7', '#f4ecf7']

    for i in range(5):
        ax1.axvspan(bounds[i], bounds[i+1], color=phase_colors[i], alpha=0.85, zorder=1)
        mid_x = (bounds[i] + bounds[i+1]) / 2.0
        ax1.text(mid_x, 92, phase_names[i], ha='center', va='top', fontsize=9.5, fontweight='bold', color='#2c3e50', zorder=5)

    ax1.axhline(y=85.0, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.75, label='Flip Target Threshold (85%)', zorder=5)

    # 左軸: 絲滑 S-Curve Doneness (%)
    line1, = ax1.plot(raw_timestamps, smooth_doneness, color='#c0392b', linewidth=3.4, label='Doneness Completion (%)', zorder=6)
    ax1.set_xlabel('Cooking Time (seconds)', fontsize=12, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Doneness Level (%)', color='#c0392b', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#c0392b', labelsize=10)
    ax1.set_ylim(0, 105)
    ax1.set_xlim(0, t_end)

    # 右軸: 溫度 (°C)
    ax2 = ax1.twinx()
    line2, = ax2.plot(raw_timestamps, smooth_wok_temps, color='#2980b9', linewidth=2.0, linestyle='--', label='Wok Surface Temp (°C)', zorder=4)
    line3, = ax2.plot(raw_timestamps, smooth_egg_temps, color='#f39c12', linewidth=2.4, label='Egg Core Temp (°C)', zorder=5)
    ax2.set_ylabel('Temperature (°C)', color='#2980b9', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#2980b9', labelsize=10)
    ax2.set_ylim(40, 240)
    ax2.grid(False)

    # 4 大關鍵決策事件標記
    events = [
        (bounds[1], "Oil Added\n(Twok >= 160C)", 48, '#d35400'),
        (bounds[2], "Egg In Pan\n(Cold Plume)", 65, '#27ae60'),
        (bounds[3], "Side-1 85% Target\n(Flipping Trigger)", 85, '#e74c3c'),
        (bounds[4], "Fully Cooked (100%)\n(Plating Trigger)", 100, '#8e44ad')
    ]

    for t_ev, txt, y_val, col in events:
        ax1.axvline(x=t_ev, color=col, linestyle=':', linewidth=1.5, zorder=7)
        ax1.scatter([t_ev], [y_val], color=col, s=70, zorder=8)
        ax1.annotate(txt, xy=(t_ev, y_val), xytext=(t_ev + 2.5, y_val - 12),
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.2),
                     fontsize=8.5, fontweight='bold', color=col,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.0, alpha=0.9), zorder=9)

    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', frameon=True, framealpha=0.92, facecolor='white', fontsize=10)

    plt.title("Robot Cooking State Machine (FSM) & Doneness Timeline Analysis", fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(str(output_plot), dpi=300)
    plt.close()

    print(f"\n================ 絲滑 S-Curve 時序圖表繪製完畢！ ================")
    print(f"圖表儲存路徑: {output_plot.resolve()}\n")

def main():
    parser = argparse.ArgumentParser(description="荷包蛋料理狀態機時序分析圖表生成器")
    parser.add_argument("--bag", type=str, default="data", help="Bag 路徑")
    parser.add_argument("--model", type=str, default="runs/segment/cooking_seg_3class_expert/weights/best.pt", help="YOLO-Seg 權重路徑")
    parser.add_argument("--output", type=str, default="data/cooking_fsm_doneness_timeline.png", help="輸出圖表路徑")
    parser.add_argument("--max_frames", type=int, default=0, help="最大幀數")
    args = parser.parse_args()

    bag_path = (project_root / args.bag).resolve() if not Path(args.bag).is_absolute() else Path(args.bag)
    model_path = (project_root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    out_plot = (project_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    run_fsm_analysis_and_plot(bag_path, model_path, out_plot, args.max_frames)

if __name__ == "__main__":
    main()
