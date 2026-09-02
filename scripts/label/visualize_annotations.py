"""
標註視覺化檢驗工具 (Dataset Annotation Visualizer)

功能：
1. 讀取影像與對應的 YOLO-Seg 多邊形標籤檔案 (.txt)
2. 在影像上疊加 3 大類別的半透明 Mask、輪廓線與標籤：
   - 0: Wok (藍色)
   - 1: Egg (黃色)
   - 2: Container (洋紅色)
3. 支援兩種模式：
   - 互動視窗模式 (預設)：按 'n'/空白鍵 下一張，按 'b' 上一張，按 'q'/ESC 退出
   - 批次輸出模式 (--save_dir)：將所有疊合檢驗圖片批次儲存至指定資料夾
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2

def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "README.md").exists() and (parent / "requirements.txt").exists():
            return parent
    return Path.cwd()

project_root = get_project_root()

CLASS_INFO = {
    0: ("Wok", (255, 0, 0)),
    1: ("Egg", (0, 255, 255)),
    2: ("Container", (255, 0, 255))
}

def load_yolo_seg_label(label_path: Path, h: int, w: int) -> list:
    objects = []
    if not label_path.exists():
        return objects

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            objects.append((cls_id, pts.astype(np.int32)))
    return objects

def render_overlay(img: np.ndarray, objects: list) -> np.ndarray:
    canvas = img.copy()
    overlay = canvas.copy()

    for cls_id, pts in objects:
        cname, color = CLASS_INFO.get(cls_id, (f"Class {cls_id}", (0, 255, 0)))
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(canvas, [pts], True, color, 2)

            top_pt = pts[np.argmin(pts[:, 1])]
            label_str = f"{cname} ({len(pts)} pts)"
            cv2.putText(canvas, label_str, (top_pt[0], max(top_pt[1]-6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
            cv2.putText(canvas, label_str, (top_pt[0], max(top_pt[1]-6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    canvas = cv2.addWeighted(overlay, 0.40, canvas, 0.60, 0)
    return canvas

def run_visualizer(images_dir: Path, labels_dir: Path, save_dir: Path = None):
    # 搜尋頂層或子層的圖片
    img_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) +
                       list((images_dir / "train").glob("*.jpg")) + list((images_dir / "val").glob("*.jpg")))
    if not img_files:
        print(f"[ERROR] 在 {images_dir} 找不到任何圖片！")
        return

    print(f"\n================ 啟動標註視覺化檢驗工具 ================")
    print(f"圖片目錄: {images_dir}")
    print(f"標籤目錄: {labels_dir}")
    print(f"總圖片數: {len(img_files)} 張\n")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"批次輸出目錄: {save_dir}")
        for idx, img_path in enumerate(img_files):
            # 智慧搜尋標籤
            candidates = [
                labels_dir / f"{img_path.stem}.txt",
                labels_dir / "train" / f"{img_path.stem}.txt",
                labels_dir / "val" / f"{img_path.stem}.txt"
            ]
            label_path = next((p for p in candidates if p.exists()), labels_dir / f"{img_path.stem}.txt")

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            objects = load_yolo_seg_label(label_path, h, w)
            rendered = render_overlay(img, objects)
            
            out_path = save_dir / f"vis_{img_path.name}"
            cv2.imwrite(str(out_path), rendered)
            print(f"[{idx+1:02d}/{len(img_files)}] 已輸出檢驗圖: {out_path.name} (含 {len(objects)} 個物件)")
        print(f"\n批次檢驗圖輸出完成！已儲存至: {save_dir}\n")
        return

    # 互動式視窗預覽
    win_name = "YOLO-Seg Annotation Visualizer (n: Next, b: Prev, q: Quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 960)

    curr_idx = 0
    while curr_idx < len(img_files):
        img_path = img_files[curr_idx]
        candidates = [
            labels_dir / f"{img_path.stem}.txt",
            labels_dir / "train" / f"{img_path.stem}.txt",
            labels_dir / "val" / f"{img_path.stem}.txt"
        ]
        label_path = next((p for p in candidates if p.exists()), labels_dir / f"{img_path.stem}.txt")

        img = cv2.imread(str(img_path))
        if img is None:
            curr_idx += 1
            continue
        h, w = img.shape[:2]

        objects = load_yolo_seg_label(label_path, h, w)
        canvas = render_overlay(img, objects)

        # 頂部 HUD 資訊
        hud = canvas.copy()
        cv2.rectangle(hud, (0, 0), (w, 36), (20, 20, 20), -1)
        canvas = cv2.addWeighted(hud, 0.75, canvas, 0.25, 0)
        info_str = f"[{curr_idx+1}/{len(img_files)}] {img_path.name} | Objects: {len(objects)} (0: Wok, 1: Egg, 2: Container)"
        cv2.putText(canvas, info_str, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(0) & 0xFF

        if key in [ord('n'), 32]:
            curr_idx = (curr_idx + 1) % len(img_files)
        elif key == ord('b'):
            curr_idx = (curr_idx - 1 + len(img_files)) % len(img_files)
        elif key in [ord('q'), 27]:
            break

    cv2.destroyAllWindows()
    print("\n檢驗結束。")

def main():
    parser = argparse.ArgumentParser(description="YOLO-Seg 標註視覺化檢驗工具")
    parser.add_argument("--images", type=str, default="data/dataset_seg_manual/images", help="圖片目錄")
    parser.add_argument("--labels", type=str, default="data/dataset_seg_manual/labels", help="標籤目錄")
    parser.add_argument("--save_dir", type=str, default="", help="批次儲存目錄 (若指定則不開啟視窗)")
    args = parser.parse_args()

    img_p = Path(args.images)
    img_dir = img_p if img_p.is_absolute() else (project_root / args.images).resolve()

    lbl_p = Path(args.labels)
    lbl_dir = lbl_p if lbl_p.is_absolute() else (project_root / args.labels).resolve()

    s_dir = Path(args.save_dir).resolve() if args.save_dir else None

    run_visualizer(img_dir, lbl_dir, s_dir)

if __name__ == "__main__":
    main()
