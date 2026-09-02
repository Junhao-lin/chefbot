"""
OpenCV 互動式多邊形標註工具 (Interactive Polygon Annotation Tool)

3 大核心類別 (由數字鍵 0~2 切換)：
0: wok        (藍色) - 鍋具工作受熱面
1: egg        (黃色) - 荷包蛋 (生蛋、蛋白、熟蛋)
2: container  (洋紅) - 右上小鋼碗

操作指南：
- 滑鼠左鍵：新增多邊形頂點
- 滑鼠右鍵：閉合目前多邊形並儲存物件
- 按 '0', '1', '2'：切換標註類別
- 按 'c'：清除當前未完成的多邊形
- 按 'd'：刪除上一個已完成的物件
- 按 'n' / 空白鍵：儲存並切換至下一張影格
- 按 'b'：返回上一張影格
- 按 'q' / ESC：儲存並退出
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2

CLASS_INFO = {
    0: ("wok", (255, 0, 0)),
    1: ("egg", (0, 255, 255)),
    2: ("container", (255, 0, 255))
}

class InteractiveAnnotator:
    def __init__(self, images_dir: Path, labels_dir: Path):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
        if not self.image_files:
            print(f"[ERROR] 在 {images_dir} 找不到任何圖片！")
            sys.exit(1)

        self.current_idx = 0
        self.current_cls = 0
        self.current_points = []
        self.completed_objects = []

        self.window_name = "RGBT Interactive Annotator (0: Wok, 1: Egg, 2: Container)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 960)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def load_existing_labels(self, label_path: Path, h: int, w: int):
        self.completed_objects = []
        if not label_path.exists():
            return
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                if cls_id not in CLASS_INFO:
                    continue
                coords = [float(x) for x in parts[1:]]
                pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= w
                pts[:, 1] *= h
                self.completed_objects.append((cls_id, pts.astype(np.int32).tolist()))

    def save_labels(self, label_path: Path, h: int, w: int):
        with open(label_path, "w") as f:
            for cls_id, pts in self.completed_objects:
                if len(pts) >= 3:
                    pts_arr = np.array(pts, dtype=np.float32)
                    pts_arr[:, 0] /= w
                    pts_arr[:, 1] /= h
                    pts_flat = [f"{coord:.5f}" for coord in pts_arr.flatten()]
                    f.write(f"{cls_id} " + " ".join(pts_flat) + "\n")
        print(f"[SAVED] 已儲存標籤：{label_path.name} (包含 {len(self.completed_objects)} 個物件)")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([x, y])
            curr_cname, _ = CLASS_INFO[self.current_cls]
            print(f"[CLICK] 新增頂點 #{len(self.current_points)}: ({x}, {y}) | 類別: {curr_cname}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_points) >= 3:
                curr_cname, _ = CLASS_INFO[self.current_cls]
                self.completed_objects.append((self.current_cls, list(self.current_points)))
                print(f"[SUCCESS] 閉合多邊形：{curr_cname} (共 {len(self.current_points)} 點)")
                self.current_points = []
            else:
                print(f"[WARN] 多邊形至少需要 3 個點 (目前只有 {len(self.current_points)} 個點)")

    def run(self):
        while self.current_idx < len(self.image_files):
            img_path = self.image_files[self.current_idx]
            label_path = self.labels_dir / f"{img_path.stem}.txt"

            img = cv2.imread(str(img_path))
            if img is None:
                self.current_idx += 1
                continue
            h, w = img.shape[:2]

            self.load_existing_labels(label_path, h, w)
            self.current_points = []

            while True:
                canvas = img.copy()

                # 繪製已完成物件
                for cls_id, pts in self.completed_objects:
                    cname, color = CLASS_INFO[cls_id]
                    pts_arr = np.array(pts, dtype=np.int32)
                    overlay = canvas.copy()
                    cv2.fillPoly(overlay, [pts_arr], color)
                    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
                    cv2.polylines(canvas, [pts_arr], True, color, 2)
                    top_pt = pts_arr[np.argmin(pts_arr[:, 1])]
                    cv2.putText(canvas, cname, (top_pt[0], max(top_pt[1]-5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(canvas, cname, (top_pt[0], max(top_pt[1]-5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 繪製當前正在標註的頂點
                if len(self.current_points) > 0:
                    _, curr_color = CLASS_INFO[self.current_cls]
                    pts_arr = np.array(self.current_points, dtype=np.int32)
                    for pt in pts_arr:
                        cv2.circle(canvas, tuple(pt), 4, (0, 0, 255), -1)
                    if len(pts_arr) > 1:
                        cv2.polylines(canvas, [pts_arr], False, curr_color, 2)

                # 繪製頂部 HUD
                curr_cname, curr_color = CLASS_INFO[self.current_cls]
                hud_bg = canvas.copy()
                cv2.rectangle(hud_bg, (0, 0), (w, 35), (20, 20, 20), -1)
                canvas = cv2.addWeighted(hud_bg, 0.75, canvas, 0.25, 0)
                info_str = f"[{self.current_idx+1}/{len(self.image_files)}] {img_path.name} | Class: [{self.current_cls}] {curr_cname} | Objs: {len(self.completed_objects)}"
                cv2.putText(canvas, info_str, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                cv2.imshow(self.window_name, canvas)
                key = cv2.waitKey(20) & 0xFF

                if key in [ord('0'), ord('1'), ord('2')]:
                    self.current_cls = int(chr(key))
                    cname, _ = CLASS_INFO[self.current_cls]
                    print(f"[SWITCH] 切換當前標註類別為: [{self.current_cls}] {cname}")
                elif key == ord('c'):
                    self.current_points = []
                    print("[CLEAR] 清除當前未閉合多邊形")
                elif key == ord('d'):
                    if self.completed_objects:
                        removed = self.completed_objects.pop()
                        cname, _ = CLASS_INFO[removed[0]]
                        print(f"[DELETE] 刪除上一個物件: {cname}")
                elif key in [ord('n'), 32]: # n 或 空白鍵
                    self.save_labels(label_path, h, w)
                    self.current_idx += 1
                    break
                elif key == ord('b'):
                    if self.current_idx > 0:
                        self.save_labels(label_path, h, w)
                        self.current_idx -= 1
                        break
                elif key in [ord('q'), 27]: # q 或 ESC
                    self.save_labels(label_path, h, w)
                    cv2.destroyAllWindows()
                    print("\n[EXIT] 已退出標註工具。")
                    return

        cv2.destroyAllWindows()
        print("\n================ 全部圖片標註完畢！ ================")

def main():
    parser = argparse.ArgumentParser(description="互動式 YOLO-Seg 標註工具")
    parser.add_argument("--images", type=str, default="../data/dataset_seg_vlm/images", help="圖片目錄")
    parser.add_argument("--labels", type=str, default="../data/dataset_seg_vlm/labels", help="標籤輸出目錄")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    img_dir = project_root / args.images
    lbl_dir = project_root / args.labels

    annotator = InteractiveAnnotator(img_dir, lbl_dir)
    annotator.run()

if __name__ == "__main__":
    main()
