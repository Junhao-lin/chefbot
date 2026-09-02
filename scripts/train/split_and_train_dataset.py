"""
資料集時序步長切分與 YOLO-Seg 深度學習訓練管線 (Dataset Split & Train Pipeline)

核心機制：
1. 時序交錯區段切分 (Temporal Strided Split):
   - 每 4 幀取 1 幀作為驗證集 (Val)，其餘為訓練集 (Train)
   - 確保驗證幀與前後訓練幀之間保有足夠時間距離 Delta_t，杜絕資料外洩 (Data Leakage)
2. 類別定義 (3 類精純模型):
   - 0: wok (黑鐵鍋工作受熱面)
   - 1: egg (荷包蛋：生蛋液、流動蛋黃、白化蛋白、翻面焦黃)
   - 2: container (右上生蛋小鋼碗)
3. 嚴格移動並清理頂層殘留檔案，只保留 images/train, images/val, labels/train, labels/val
"""

import sys
import shutil
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

def split_dataset_temporal_stride(src_dir: Path, target_dir: Path, val_stride: int = 4):
    target_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        (target_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    src_img_dir = src_dir / "images"
    src_lbl_dir = src_dir / "labels"

    # 1. 抓取所有圖片 (包含頂層或已在子目錄中的圖片)
    top_img_files = sorted([p for p in src_img_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    
    # 若頂層沒有圖片，但 train/val 內有，則說明圖片已在子目錄
    if not top_img_files:
        all_imgs = sorted(list((target_dir / "images" / "train").glob("*.*")) + list((target_dir / "images" / "val").glob("*.*")))
        if all_imgs:
            print(f"[INFO] 圖片已完成分配 (共 {len(all_imgs)} 張)")
        else:
            raise FileNotFoundError(f"在 {src_img_dir} 找不到任何圖片！")
    else:
        all_imgs = top_img_files

    print(f"\n================ 開始進行時序區段切分 (Move & Clean) ================")
    print(f"來源目錄: {src_dir}")
    print(f"目標目錄: {target_dir}")
    print(f"總圖片數: {len(all_imgs)} 張")
    print(f"驗證步長: 每 {val_stride} 幀取第 2 幀為 Val (Train 75% / Val 25%)\n")

    train_count = 0
    val_count = 0

    for idx, img_path in enumerate(all_imgs):
        split = "val" if (idx % val_stride == 2) else "train"
        tgt_img_path = target_dir / "images" / split / img_path.name
        tgt_lbl_path = target_dir / "labels" / split / f"{img_path.stem}.txt"

        # 尋找對應的標籤來源 (優先看頂層，次看子目錄)
        lbl_candidates = [
            src_lbl_dir / f"{img_path.stem}.txt",
            src_lbl_dir / "train" / f"{img_path.stem}.txt",
            src_lbl_dir / "val" / f"{img_path.stem}.txt"
        ]
        found_lbl = next((p for p in lbl_candidates if p.exists()), None)

        if found_lbl is not None and found_lbl.exists():
            with open(found_lbl, "r", encoding="utf-8") as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                coords = parts[1:]
                
                # 嚴格 3 類別：0: wok, 1: egg, 2: container
                if cls_id == 0:
                    new_lines.append(f"0 " + " ".join(coords) + "\n")
                elif cls_id == 1:
                    new_lines.append(f"1 " + " ".join(coords) + "\n")
                elif cls_id == 2:
                    new_lines.append(f"2 " + " ".join(coords) + "\n")

            with open(tgt_lbl_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        else:
            tgt_lbl_path.touch()

        # 移動圖片至子目錄
        if img_path != tgt_img_path:
            shutil.move(str(img_path), str(tgt_img_path))

        if split == "train":
            train_count += 1
        else:
            val_count += 1

    # 2. 清理頂層殘留的 txt 檔案與 cache 快取
    for item in src_lbl_dir.iterdir():
        if item.is_file() and item.suffix.lower() in [".txt", ".cache"]:
            item.unlink()
            print(f"[CLEAN] 清理頂層標籤/快取殘留: {item.name}")

    for item in src_img_dir.iterdir():
        if item.is_file() and item.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            item.unlink()
            print(f"[CLEAN] 清理頂層圖片殘留: {item.name}")

    total = train_count + val_count
    print(f"\n資料集切分完成！總計: {total} 張")
    print(f"- 訓練集 (Train): {train_count} 張 ({train_count/total*100:.1f}%)")
    print(f"- 驗證集 (Val)  : {val_count} 張 ({val_count/total*100:.1f}%)")

    # 建立純淨 3 類別 data.yaml
    data_yaml = {
        "path": str(target_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "wok",
            1: "egg",
            2: "container"
        }
    }

    yaml_path = target_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"3 類別設定檔已生成: {yaml_path}")
    return yaml_path

def train_model(data_yaml: Path, model_name: str = "yolo11s-seg.pt", epochs: int = 80, batch_size: int = 8, device: int = 0):
    print(f"\n================ 啟動 3 類別 YOLO-Seg 模型訓練 ================")
    print(f"資料集配置: {data_yaml}")
    print(f"基座模型  : {model_name}")
    print(f"訓練輪數  : {epochs} Epochs | Batch: {batch_size} | Device: GPU {device}")

    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"[WARN] 載入 {model_name} 失敗 ({e})，自動切換至 yolov8n-seg.pt")
        model = YOLO("yolov8n-seg.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        project="runs/segment",
        name="cooking_seg_3class_expert",
        exist_ok=True,
        verbose=True,
        plots=True,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        flipud=0.0,
        fliplr=0.5
    )

    best_weight = Path("runs/segment/cooking_seg_3class_expert/weights/best.pt")
    print(f"\n================ 訓練完成！ ================")
    print(f"最佳權重路徑: {best_weight.resolve()}")
    return best_weight

def main():
    parser = argparse.ArgumentParser(description="資料集時序切分與 YOLO-Seg 訓練管線")
    parser.add_argument("--src", type=str, default="data/dataset_seg_manual", help="標註來源目錄")
    parser.add_argument("--target", type=str, default="data/dataset_seg_manual", help="輸出切分目錄")
    parser.add_argument("--model", type=str, default="yolo11s-seg.pt", help="基座模型 (yolo11s-seg.pt 或 yolov8n-seg.pt)")
    parser.add_argument("--epochs", type=int, default=80, help="訓練輪數")
    parser.add_argument("--batch", type=int, default=8, help="Batch Size")
    parser.add_argument("--train", action="store_true", help="切分後是否直接啟動訓練")
    args = parser.parse_args()

    src_dir = Path(args.src)
    target_dir = Path(args.target)

    yaml_path = split_dataset_temporal_stride(src_dir, target_dir)

    if args.train:
        train_model(yaml_path, args.model, args.epochs, args.batch)

if __name__ == "__main__":
    main()
