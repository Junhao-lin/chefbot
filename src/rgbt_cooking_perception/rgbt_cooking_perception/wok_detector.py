"""
鍋具受熱面 Mask 偵測器 (支援 YOLO-Seg 與幾何圓融合)
"""

import numpy as np
import cv2

class WokDetector:
    def __init__(self, method: str = 'hybrid', model_path: str = None):
        self.method = method
        self.model = None
        self.ema_mask = None
        self.alpha = 0.25

        if model_path:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            except Exception:
                pass

    def detect_mask(self, bgr: np.ndarray, thermal_raw: np.ndarray = None) -> np.ndarray:
        h, w = bgr.shape[:2]
        prior_circle = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(prior_circle, (int(w * 0.52), int(h * 0.58)), int(min(h, w) * 0.43), 255, -1)

        seg_mask = np.zeros((h, w), dtype=np.uint8)
        if self.model is not None:
            try:
                res = self.model.predict(bgr, conf=0.30, verbose=False, device=0)[0]
                if res.masks is not None:
                    for box, poly in zip(res.boxes, res.masks.xy):
                        if int(box.cls[0]) == 0:
                            pts = poly.astype(np.int32)
                            if len(pts) >= 3:
                                cv2.fillPoly(seg_mask, [pts], 255)
            except Exception:
                pass

        if np.sum(seg_mask > 0) > (h * w * 0.15):
            raw_fused = cv2.bitwise_or(seg_mask, cv2.bitwise_and(prior_circle, cv2.dilate(seg_mask, np.ones((7, 7), np.uint8))))
            fused = cv2.morphologyEx(raw_fused, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        else:
            fused = prior_circle

        if self.ema_mask is None:
            self.ema_mask = fused.astype(np.float32)
        else:
            self.ema_mask = (1.0 - self.alpha) * self.ema_mask + self.alpha * fused.astype(np.float32)

        return (self.ema_mask > 127).astype(np.uint8) * 255
