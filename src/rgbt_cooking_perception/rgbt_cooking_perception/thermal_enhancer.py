"""
Thermal 影像增強器模組 (支援 Method 3 熱梯度等溫線 HUD 與多種時間一致性增強)
"""

import numpy as np
import cv2

class ThermalEnhancer:
    def __init__(self, mode: str = 'gradient_hud', min_val: float = 130.0, max_val: float = 215.0, ema_alpha: float = 0.05, colormap: str = 'inferno', apply_clahe: bool = True):
        self.mode = mode
        self.min_val = min_val
        self.max_val = max_val
        self.ema_alpha = ema_alpha
        self.colormap = colormap
        self.apply_clahe = apply_clahe
        self.ema_min = None
        self.ema_max = None

        if apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def enhance(self, thermal_raw: np.ndarray, bgr_raw: np.ndarray = None, wok_mask: np.ndarray = None) -> np.ndarray:
        h, w = thermal_raw.shape[:2]
        if wok_mask is None:
            wok_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(wok_mask, (int(w * 0.52), int(h * 0.58)), int(min(h, w) * 0.43), 255, -1)

        # 預設採用 Method 3: 熱梯度等溫線 HUD 注入
        if bgr_raw is not None:
            t_blur = cv2.GaussianBlur(thermal_raw, (5, 5), 0)
            gx = cv2.Sobel(t_blur, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(t_blur, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx**2 + gy**2)
            mag_norm = np.clip(mag / 25.0, 0.0, 1.0)
            edge_color = cv2.applyColorMap((mag_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

            out = bgr_raw.copy()
            edge_mask = (mag_norm > 0.15) & (wok_mask > 0)
            out[edge_mask] = cv2.addWeighted(bgr_raw[edge_mask], 0.35, edge_color[edge_mask], 0.65, 0)
            return out

        # 純 Thermal 偽彩色正規化
        t_norm = np.clip((thermal_raw.astype(np.float32) - self.min_val) / (self.max_val - self.min_val + 1e-5), 0.0, 1.0)
        return cv2.applyColorMap((t_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
