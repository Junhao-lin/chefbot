"""
ROS2 RGBT 影像處理與即時鍋具 Masking 節點 (標準穩定版)
"""

import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage

def ndarray_to_imgmsg(img: np.ndarray, encoding: str, header=None) -> Image:
    msg = Image()
    if header is not None:
        msg.header = header
    h, w = img.shape[:2]
    msg.height = int(h)
    msg.width = int(w)
    msg.encoding = encoding
    msg.is_bigendian = False
    channels = 1 if len(img.shape) == 2 else img.shape[2]
    msg.step = int(w * channels * img.dtype.itemsize)
    msg.data = img.tobytes()
    return msg

class RGBTStreamProcessorNode(Node):
    def __init__(self):
        super().__init__('rgbt_stream_processor')

        # 發布 Topics (同時發布帶 /camera 與不帶 /camera 前綴，確保所有選單皆有畫面)
        self.pub_rgb1 = self.create_publisher(Image, '/rgb/image_raw', 10)
        self.pub_rgb2 = self.create_publisher(Image, '/camera/rgb/image_raw', 10)

        self.pub_thermal1 = self.create_publisher(Image, '/thermal/enhanced', 10)
        self.pub_thermal2 = self.create_publisher(Image, '/camera/thermal/enhanced', 10)

        self.pub_mask = self.create_publisher(Image, '/wok/mask', 10)
        self.pub_dashboard = self.create_publisher(Image, '/camera/dashboard', 10)

        # 訂閱 Bag 中的真實主題
        self.sub_comp = self.create_subscription(
            CompressedImage,
            '/rgbt/rgbt/compressed',
            self.image_callback,
            10
        )

        self.frame_count = 0
        self.last_time = time.time()
        self.get_logger().info('====== [READY] 成功啟動 rgbt_stream_processor，正在訂閱 /rgbt/rgbt/compressed ======')

    def image_callback(self, msg: CompressedImage):
        t0 = time.time()

        # 1. 解碼 4 通道 PNG (BGRA)
        np_arr = np.frombuffer(msg.data, np.uint8)
        rgbt_bgra = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if rgbt_bgra is None:
            return

        # 2. 轉換 BGRA -> RGBA 還原正確通道排布
        rgbt_rgba = cv2.cvtColor(rgbt_bgra, cv2.COLOR_BGRA2RGBA)
        rgb_raw = np.ascontiguousarray(rgbt_rgba[:, :, :3], dtype=np.uint8)
        thermal_raw = np.ascontiguousarray(rgbt_rgba[:, :, 3], dtype=np.uint8)
        bgr_raw = np.ascontiguousarray(cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR))
        h, w = bgr_raw.shape[:2]

        # 3. 鍋具 Masking (幾何圓先驗)
        wok_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(wok_mask, (int(w * 0.52), int(h * 0.58)), int(min(h, w) * 0.43), 255, -1)

        # 4. Method 3 Thermal 熱梯度等溫線 HUD 注入增強
        t_blur = cv2.GaussianBlur(thermal_raw, (5, 5), 0)
        gx = cv2.Sobel(t_blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(t_blur, cv2.CV_32F, 0, 1, ksize=3)
        mag_norm = np.clip(np.sqrt(gx**2 + gy**2) / 25.0, 0.0, 1.0)
        edge_color = cv2.applyColorMap((mag_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

        t_enhanced = bgr_raw.copy()
        edge_mask = (mag_norm > 0.15) & (wok_mask > 0)
        t_enhanced[edge_mask] = cv2.addWeighted(bgr_raw[edge_mask], 0.35, edge_color[edge_mask], 0.65, 0)

        # 5. 綜合多模態 Dashboard
        dashboard = bgr_raw.copy()
        mask_overlay = dashboard.copy()
        mask_overlay[wok_mask > 0] = (255, 200, 0)
        dashboard = cv2.addWeighted(mask_overlay, 0.35, dashboard, 0.65, 0)
        cnts, _ = cv2.findContours(wok_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dashboard, cnts, -1, (0, 255, 255), 2)
        dashboard[edge_mask] = cv2.addWeighted(dashboard[edge_mask], 0.30, edge_color[edge_mask], 0.70, 0)

        hud = dashboard.copy()
        cv2.rectangle(hud, (0, 0), (w, 35), (20, 20, 20), -1)
        dashboard = cv2.addWeighted(hud, 0.75, dashboard, 0.25, 0)
        avg_temp = (np.mean(thermal_raw[wok_mask > 0]) - 120.0) * 1.5 + 60.0
        cv2.putText(dashboard, f"RGBT Dashboard | Twok: {avg_temp:.1f}C | /wok/mask: Active", (12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        # 6. 發布 Topics (完整帶上原始時間戳與 header)
        rgb_msg = ndarray_to_imgmsg(bgr_raw, encoding='bgr8', header=msg.header)
        self.pub_rgb1.publish(rgb_msg)
        self.pub_rgb2.publish(rgb_msg)

        th_msg = ndarray_to_imgmsg(t_enhanced, encoding='bgr8', header=msg.header)
        self.pub_thermal1.publish(th_msg)
        self.pub_thermal2.publish(th_msg)

        mask_msg = ndarray_to_imgmsg(wok_mask, encoding='mono8', header=msg.header)
        self.pub_mask.publish(mask_msg)

        dash_msg = ndarray_to_imgmsg(dashboard, encoding='bgr8', header=msg.header)
        self.pub_dashboard.publish(dash_msg)

        self.frame_count += 1
        if self.frame_count % 60 == 0:
            fps = 60.0 / (time.time() - self.last_time + 1e-6)
            self.get_logger().info(f"[ROS2 處理中] 已轉發第 {self.frame_count} 幀至 /rgb/image_raw, /thermal/enhanced (即時幀率: {fps:.1f} FPS)")
            self.last_time = time.time()

def main(args=None):
    rclpy.init(args=args)
    node = RGBTStreamProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
