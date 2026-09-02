# Shennongxi - CV Take-Home Technical Assessment
> **料理機器人 RGBT 電腦視覺與成熟度感知系統**

---

## To Do List (任務執行清單)

- [x] **階段一：環境規範、依賴設定與開源方案調研**
  - [x] 配置 WSL2 Ubuntu 22.04 與 ROS2 Humble 執行環境
  - [x] 盤點可用函式庫與開源視覺模型 (SAM 2, YOLO-Seg, SegFormer, Qwen2.5-VL)
  - [x] 建立 Python 依賴設定檔 (`requirements.txt`)
  - [x] 撰寫資料集下載與 RGBT ROS Bag 探索解碼腳本 (`scripts/explore_bag.py`)
- [x] **階段二：ROS2 Package 實作與 Thermal 多模態正規化演算法 (任務 1)**
  - [x] 建立 ROS2 Package (`rgbt_cooking_perception`) 與節點架構
  - [x] 實作 4-channel RGBT 解壓縮與 RGB / Thermal 分離
  - [x] 實作並評測 6 種 Thermal 正規化、校正與融合演算法 (`scripts/compare_thermal_fusions.py`)
  - [x] 選定最佳方案：**Method 3 - Thermal Gradient Edge HUD 注入**（鍋子與蛋熱反差最明顯）
  - [x] 實作即時鍋子幾何 Masking 發佈 (`/wok/mask`)
- [ ] **階段三：Segmentation 資料集製作、模型訓練與泛化驗證 (任務 2)**
  - [ ] 建立半自動標註管線 (VLM / SAM / 手動精細多邊形標註)
  - [ ] 嚴謹定義 Train/Val 切分策略，計算驗證幀與相鄰訓練幀之時間距離 $\Delta t$
  - [ ] 機器人運動學本體感知解耦：精簡為 3 大核心類別 (`wok`, `egg`, `container`)
  - [ ] 實作訓練腳本與 Jupyter Notebook，記錄調參歷程 (Loss, Backbone, Augmentation)
- [ ] **階段四：荷包蛋料理狀態機與 Doneness 完成度估計器 (任務 3)**
  - [ ] 形式化定義煎荷包蛋 5 步驟狀態機 (進入條件、監控訊號、完成判準、異常處理)
  - [ ] 空間幾何門控與熱量累積積分模型實作 (Zero Hardcoded Timestamps)
  - [ ] 標定判定閾值，詳列失敗模式 (油煙、反光、下油干擾) 與補償對策
- [ ] **階段五：深度技術討論題、完整報告與成果交付 (任務 4 & 討論題)**
  - [ ] 深入論述機器人料理關鍵 Perception 資訊與系統架構設計
  - [ ] 闡述感知模型於閉迴路控制 (Closed-loop Execution) 的實機落地
  - [ ] 評析 Sim-to-Real / RL / CV 三大領域精選前沿論文

---

## 🛠️ 環境配置指南 (Environment Setup)

本專案採用 **Windows 11 + WSL2 (Ubuntu 22.04 LTS)** 進行開發與驗證，底層通訊架構採用 **ROS2 Humble Hawksbill**。

### 1. 更新並加入 ROS2 官方源
```bash
sudo apt update && sudo apt install -y locales curl gnupg lsb-release software-properties-common
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo add-apt-repository universe -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### 2. 安裝 ROS2 Humble 桌面版與開發工具
```bash
sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions ros-humble-cv-bridge ros-humble-rqt-image-view python3-pip
```

### 3. 安裝 Python 視覺與演算法依賴
```bash
pip3 install -r requirements.txt
```

### 4. 設定環境自動載入
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 開源函式庫與模型技術調研 (Survey & Tools)

### 1. 影像處理與 RGBT 處理函式庫
* **OpenCV (`opencv-python`)**：支援多通道影像解碼、色空間轉換 (`COLOR_BGRA2RGBA`)、形態學運算、霍夫變換 (Hough Circle / Ellipse Fitting) 及 Colormap (Inferno/Turbo) 著色。
* **rosbags**：純 Python 實現的 ROS2 bag (`.db3` / `.mcap`) 解包與反序列化工具，可免除對完整 ROS2 安裝依賴進行輕量資料提取與預處理。
* **SciPy & NumPy**：用於溫度矩陣空間統計、高斯平滑、時間維度指數滑動平均 (EMA) 降噪。

### 2. 語意分割 (Semantic Segmentation) 開源模型評估
* **YOLOv8-Seg / YOLO11-Seg (Ultralytics)**：
  - **優點**：極致的推論速度 (> 60 FPS on GPU)，支援輕量級 Nano/Small 模型，導出 TensorRT 極為成熟，適合即時嵌入式部署。
  - **缺點**：對於微小流體邊緣（如半透明未熟蛋白）需精細標註才能收斂良好。
* **SegFormer (B0 / B1, Hugging Face `transformers` / `timm`)**：
  - **優點**：基於 Transformer 的 Hierarchical 架構，具有全域感受野與強大的多尺度特徵提取能力，對不同光照與反光有極佳強健性。
  - **缺點**：相對 CNN 架構運算量略高，需適當量化優化。
* **SAM (Segment Anything Model) & SAM 2**：
  - **應用定位**：作為**離線自動標註標籤生成器**。搭配 Prompt（如鍋底高溫熱區坐標或 VLM 生成的 Bounding Box），快速產生高精度物體 Mask。

---

## 階段二：ROS2 Package 實作與 Thermal 多模態正規化演算法 (任務 1)

### 1. ROS2 Package 架構設計 (`rgbt_cooking_perception`)
本階段建立了專屬的 ROS2 影像處理節點，負責將 4-Channel RGBT 影像解壓縮、分離為可見光 RGB 與熱成像 Thermal，並進行即時多模態影像增強與鍋具工作空間遮罩發佈：

* **節點名稱**：`rgbt_stream_processor`
* **訂閱 Topic**：`/camera/rgbt/compressed` (`sensor_msgs/msg/CompressedImage`)
* **發布 Topics**：
  * `/camera/rgb/image_raw` (`sensor_msgs/msg/Image`) - 原始可見光串流
  * `/camera/thermal/enhanced` (`sensor_msgs/msg/Image`) - 時間一致性增強熱成像
  * `/camera/rgbt/fusion` (`sensor_msgs/msg/Image`) - 多模態 HUD 疊合串流
  * `/wok/mask` (`sensor_msgs/msg/Image`) - 鍋具受熱面幾何遮罩

---

### 2. Thermal 6 種正規化與多模態融合演算法評測

熱成像感測器在料理過程中記錄的是紅外輻射能量。為了精確捕捉**鍋底升溫預熱、熱油流動、蛋液吸熱相變與翻面熟化**，我們實作並評測了 **6 種 Thermal 正規化與多模態融合方法**：

| 編號 | 方法名稱 (Method) | 演算法原理 | 優點 | 局限性 / 缺點 |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Wok-Only Masked Fusion<br>(空間局部遮罩融合)** | 限制只在黑鐵鍋受熱面內部疊加 Inferno 熱力圖，背景不銹鋼流理台維持 100% 純淨 RGB。 | 徹底消除流理台的金屬反光雜色，背景純淨。 | 鍋底高溫區容易過曝，細微溫差仍有遮蔽。 |
| **2** | **Calibrated Turbo Range<br>(物理溫差量程校準)** | 將量程動態鎖定在蛋白質變性關鍵物理區間 $[60^\circ\text{C}, 210^\circ\text{C}]$（數值 130~215），採用 Turbo Colormap。 | 呈現冷藍（生蛋）$\to$ 暖綠 $\to$ 亮黃 $\to$ 鮮紅（高溫鍋底）的豐富色彩層次。 | 全域顏色覆蓋仍會些微改變蛋白的原始 RGB 自然紋理。 |
| **3** | **Thermal Gradient Edge HUD<br>(熱梯度等溫線邊界注入) 🏆** | 利用 Sobel 一階微分提取熱通量梯度（$\nabla T$ / 等溫線），將溫差突變輪廓像抬頭顯示器 (HUD) 般以亮線注入 RGB。 | **【最佳選用 🌟】**<br>1. **鍋子與蛋的熱反差對比最明顯**：生蛋入鍋的吸熱降溫冷斑與高溫鍋底交界一目了然。<br>2. **不破壞 RGB 自然色彩**：完全保留蛋白白化紋理。 | 需注意高頻噪訊濾波（已加入 Gaussian Blur 平滑）。 |
| **4** | **Guided Filter Edge-Preserving<br>(引導濾波邊緣保真融合)** | 利用 RGB 高頻邊緣引導 Thermal 圖像平滑濾波，消除熱成像的像素毛邊。 | 邊緣過渡自然平滑，熱斑貼合物體外觀。 | 計算量稍大，在雙鏡頭近距離視差 (Parallax) 處偶有光暈。 |
| **5** | **Adaptive CLAHE + Magma<br>(自適應局部對比增強)** | 套用限制對比度自適應直方圖均衡化 (CLAHE)，局部拉伸熱反差。 | 大幅放大低溫區域的細微溫差。 | 在時序上會引起輕微的幀間亮度抖動 (Flickering)。 |
| **6** | **Baseline Fixed Range<br>(全域固定量程基準對照)** | 傳統固定量程 $[0, 255]$ 直接轉換為 Inferno 偽彩色並與 RGB 做加權疊合。 | 實現最簡單，絕對時間一致性最高。 | 高溫鍋底呈現一片死白/平坦過曝，喪失熱梯度細節。 |

---

### 3. 為什麼最終選擇「第 3 種方法 (Thermal Gradient Edge HUD)」？

1. **鍋子與蛋的熱反差對比最為強烈、最直觀**：
   * 雞蛋富含水分，打入 $200^\circ\text{C}$ 熱鍋時會瞬間產生劇烈吸熱降溫（$\Delta T \le 80^\circ\text{C}$）。
   * 一階熱通量梯度 $\nabla T$ 能在交界面產生極強的等溫線數值突變，將「低溫蛋黃/生蛋白」與「高溫黑鐵鍋底」的邊界如 HUD 輪廓線般精準割裂開來，**鍋子與蛋的視覺對比最為清晰明顯**！
2. **零破壞 RGB 可見光語意與蛋白白化辨識**：
   * 其他偽彩色方法（如 Inferno / Turbo）會把整顆蛋染成黃色或綠色，破壞了「蛋白由透明 $\to$ 白化變性」的自然顏色判準。
   * 熱梯度注入法僅在溫差邊界疊加微細輪廓線，讓模型與人眼能同時看清 **RGB 蛋白白化程度** 與 **熱力學溫度分佈**！

---

### 4. 執行 6 種方法 2x3 六分割對照影片生成

在已下載 Bag 檔案的環境下，執行以下腳本即可**一次同步生成兩部高畫質 2x3 六分割對比展示影片**：

```bash
python3 scripts/compare_thermal_fusions.py --out_pure data/thermal_pure_benchmark.mp4 --out_fusion data/thermal_fusion_benchmark.mp4
```

* **影片一（純 Thermal 處理對比）**：`data/thermal_pure_benchmark.mp4`（呈現 6 種不同的熱成像動態範圍拉伸、偽彩色著色與梯度場）。
* **影片二（RGBT 多模態融合對比）**：`data/thermal_fusion_benchmark.mp4`（呈現 6 種熱成像與可見光 RGB 的精準疊合效果，含 Method 3 熱梯度 HUD 注入）。
