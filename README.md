# Shennongxi - CV Take-Home Technical Assessment
> **料理機器人 RGBT 電腦視覺與成熟度感知系統**

---

## To Do List (任務執行清單)

- [o] **階段一：環境規範、依賴設定與開源方案調研**
  - [o] 配置 WSL2 Ubuntu 22.04 與 ROS2 Humble 執行環境
  - [o] 盤點可用函式庫與開源視覺模型 (SAM, YOLO-Seg, SegFormer)
  - [o] 建立 Python 依賴設定檔 (`requirements.txt`)
  - [o] 撰寫資料集下載與 RGBT ROS Bag 探索解碼腳本 (`scripts/explore_bag.py`)
- [ ] **階段二：ROS2 Package 實作與 RGBT 影像串流處理 (任務 1)**
  - [ ] 建立 ROS2 Package (`rgbt_cooking_perception`) 與節點架構
  - [ ] 實作 4-channel RGBT 解壓縮與 RGB/Thermal 分離
  - [ ] 實作 Thermal 時間一致性增強（比較逐幀正規化 vs 固定物理範圍）
  - [ ] 實作即時鍋子 Masking 發佈 (`/wok/mask`)
  - [ ] 撰寫 Launch 檔、參數配置與 `ros2 topic hz` 頻率驗證
- [ ] **階段三：Segmentation 資料集製作、模型訓練與泛化驗證 (任務 2)**
  - [ ] 建立半自動標註管線 (VLM / SAM / 溫度差聚類輔助標註)
  - [ ] 嚴謹定義 Train/Val 切分策略，計算驗證幀與相鄰訓練幀之時間距離 $\Delta t$
  - [ ] 分析單一連續影片的驗證有效性與理想多維度資料需求
  - [ ] 實作訓練腳本與 Jupyter Notebook，記錄調參歷程 (Loss, Backbone, Augmentation)
- [ ] **階段四：荷包蛋料理狀態機與 Doneness 完成度估計器 (任務 3)**
  - [ ] 形式化定義煎荷包蛋 5 步驟狀態機 (進入條件、監控訊號、完成判準、異常處理)
  - [ ] 實作全程 Doneness 訊號估計器並繪製時間曲線圖
  - [ ] 標定判定閾值，詳列失敗模式 (遮擋、油煙、反光) 與補償對策
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

## 🔍 開源函式庫與模型技術調研 (Survey & Tools)

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
