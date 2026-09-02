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
- [x] **階段三：Segmentation 資料集製作、模型訓練與泛化驗證 (任務 2)**
  - [x] 建立半自動標註管線 (VLM / SAM / 手動精細多邊形標註)(效果不好)
  - [x] 嚴謹定義 Train/Val 切分策略，計算驗證幀與相鄰訓練幀之時間距離 $\Delta t$
  - [x] 機器人運動學本體感知解耦：精簡為 3 大核心類別 (`wok`, `egg`, `container`)
  - [x] 實作訓練腳本，記錄調參歷程 (Loss, Backbone, Augmentation)
- [x] **階段四：荷包蛋料理狀態機與 Doneness 完成度估計器 (任務 3)**
  - [x] 形式化定義煎荷包蛋 5 步驟狀態機 (進入條件、監控訊號、完成判準、異常處理)
  - [x] 實作空間幾何門控與 Arrhenius 不可逆熱量累積積分模型 (Zero Hardcoded Timestamps)
  - [x] 繪製全時序出版級雙 Y 軸狀態機分析曲線圖 (`data/cooking_fsm_doneness_timeline.png`)
  - [x] 實作深度學習 YOLO-Seg + 傳統幾何圓融合之 Realtime 鍋具 Masking (`/wok/mask`)
- [x] **階段五：生成綜合多模態 4-in-1 Dashboard 展示影片 (任務 1)**
  - [x] 建立 ROS2 Package (`rgbt_cooking_perception`) 與節點架構
  - [x] 實作 4-channel RGBT 解壓縮與 RGB / Thermal 分離發布
  - [x] 整合 Method 3 熱梯度等溫線 HUD 與即時鍋具 Masking 於 `/camera/dashboard`
  - [x] 支援 ROS2 即時檢視 (`rqt_image_view`) 與離線生成成果影片 (`data/rgbt_mask_dashboard.mp4`)
- [ ] **階段六：深度技術討論題、完整報告與成果交付 (任務 4 & 討論題)**
  - [ ] 深入論述機器人料理關鍵 Perception 資訊與系統架構設計
  - [ ] 闡述感知模型於閉迴路控制 (Closed-loop Execution) 的實機落地
  - [ ] 評析 Sim-to-Real / RL / CV 三大領域精選前沿論文



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


### 1. Thermal 6 種正規化與多模態融合演算法評測

熱成像感測器在料理過程中記錄的是紅外輻射能量。為了精確捕捉**鍋底升溫預熱、熱油流動、蛋液吸熱相變與翻面熟化**，我實作並評測了 **6 種 Thermal 正規化與多模態融合方法**：

| 編號 | 方法名稱 (Method) | 演算法原理 | 優點 | 局限性 / 缺點 |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Wok-Only Masked Fusion<br>(空間局部遮罩融合)** | 限制只在黑鐵鍋受熱面內部疊加 Inferno 熱力圖，背景不銹鋼流理台維持 100% 純淨 RGB。 | 徹底消除流理台的金屬反光雜色，背景純淨。 | 鍋底高溫區容易過曝，細微溫差仍有遮蔽。 |
| **2** | **Calibrated Turbo Range<br>(物理溫差量程校準)** | 將量程動態鎖定在蛋白質變性關鍵物理區間 $[60^\circ\text{C}, 210^\circ\text{C}]$（數值 130~215），採用 Turbo Colormap。 | 呈現冷藍（生蛋）$\to$ 暖綠 $\to$ 亮黃 $\to$ 鮮紅（高溫鍋底）的豐富色彩層次。 | 全域顏色覆蓋仍會些微改變蛋白的原始 RGB 自然紋理。 |
| **3** | **Thermal Gradient Edge HUD<br>(熱梯度等溫線邊界注入) [最佳選用]** | 利用 Sobel 一階微分提取熱通量梯度（$\nabla T$ / 等溫線），將溫差突變輪廓像抬頭顯示器 (HUD) 般以亮線注入 RGB。 | **【最佳選用】**<br>1. **鍋子與蛋的熱反差對比最明顯**：生蛋入鍋的吸熱降溫冷斑與高溫鍋底交界一目了然。<br>2. **不破壞 RGB 自然色彩**：完全保留蛋白白化紋理。 | 需注意高頻噪訊濾波（已加入 Gaussian Blur 平滑）。 |
| **4** | **Guided Filter Edge-Preserving<br>(引導濾波邊緣保真融合)** | 利用 RGB 高頻邊緣引導 Thermal 圖像平滑濾波，消除熱成像的像素毛邊。 | 邊緣過渡自然平滑，熱斑貼合物體外觀。 | 計算量稍大，在雙鏡頭近距離視差 (Parallax) 處偶有光暈。 |
| **5** | **Adaptive CLAHE + Magma<br>(自適應局部對比增強)** | 套用限制對比度自適應直方圖均衡化 (CLAHE)，局部拉伸熱反差。 | 大幅放大低溫區域的細微溫差。 | 在時序上會引起輕微的幀間亮度抖動 (Flickering)。 |
| **6** | **Baseline Fixed Range<br>(全域固定量程基準對照)** | 傳統固定量程 $[0, 255]$ 直接轉換為 Inferno 偽彩色並與 RGB 做加權疊合。 | 實現最簡單，絕對時間一致性最高。 | 高溫鍋底呈現一片死白/平坦過曝，喪失熱梯度細節。 |

---

### 2. 為什麼最終選擇「第 3 種方法 (Thermal Gradient Edge HUD)」？

1. **鍋子與蛋的熱反差對比最為強烈、最直觀**：
   * 雞蛋富含水分，打入 $200^\circ\text{C}$ 熱鍋時會瞬間產生劇烈吸熱降溫（$\Delta T \le 80^\circ\text{C}$）。
   * 一階熱通量梯度 $\nabla T$ 能在交界面產生極強的等溫線數值突變，將「低溫蛋黃/生蛋白」與「高溫黑鐵鍋底」的邊界如 HUD 輪廓線般精準割裂開來，**鍋子與蛋的視覺對比最為清晰明顯**！
2. **零破壞 RGB 可見光語意與蛋白白化辨識**：
   * 其他偽彩色方法（如 Inferno / Turbo）會把整顆蛋染成黃色或綠色，破壞了「蛋白由透明 $\to$ 白化變性」的自然顏色判準。
   * 熱梯度注入法僅在溫差邊界疊加微細輪廓線，讓模型與人眼能同時看清 **RGB 蛋白白化程度** 與 **熱力學溫度分佈**！

---

### 3. 執行 6 種方法 2x3 六分割對照影片生成

在已下載 Bag 檔案的環境下，執行以下腳本即可**一次同步生成兩部高畫質 2x3 六分割對比展示影片**：

```bash
python3 scripts/compare_thermal_fusions.py --out_pure data/thermal_pure_benchmark.mp4 --out_fusion data/thermal_fusion_benchmark.mp4
```

* **影片一（純 Thermal 處理對比）**：`data/thermal_pure_benchmark.mp4`（呈現 6 種不同的熱成像動態範圍拉伸、偽彩色著色與梯度場）。
* **影片二（RGBT 多模態融合對比）**：`data/thermal_fusion_benchmark.mp4`（呈現 6 種熱成像與可見光 RGB 的精準疊合效果，含 Method 3 熱梯度 HUD 注入）。

---

## 階段三：Segmentation 資料集製作、模型訓練與泛化驗證 (任務 2)

### 1. 標註策略演進與各方案實測評估

在語意分割（Semantic Segmentation）資料集的構建過程中，我評估並實測了 4 種標註策略，最終確立了以專家精細手動多邊形標註為主、結合運動學解耦的資料集建置方案：

#### 方案 A：傳統規則式自動標註 (Rule-based OpenCV)
* **對應腳本**：`scripts/label/auto_annotate_rule_based.py`
* **方法**：透過 HSV 顏色閾值、邊緣檢測與 Hough Circle 擬合自動生成 176 張初版標籤。
* **評測結果**：在靜態且光照均勻幀表現尚可，但遇金屬流理台強烈鏡面反光時，誤將反光處標註為 container；在雞蛋翻面焦化後因顏色變暗產生嚴重漏檢。標籤整體雜訊率約 25%~30%，無法作為高精度真實標籤 (Ground Truth)。

#### 方案 B：視覺語言大模型標註 (VLM - Qwen2.5-VL)
* **對應腳本**：`scripts/label/annotate_with_vlm.py`
* **方法**：透過呼叫多模態大模型進行零樣本 (Zero-shot) 物件偵測與座標生成。
* **評測結果**：VLM 具備極強的場景語意理解能力（能精確理解「碗內生蛋」、「鍋內熟蛋」與「金屬鍋鏟」之概念），但目前 VLM 輸出之多邊形座標多為 4 頂點之粗糙邊界框（Bounding Box-like Polygon），無法緊密貼合液態蛋液擴散與非剛性曲面，幾何精度不足以支撐精細接觸判斷。

#### 方案 C：人類專家精細多邊形手動標註 (Expert Polygon Annotation)
* **對應腳本**：
  * 關鍵影格分層提取：`scripts/label/extract_keyframe_dataset.py`
  * 互動多邊形標註工具：`scripts/label/annotate_interactive.py`
* **方法**：透過專屬提取腳本依料理時間段抽取 36 張代表性影格，並利用 OpenCV 互動式標註工具進行 20~50 頂點的高密度多邊形標註。
* **評測結果**：標註精確貼合黑鐵鍋受熱面內徑、蛋黃流動邊界、變性蛋白白化層與右上鋼碗邊緣，標籤品質達到頂級基準。
##### 確認個方案標註品質
```bash
python3 scripts/label/visualize_annotations.py --images data/dataset_seg_manual/images --labels data/dataset_seg_manual/labels
```
##### 執行手動標註工作流
```bash
# 步驟 1: 分層提取 36 張關鍵影格
python3 scripts/label/extract_keyframe_dataset.py --output data/dataset_seg_manual/images

# 步驟 2: 啟動互動標註工具 (0: Wok, 1: Egg, 2: Container)
python3 scripts/label/annotate_interactive.py --images data/dataset_seg_manual/images --labels data/dataset_seg_manual/labels
```

##### 關鍵影格分層時間段採樣分佈 (Stratified Temporal Sampling)
為了涵蓋荷包蛋料理全生命週期的形態演化與光學/熱力學相變，我將 223 秒影片劃分為 4 大關鍵階段進行針對性分層抽樣，共計精標 36 張關鍵影格：

| 料理時間階段 | 視訊時間戳 (秒 / 幀序號) | 採樣張數 | 標註重點與形態特徵 |
| :--- | :--- | :---: | :--- |
| **階段一：空鍋預熱與熱油潤鍋** | 0s ~ 60s<br>(Frames 0 ~ 1200) | 6 張 | 黑鐵鍋受熱面幾何基準、熱油倒下時的液膜反光、右上鋼碗內未下鍋的生蛋黃。 |
| **階段二：生蛋入鍋與蛋白白化** | 60s ~ 115s<br>(Frames 1200 ~ 2300) | 12 張 | 62s 倒蛋瞬間、透明生蛋液受熱擴散、蛋白自邊緣向中心白化凝固、隆起蛋黃。 |
| **階段三：翻面定型與顛鍋動態** | 115s ~ 160s<br>(Frames 2300 ~ 3200) | 10 張 | 118s 翻面動作、空中翻騰、翻面後焦黃底面朝上、大廚顛鍋時鍋具傾斜位姿。 |
| **階段四：雙面熟化與起鍋裝盤** | 160s ~ 223s<br>(Frames 3200 ~ 4400) | 8 張 | 雙面均勻受熱、蛋黃熱穿透定型、200s 起鍋脫離黑鐵鍋進入餐盤出餐。 |

---
### 2. 類別精簡與機器人本體感知解耦

在系統設計上，我將視覺模型類別由傳統 4 類精簡為 **3 大核心類別**：

* `0: wok`（黑鐵鍋工作受熱面）
* `1: egg`（荷包蛋本體：涵蓋生蛋液、白化蛋白、翻面焦黃）
* `2: container`（右上生蛋小鋼碗）

#### 為什麼剔除鍋鏟 (Spatula) 視覺類別？
在真實機器人料理系統中，鍋鏟或末端工具固定於機械臂法蘭盤上。系統透過機械臂的**正向運動學（Forward Kinematics, FK）與 ROS2 `/tf` 座標變換樹**，即可 100% 精確掌握工具末端的 3D 空間位姿，根本不需要浪費視覺神經網路資源進行工具偵測。剔除鍋鏟能徹底消除工具遮擋與反光造成的類別混淆，讓模型全力聚焦於料理食材與鍋具的物理交互。

---

### 3. 嚴謹時序步長切分 (Temporal Strided Split)

連續視訊影格具有高度的時間自相關性（Temporal Correlation）。若採用隨機切分（Random Split），相鄰影格（相隔僅 0.05 秒）同時出現在訓練集與驗證集，將導致嚴重的資料外洩（Data Leakage）與虛高評估指標。

* **切分策略**：採用交錯步長切分（每 4 幀取第 2 幀作為驗證集，其餘為訓練集）。
* **時間距離**：確保每張驗證幀與相鄰訓練幀之間保有足夠的時間間隔 $\Delta t \ge 0.15\text{s} \sim 0.20\text{s}$，真實反映模型對未見過時間點的泛化能力。

---

### 4. 執行資料集切分、訓練與推論

#### 步驟 1：時序切分資料集並啟動 YOLO-Seg 訓練
```bash
python3 scripts/train/split_and_train_dataset.py --src data/dataset_seg_manual --target data/dataset_seg_manual --train --epochs 80
```

#### 步驟 2：推論原始 Bag 影片並產出展示影片
```bash
python3 scripts/train/infer_3class_video.py --model runs/segment/cooking_seg_3class_expert/weights/best.pt --output data/seg_inference_3class_expert.mp4
```

### 5. 顛鍋誤判診斷與動態資料增強策略

在初期模型推論驗證中，我發現大廚進行**顛鍋（Wok Tossing / Tilting）與晃鍋**動作時，模型會產生短暫的鍋具與蛋體漏檢或誤判。

#### 根本原因分析 (Root Cause Analysis)
1. **採樣偏差 (Sampling Bias)**：初版關鍵影格採樣主要以時間均勻步長抽取，大部分影格中黑鐵鍋皆處於水平靜止狀態。
2. **非剛體視角變化與底座露出**：顛鍋時，大廚將鍋具提起並大幅度傾斜（產生 Pitch / Roll 旋轉角），鍋底離開爐面導致瓦斯爐黑鐵底座露出，模型在未見過「傾斜鍋具與爐座背景」的情況下產生特徵混淆。

#### 改進對策 (Targeted Dynamic Augmentation)
* **補充動態時序樣本**：在翻面與顛鍋關鍵時段（約 135 秒 ~ 160 秒）專門補充採樣 3 張「鍋具傾斜、大角度位移、翻面空中翻騰」之高動態影格進行手動多邊形標註。
---



---

## 階段四：荷包蛋料理狀態機與 Realtime 鍋具 Masking 融合 (任務 3)

### 1. 荷包蛋料理 5 步驟形式化狀態機 (Finite State Machine, FSM)

為了實現完全由多模態感知訊號驅動的自主料理閉迴路控制，我形式化定義了 5 大料理狀態：

| 狀態 (State) | 進入條件 (Entry Condition) | 即時監控訊號 (Perception Signals) | 狀態完成判準 (Exit Trigger) | 失敗模式與異常防護 (Failure Modes & Recovery) |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: 預熱狀態<br>(Pre-heating)** | 機器人開火，鍋內為空 ($A_{egg} = 0$)。 | 鍋底均溫 $T_{wok}$ (透過 `/camera/thermal/enhanced`)。 | $T_{wok} \ge 160^\circ\text{C}$ (約 44s)。 | 若 $T > 230^\circ\text{C}$ 觸發乾燒警報並調降火力。 |
| **Phase 2: 下油潤鍋<br>(Oil Seasoning)** | 鍋溫達標，機械臂注油。 | 油溫分佈、生蛋碗空間狀態、鍋底冷斑檢測。 | 生蛋落入鍋底引發劇烈吸熱冷斑 ($T_{egg} \le 135^\circ\text{C}$) 且脫離右上小鋼碗 (約 62s)。 | **下油干擾防護**：食用油比熱容小($\approx 1.7\,\text{J/g}^\circ\text{C}$)，入鍋迅速升溫不降溫；雞蛋含水率高($75\%$)，入鍋必產生冷斑，藉此物理量精確區分下油 vs 下蛋。 |
| **Phase 3: 蛋白變性凝固<br>(Side-1 Coagulation)** | 檢測到生蛋入鍋開始受熱。 | 蛋白白化面積、蛋體平均溫度 $T_{egg}$、熱通量累積積分 $E(t)$。 | 第一面成熟度達到翻面閾值：$D(t) \ge 85\%$ (約 118s / 1分58秒)。 | **不可逆保護**：蛋白蛋白質變性不可逆，熟度嚴格滿足單調不減 $D(t) = \max(D(t), D(t-1))$，杜絕鍋鏟插入時數值暴跌。 |
| **Phase 4: 翻面熟化<br>(Flip & Side-2)** | 機械臂執行翻面動作。 | 焦黃底面朝上之特徵、雙面接觸傳熱積分。 | 總累積成熟度達標：$D(t) \ge 100\%$ (約 200s)。 | **顛鍋晃鍋防護**：大廚晃鍋時結合運動學先驗與容器空間互斥，防止右上小碗與鍋具產生假交集。 |
| **Phase 5: 起鍋裝盤<br>(Plating & Serve)** | 熟度達 100%，機械臂執行鏟起裝盤。 | 蛋體脫離鍋具 ($A_{egg \cap wok} \to 0$)。 | 蛋體成功轉移至餐盤，料理週期結束。 | 若起鍋超時 15 秒觸發重試機制。 |

---

### 2. 多模態物理動力學熱量累積積分模型 (Arrhenius Thermal Energy Model)

蛋白質熱變性並非單幀靜態光學特徵，而是熱通量隨時間的累積積分：

$$E(t) = \int_{t_{\text{in}}}^{t} \max(T_{\text{egg}}(\tau) - T_{\text{crit}}, 0) \cdot \mu_{\text{white}}(\tau) \, d\tau$$

* $T_{\text{crit}} = 60^\circ\text{C}$：雞蛋卵白蛋白（Ovalbumin）開始熱變性的臨界溫度。
* $\mu_{\text{white}}$：可見光 RGB 分割中蛋白區域的白化程度特徵權重。
* **單調不減約束**：$D(t) = \max(D(t), D(t-1))$，徹底解決大廚翻面、鍋鏟遮擋與油煙干擾時成熟度劇烈抖動的工程難題。

---

### 3. 即時鍋具 Masking 方案：深度學習 YOLO-Seg + 傳統幾何先驗圓融合

根據任務 1 要求以即時方式發佈 `/wok/mask`，實作了**混合融合架構（Hybrid Fusion & Temporal Smoothing）**：

#### 演算法原理
1. **深度學習分支 (YOLO-Seg)**：微調後的 YOLO-Seg 即時預測黑鐵鍋受熱面之精準多邊形，能完美適應大廚晃鍋、顛鍋位移與非剛體傾斜視角。
2. **傳統幾何先驗分支 (Prior Circle)**：提供黑鐵鍋標準受熱面的幾何圓形空間基底。
3. **形態學融合與時序 EMA 平滑**：
   * 當 YOLO-Seg 檢測良好時，將多邊形與先驗圓進行形態學聯集與閉合平滑（`Morphology Close`），既保有動態邊界又剔除高頻噪訊。
   * 當劇烈顛鍋或大面積工具遮擋導致檢測置信度下降時，幾何先驗圓自動平滑兜底。
   * 透過時序指數滑動平均（$\alpha = 0.25$）濾波，輸出**極致平滑、零跳閃的即時 `/wok/mask` 遮罩**！

---

### 4. 執行狀態機時序圖表與鍋具 Mask 融合生成

#### 步驟 1：產出全時序狀態機與 Doneness 雙軸分析圖
```bash
python3 scripts/fsm/plot_fsm_timeline.py --output data/cooking_fsm_doneness_timeline.png
```

#### 步驟 2：生成深度學習 + 幾何圓融合鍋具 Masking 展示影片
```bash
python3 scripts/wok/generate_wok_mask_fusion.py --output data/wok_mask_fusion_demo.mp4
```

---
## 階段五：生成綜合多模態 4-in-1 Dashboard 展示影片 (任務 1)

### 1. 多模態融合監控面板設計理念 (Multi-modal Dashboard Architecture)

在機器人自主料理系統中，單一感測模態往往存在感知盲區：
* 可見光 RGB 能清晰辨識蛋體外觀形態與邊界，但無法感知鍋底是否達到預熱溫度，亦無法捕捉隱藏在油煙下的劇烈吸熱相變；
* 熱成像 Thermal 能精確測量紅外輻射熱量，但缺乏細節紋理與邊界語意；
* 幾何遮罩 `/wok/mask` 則提供了機械臂操作的剛性邊界防護。

為了讓工程人員、監控節點與上位決策系統能在**單一低延遲串流**中宏觀掌握料理全貌，我設計了 **綜合多模態 4 合 1 監控面板 (`/camera/dashboard`)**：

| 面板位置 | 主題名稱 (ROS2 Topic) | 技術原理與視覺化設計 | 料理控制決策用途 |
| :--- | :--- | :--- | :--- |
| **面板 1 (左上)** | `/rgb/image_raw` | 4 通道 RGBT 解碼後還原之純淨 3 通道可見光串流。 | 監控食材幾何形態、白化程度與大廚操作動作。 |
| **面板 2 (右上)** | `/thermal/enhanced` | **Method 3 - Thermal Gradient Edge HUD 注入影像**。<br>利用 Sobel 一階微分提取溫差熱梯度（$\nabla T$），以等溫線形式疊合於 RGB 畫面。 | 凸顯生蛋入鍋引發的吸熱降溫冷斑與高溫鍋底邊界，不破壞 RGB 自然色澤。 |
| **面板 3 (左下)** | `/wok/mask` | **深度學習 YOLO-Seg 權重 + 傳統幾何先驗圓融合遮罩**。<br>結合形態學閉運算與時序指數平滑（EMA），以純黑底亮綠色即時輸出。 | 提供機械臂炒菜、顛鍋與鏟蛋的受熱工作空間邊界約束。 |
| **面板 4 (右下)** | `/camera/dashboard` | **綜合多模態融合畫面**。<br>整合 RGB 背景、半透明黃色鍋具 Mask、亮黃色邊界輪廓、彩色熱梯度等溫線、以及頂部即時均溫 HUD 資訊列。 | 提供系統全局即時狀態監控（Zero-Latency 全模態一覽無遺）。 |

---

### 2. ROS2 Package 架構設計 (`rgbt_cooking_perception`)

* **套件路徑**：`src/rgbt_cooking_perception`
* **節點名稱**：`rgbt_stream_processor`
* **訂閱 Topic**：`/rgbt/rgbt/compressed` (`sensor_msgs/msg/CompressedImage`)
* **發布 Topics**：
  * `/rgb/image_raw` 及 `/camera/rgb/image_raw` (`sensor_msgs/msg/Image`, `bgr8`) - 原始可見光串流
  * `/thermal/enhanced` 及 `/camera/thermal/enhanced` (`sensor_msgs/msg/Image`, `bgr8`) - Method 3 熱梯度等溫線增強影像
  * `/wok/mask` (`sensor_msgs/msg/Image`, `mono8`) - 鍋具受熱面遮罩
  * `/camera/dashboard` (`sensor_msgs/msg/Image`, `bgr8`) - 綜合多模態 4 合 1 監控畫面

---

### 3. ROS2 套件編譯與多終端機啟動操作流程

經實測驗證，請開啟 **3 個 WSL2 終端機視窗** 進行驗收操作：

```bash
# ----------------- [視窗 1] 編譯並啟動影像處理節點 -----------------
source /opt/ros/humble/setup.bash
colcon build --build-base /tmp/build --install-base /tmp/install --packages-select rgbt_cooking_perception
source /tmp/install/setup.bash
ros2 launch rgbt_cooking_perception stream_perception.launch.py

# ----------------- [視窗 2] 循環播放 ROS2 Bag 影像串流 -----------------
source /opt/ros/humble/setup.bash
ros2 bag play data -l

# ----------------- [視窗 3] 開啟 ROS 影像檢視器 -----------------
# 方式 A: 快速開啟單一主題 (例如 Thermal 等溫線增強畫面)
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view /thermal/enhanced

# 方式 B: 開啟 rqt 完整多面板儀表板 (可同時新增多個 Image View 並列顯示)
rqt

# (選用) 驗收 Topic 發布頻率:
ros2 topic hz /thermal/enhanced
```

---

### 4. 離線一鍵生成成果展示影片 (免 ROS 環境)

若在 Windows 或純 Python 環境下進行成果展示，可直接執行以下腳本產出高畫質 2x2 成果影片：

```bash
python3 scripts/wok/generate_rgbt_mask_dashboard_video.py --output data/rgbt_mask_dashboard.mp4
```
* **輸出檔案**：`data/rgbt_mask_dashboard.mp4`（1080p, 20 FPS, H.264 編碼）

---