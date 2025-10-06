# 胸部 X 光肺炎檢測

使用深度學習模型（ResNet）進行胸部 X 光影像的肺炎檢測分類。

## 專案說明

- 二分類問題：正常（NORMAL）vs 肺炎（PNEUMONIA）
- 支援三種 ResNet 模型：ResNet18、ResNet50、ResNet101
- 包含訓練、驗證和測試功能
- 自動生成準確率曲線、F1 分數圖表和混淆矩陣

## 環境需求

Python 3.9 與以下套件：

```bash
pip install -r requirements.txt
```

主要相依套件：
- PyTorch
- torchvision
- matplotlib
- numpy
- seaborn
- tqdm

## 資料集結構

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## 使用方法

### 基本方法

```bash
python lab1.py --model ResNet18
```

### 參數設定

- `--model`: 選擇模型 (ResNet18, ResNet50, ResNet101)
- `--num_epochs`: 訓練週期數 (預設: 20)
- `--batch_size`: 批次大小 (預設: 64)
- `--lr`: 學習率 (預設: 1e-5)
- `--resize`: 影像大小 (預設: 224)
- `--degree`: 旋轉角度 (預設: 90)

### 完整參數範例

```bash
python lab1.py --model ResNet50 --num_epochs 30 --batch_size 32 --lr 1e-4
```

## 輸出結果

### 模型權重
- `result/{model_name}_best_model.pt`: 最佳模型權重
- `result/{model_name}/weight{epoch}.pt`: 各週期權重

### 視覺化圖表
- `result/plot/{model_name}_accuracy_curve.png`: 準確率曲線
- `result/plot/{model_name}_f1_score.png`: F1 分數曲線
- `result/plot/{model_name}_test_confusion.png`: 測試混淆矩陣

### 訓練紀錄
- `train_log.txt`: 詳細訓練日誌

## 模型特色

- 使用預訓練 ResNet 模型進行遷移學習
- 加權交叉熵損失函數處理類別不平衡
- 數據增強：隨機旋轉
- 自動保存最佳模型權重
- 完整的評估指標：準確率、精確率、召回率、F1 分數

## 評估指標

- **準確率**: 正確分類的比例
- **精確率**: 預測為陽性中真正為陽性的比例
- **召回率**: 實際陽性中被正確預測的比例
- **F1 分數**: 精確率和召回率的調和平均
