# 資料前處理
本資料夾內含兩個程式:
- preprocess.py: 載入原始資料並以 Feature engineering 新增欄位
- eda.py: 用以計算 Spearman's correlation 以作為去除和 label 低相關性 (|correlation| <= 0.02>) 的 columns
