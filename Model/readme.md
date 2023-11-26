# 模型訓練
本資料夾內含兩個程式:
- train.py: 訓練模型
- inference.py: 使用訓練好的模型進行預測，並產生繳交的csv檔

## 參數設定
- XGBoost
  - n_estimators: 200
  - learning_rate: 0.3
  - max_depth: 12
  - tree_method: "hist"
  - enable_categorical: True
  - loss: "binary:logistic"