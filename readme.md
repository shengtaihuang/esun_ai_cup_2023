
## 檔案用途:
- Preprocess/: 存放前處理的code
- Model/: 存放模型相關code
- requirements.txt: 需要的套件

## 檔案結構
```
.
├ Preprocess
│ ├ preprocess.py
│ ├ eda.py
│ └ README
├ Model
│ ├ train.py
│ ├ inference.py
│ └ README
├ requirements.txt
└ README
```

## 執行流程:
```
# 安裝所需套件
$ pip install -r requirements.txt 

# 執行資料前處理
$ python ./Preprocess/preprocess.py
# Run eda to compute Spearman correlation between each predictors and label
$ python ./Preprocess/eda.py
# Rerun the preprocess to remove |correlation| <= 0.02
$ python ./Preprocess/preprocess.py

# training inference
$ python ./Model/train.py
# inference
$ python ./Model/inference.py
```
