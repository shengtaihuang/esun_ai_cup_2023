"""
Use EDA tools to recognize low correlation columns
"""
from dataprep import eda as dpeda
import polars as pl

df = pl.read_csv("train.csv", ignore_errors=True)

# EDA for added numerical columns
count_cols = "^.*count$"
mean_cols = "^.*mean$"
std_cols = "^.*std$"
dev_cols = "^.*dev$"

cols_list = [count_cols, mean_cols, std_cols, dev_cols]
filenames = ["count", "mean", "std", "dev"]
for cols, filename in zip(cols_list, filenames):
    out = df.select(pl.col(cols), pl.col("label")).to_pandas()
    report = dpeda.create_report(out, title=f"ESun {cols} Data")
    report.save(f"esun2023_{filename}_eda")

# EDA for added NA counting (group by another coumn) column
NA_cols = ["etymd", "mcc", "stocn", "scity", "stscd", "hcefg", "csmcu"]
cols_list = [f"^{nc}.*NAs$" for nc in NA_cols]
for nc, cols in zip(NA_cols, cols_list):
    out = df.select(pl.col(cols), pl.col("label")).to_pandas()
    report = dpeda.create_report(out, title=f"ESun {cols} Data")
    report.save(f"esun2023_{nc}_NAs_eda")

# EDA for added binary summing (group by another coumn) column
bin_cols = ["ecfg", "insfg", "bnsfg", "ovrlt", "flbmk", "flg_3dsmk"]
cols_list = [f"^{bc}.*1$" for bc in bin_cols]
for bc, cols in zip(bin_cols, cols_list):
    out = df.select(pl.col(cols), pl.col("label")).to_pandas()
    report = dpeda.create_report(out, title=f"ESun {cols} Data")
    report.save(f"esun2023_{bc}_sum_eda")
