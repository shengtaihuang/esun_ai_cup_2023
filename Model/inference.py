"""
Inference and save csv to submit
"""
import numpy as np
import polars as pl
import xgboost as xgb

df_valid = pl.read_csv("valid.csv", ignore_errors=True)
# Drop the validation label to align with test data
df_valid = df_valid.drop("label")

df_test = pl.read_csv("test.csv", ignore_errors=True)
df = pl.concat([df_valid, df_test])
df = df.with_columns(
    pl.col("stscd").cast(pl.Categorical),
)
# Load submit example data
df_keys = pl.read_csv("31_範例繳交檔案.csv", ignore_errors=True)
print(df.shape)
# Map validation/test data to submit example data
df = df.filter(pl.col("txkey").is_in(df_keys["txkey"]))
print(df.shape)
print(df_keys.shape)
# Drop the keys of transactions
x_test = df.drop("txkey").to_pandas()

# Load the best iteration model for validation/test data prediction
y_preds = []
for i in range(1):
    model2 = xgb.XGBClassifier()
    model2.load_model(f"xgb_fold{i}.json")
    y_preds.append(
        model2.predict(x_test, iteration_range=(0, model2.best_iteration + 1))
    )

# Aggregate the predictions as a matrix
y_preds = np.vstack(y_preds)
# Take the mean of predictions and round to 0 or 1
y_pred = np.mean(y_preds, axis=0)
y_pred = np.round(y_pred)

# Save predictions as submit.csv
submit_df = pl.DataFrame({"txkey": df["txkey"], "pred": y_pred})
submit_df = submit_df.with_columns(
    pl.col("pred").cast(int),
)
print(submit_df["pred"].value_counts())
submit_df.write_csv("submit.csv")
