"""
Train XGBoost model
"""
import numpy as np
import polars as pl
from sklearn.metrics import f1_score
import xgboost as xgb

# Add this to avoid error while concatenate string columns
pl.enable_string_cache()
df_train = pl.read_csv("train.csv", ignore_errors=True)
# Add some validation data to training data
df_valid = pl.read_csv("valid.csv", ignore_errors=True)
df = pl.concat([df_train, df_valid])
df = df.with_columns(
    pl.col("stscd").cast(pl.Categorical),
)
#
x = df.drop(["txkey", "label"]).to_pandas()
# Assign label column as "y"
y = df["label"].to_pandas()
# Keep some validation data for evaluation
idx = -1_000
x_train = x[:idx]
y_train = y[:idx]

x_valid = x[idx:]
y_valid = y[idx:]

# Initial XGBoost classifier
clf = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.3,
    max_depth=12,
    tree_method="hist",
    enable_categorical=True,
    objective="binary:logistic",
)

# Fit XGBoost model
for i in range(1):
    print(i)
    clf.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
    )
    # Save model into JSON format.
    clf.save_model(f"xgb_fold{i}.json")

# Load the best iteration model for validation data prediction
y_preds = []
for i in range(1):
    model2 = xgb.XGBClassifier()
    model2.load_model(f"xgb_fold{i}.json")
    y_preds.append(
        model2.predict(x_valid, iteration_range=(0, model2.best_iteration + 1))
    )
# Aggregate the predictions as a matrix
y_preds = np.vstack(y_preds)
# Take the mean of predictions and round to 0 or 1
y_pred = np.mean(y_preds, axis=0)
y_pred = np.round(y_pred)
# Compute F1-score
print(f1_score(y_true=y_valid, y_pred=y_pred))
