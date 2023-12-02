import polars as pl
from preprocess_utils import drop_columns

path = "train.csv"
df_train = pl.read_csv(path, ignore_errors=True)
df_train = drop_columns(df=df_train)
# Save training data as .csv
df_train.write_csv(f"new_{path}", separator=",")

path = "valid.csv"
df_valid = pl.read_csv(path, ignore_errors=True)
df_valid = drop_columns(df=df_valid)
# Save training data as .csv
df_valid.write_csv(f"new_{path}", separator=",")

path = "real_valid2.csv"
df_valid = pl.read_csv(path, ignore_errors=True)
df_valid = drop_columns(df=df_valid)
# Save training data as .csv
df_valid.write_csv(f"new_{path}", separator=",")

path = "test.csv"
df_test = pl.read_csv(path, ignore_errors=True)
df_test = drop_columns(df=df_test)
# Save training data as .csv
df_test.write_csv(f"new_{path}", separator=",")
