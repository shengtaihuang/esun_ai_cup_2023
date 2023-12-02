"""
前處理資料
training, public 前處理包含 x 和 y
private_1_processed 前處理只包含 x
"""
import polars as pl
from preprocess_utils import load_data

# Add this to avoid error while concatenate string columns
pl.enable_string_cache()
# Preprocess training data
df_train, train_label = load_data(
    path="dataset_1st/training.csv", mode="train", limit=None
)

# Preprocess validation data
df_valid, valid_label = load_data(
    path="dataset_2nd/public.csv", mode="valid", limit=None
)

# Preprocess test data
df_valid2, valid2_label = load_data(
    path="dataset_2nd/private_1.csv", mode="valid2", limit=None
)

# Preprocess new test data
df_test = load_data(path="private_2_processed.csv", mode="test", limit=None)


# Concatenate training/validation/test to preprocess them together
df = pl.concat([df_train, df_valid, df_valid2, df_test])

df = df.with_columns(
    pl.col("contp").cast(str).cast(pl.Categorical),
    pl.col("etymd").cast(str).cast(pl.Categorical),
    pl.col("mcc").cast(str).cast(pl.Categorical),
    pl.col("stocn").cast(str).cast(pl.Categorical),
    pl.col("scity").cast(str).cast(pl.Categorical),
    pl.col("hcefg").cast(str).cast(pl.Categorical),
    pl.col("csmcu").cast(str).cast(pl.Categorical),
)

# Transform time column "loctm" to string and pad them to the same length = 6
df = df.with_columns(
    pl.col("loctm").cast(str).str.zfill(6),
)
# Transform the timestamp to seconds
df = df.with_columns(
    pl.col("loctm").str.slice(offset=0, length=2).alias("loctm_hh").cast(int) * 60 * 60,
    pl.col("loctm").str.slice(offset=2, length=2).alias("loctm_mm").cast(int) * 60,
    pl.col("loctm").str.slice(offset=4, length=2).alias("loctm_ss").cast(int),
)
# Sum up the seconds with date as seconds
df = df.with_columns(
    (
        df["locdt"] * 24 * 60 * 60 + df["loctm_hh"] + df["loctm_mm"] + df["loctm_ss"]
    ).alias("loctm")
)
# Drop the hour, minute, second
df = df.drop(
    [
        "loctm_hh",
        "loctm_mm",
        "loctm_ss",
    ]
)

# The absolute last timestamp of buying
last_time = df.group_by("chid").agg(pl.max("loctm").alias("last_time"))
df = df.join(last_time, on="chid")

# The absolute first timestamp of buying
first_time = df.group_by("chid").agg(pl.min("loctm").alias("first_time"))
df = df.join(first_time, on="chid")
# Transform "loctm" as relative timestamp to the absolute first timestamp
df = df.with_columns((df["loctm"] - df["first_time"]).alias("loctm"))

# The relative last timestamp to the relative timestamps
rel_last_time = df.group_by("chid").agg(pl.max("loctm").alias("rel_last_time"))
df = df.join(rel_last_time, on="chid")
# The period between relative last timestamp to the relative timestamps
df = df.with_columns((df["rel_last_time"] - df["loctm"]).alias("neg_loctm"))

# The previous transaction, if not, set as 0
result_df = df.group_by("chid").agg(pl.col("loctm").diff().alias("prev_loctm"))
result_df = result_df.with_columns(
    result_df["prev_loctm"].map_elements(lambda x: x[-1]).fill_null(0)
)
df = df.join(result_df, on="chid")


# The transaction span, if not, set as 0
result_df = (
    df.group_by("chid")
    .agg(pl.col("loctm").diff().mean().alias("loctm_span"))
    .fill_null(0)
)
df = df.join(result_df, on="chid")

for col in ["cano", "mchno", "acqic", "stocn", "scity"]:
    result_df = df.group_by(col).agg(pl.mean("loctm_span").alias(f"{col}_span_mean"))
    df = df.join(result_df, on=col)
    result_df = df.group_by(col).agg(pl.mean("loctm_span").alias(f"{col}_span_std"))
    df = df.join(result_df, on=col)
    df = df.with_columns(
        ((df["loctm_span"] - df[f"{col}_span_mean"]) / df[f"{col}_span_std"]).alias(
            f"{col}_span_dev"
        )
    )

# Compute the difference between real pay and conam
df = df.with_columns((df["conam"] - df["flam1"]).alias("pay_diff"))

# Count the times of each categorical column
for col in ["chid", "cano", "mchno", "acqic", "stocn", "scity"]:
    result_df = df.group_by(col).agg(pl.count().alias(f"{col}_count"))
    df = df.join(result_df, on=col)

# Count the number of cities in each country
result_df = df.group_by("stocn").agg(pl.count("scity").alias(f"scity_in_stocn_count"))
df = df.join(result_df, on="stocn")

# Mark is NA as a column
NA_cols = ["etymd", "mcc", "stocn", "scity", "stscd", "hcefg", "csmcu"]
for nc in NA_cols:
    df = df.with_columns(pl.col(nc).is_null().cast(int).alias(f"{nc}_is_NAs"))

## Preprocess id columns (group by id columns)
# Compute the mean of conam
id_class = ["chid", "cano", "mchno", "acqic"]
for col in id_class:
    result_df = df.group_by(col).agg(pl.mean("conam").alias(f"{col}_conam_mean"))
    df = df.join(result_df, on=col)
# Compute the std of conam
for col in id_class:
    result_df = df.group_by(col).agg(pl.std("conam").alias(f"{col}_conam_std"))
    df = df.join(result_df, on=col)
# Compute the deviation of conam
for col in id_class:
    df = df.with_columns(
        ((df["conam"] - df[f"{col}_conam_mean"]) / df[f"{col}_conam_std"]).alias(
            f"{col}_conam_dev"
        )
    )

# Compute the number of NA in NA existed columns
for col in id_class:
    for nc in NA_cols:
        result_df = df.group_by(col).agg(
            pl.col(nc).null_count().alias(f"{nc}_in_{col}_NAs")
        )
        df = df.join(result_df, on=col)

# Compute the number of "1" in binary class columns
bin_cols = ["ecfg", "insfg", "bnsfg", "ovrlt", "flbmk", "flg_3dsmk"]
for col in id_class:
    for bc in bin_cols:
        result_df = df.group_by(col).agg(pl.col(bc).sum().alias(f"{bc}_in_{col}_1"))
        df = df.join(result_df, on=col)

## Preprocess multiclass columns (group by these columns)
# "scity" is ignored
multiclass = ["contp", "etymd", "mcc", "stocn", "csmcu"]
# Compute the mean of conam
for col in multiclass:
    result_df = df.group_by(col).agg(pl.mean("conam").alias(f"{col}_conam_mean"))
    df = df.join(result_df, on=col)
# Compute the std of conam
for col in multiclass:
    result_df = df.group_by(col).agg(pl.std("conam").alias(f"{col}_conam_std"))
    df = df.join(result_df, on=col)
# Compute the deviation of conam
for col in multiclass:
    df = df.with_columns(
        ((df["conam"] - df[f"{col}_conam_mean"]) / df[f"{col}_conam_std"]).alias(
            f"{col}_conam_dev"
        )
    )

# Compute the number of NA in NA existed columns
for col in ["contp", "etymd", "mcc", "stocn", "csmcu"]:
    for nc in NA_cols:
        result_df = df.group_by(col).agg(
            pl.col(nc).null_count().alias(f"{nc}_in_{col}_NAs")
        )
        df = df.join(result_df, on=col)

# Compute the number of "1" in binary class columns
for col in multiclass:
    for bc in bin_cols:
        result_df = df.group_by(col).agg(pl.col(bc).sum().alias(f"{bc}_in_{col}_1"))
        df = df.join(result_df, on=col)

## Preprocess binary class columns (group by these columns)
# "bnsfg" is ignored
binary_class = ["ecfg", "insfg", "stscd", "ovrlt", "flbmk", "flg_3dsmk"]
# Compute the mean of conam
for col in binary_class:
    result_df = df.group_by(col).agg(pl.mean("conam").alias(f"{col}_conam_mean"))
    df = df.join(result_df, on=col)
# Compute the std of conam
for col in binary_class:
    result_df = df.group_by(col).agg(pl.std("conam").alias(f"{col}_conam_std"))
    df = df.join(result_df, on=col)
# Compute the deviation of conam
for col in binary_class:
    df = df.with_columns(
        ((df["conam"] - df[f"{col}_conam_mean"]) / df[f"{col}_conam_std"]).alias(
            f"{col}_conam_dev"
        )
    )

# Compute the number of NA in NA existed columns
for col in binary_class:
    for nc in NA_cols:
        result_df = df.group_by(col).agg(
            pl.col(nc).null_count().alias(f"{nc}_in_{col}_NAs")
        )
        df = df.join(result_df, on=col)

# Compute the number of "1" in binary class columns
for col in binary_class:
    for bc in bin_cols:
        result_df = df.group_by(col).agg(pl.col(bc).sum().alias(f"{bc}_in_{col}_1"))
        df = df.join(result_df, on=col)

# city in country
# df = df.with_columns(
#     ((df["conam"] - df[f"{col}_conam_mean"]) / df[f"{col}_conam_std"]).alias(
#         f"{col}_conam_dev"
#     )
# )

# Filter training data from preprocessed data
df_train = df.filter(pl.col("set") == "train")
# Add the label back to training data
df_train = df_train.with_columns(label=train_label)
# Drop the "set" column
df_train = df_train.drop(["set"])
# Check the shape
print(df_train.shape)
# Check the columns (usually for "label")
print(df_train.columns)
# Save training data as .csv
df_train.write_csv("real_train.csv", separator=",")

# Filter validation data from preprocessed data
df_valid = df.filter(pl.col("set") == "valid")
# Add the label back to validation data
df_valid = df_valid.with_columns(label=valid_label)
# Drop the "set" column
df_valid = df_valid.drop(["set"])
# Check the shape
print(df_valid.shape)
# Check the columns (usually for "label")
print(df_valid.columns)
# Save validation data as .csv
df_valid.write_csv("real_valid.csv", separator=",")

# Filter test data from preprocessed data
df_valid2 = df.filter(pl.col("set") == "valid2")
# Add the label back to validation data
df_valid2 = df_valid2.with_columns(label=valid2_label)
# Drop the "set" column
df_valid2 = df_valid2.drop(["set"])
# Check the shape (should be -1 with training/validation data)
print(df_valid2.shape)
# Check the columns (usually for "label")
print(df_valid2.columns)
# Save test data as .csv
df_valid2.write_csv("real_valid2.csv", separator=",")

# Filter test data from preprocessed data
df_test = df.filter(pl.col("set") == "test")
# Drop the "set" column
df_test = df_test.drop(["set"])
# Check the shape (should be -1 with training/validation data)
print(df_test.shape)
# Check the columns (usually for "label")
print(df_test.columns)
# Save test data as .csv
df_test.write_csv("real_test.csv", separator=",")
