"""
前處理資料
training, public 前處理包含 x 和 y
private_1_processed 前處理只包含 x
"""

import polars as pl

# Add this to avoid error while concatenate string columns
pl.enable_string_cache()
# Preprocess training data
df_train = pl.read_csv("dataset_1st/training.csv", ignore_errors=True)
# df_train = df_train[:10000]
# Transform type of the following columns
df_train = df_train.with_columns(
    pl.col("etymd").cast(int),
    pl.col("mcc").cast(int),
    pl.col("stocn").cast(int),
    pl.col("scity").cast(int),
    pl.col("stscd").cast(pl.Categorical),
    pl.col("hcefg").cast(int),
    pl.col("csmcu").cast(int),
)
# Store the training label in another variable
train_label = df_train["label"]
# Drop the training label to align with test data
df_train = df_train.drop("label")
# Add a column "set" for split the training/validation/test data after preprocessing
df_train = df_train.with_columns(set=pl.lit("train"))

# Preprocess validation data
df_valid = pl.read_csv("dataset_2nd/public.csv", ignore_errors=True)
# df_valid = df_valid[:10000]
# Transform type of the following columns
df_valid = df_valid.with_columns(
    pl.col("etymd").cast(int),
    pl.col("mcc").cast(int),
    pl.col("stocn").cast(int),
    pl.col("scity").cast(int),
    pl.col("stscd").cast(pl.Categorical),
    pl.col("hcefg").cast(int),
    pl.col("csmcu").cast(int),
)
# Store the validation label in another variable
valid_label = df_valid["label"]
# Drop the validation label to align with test data
df_valid = df_valid.drop("label")
# Add a column "set" for split the training/validation/test data after preprocessing
df_valid = df_valid.with_columns(set=pl.lit("valid"))

# Preprocess test data
df_test = pl.read_csv("dataset_2nd/private_1_processed.csv", ignore_errors=True)
# df_test = df_test[:10000]
# Transform type of the following columns
df_test = df_test.with_columns(
    pl.col("etymd").cast(int),
    pl.col("mcc").cast(int),
    pl.col("stocn").cast(int),
    pl.col("scity").cast(int),
    pl.col("stscd").cast(pl.Categorical),
    pl.col("hcefg").cast(int),
    pl.col("csmcu").cast(int),
)
# Add a column "set" for split the training/validation/test data after preprocessing
df_test = df_test.with_columns(set=pl.lit("test"))

# Concatenate training/validation/test to preprocess them together
df = pl.concat([df_train, df_valid, df_test])

df = df.with_columns(
    pl.col("etymd").cast(pl.Utf8).cast(pl.Categorical),
    pl.col("mcc").cast(pl.Utf8).cast(pl.Categorical),
    pl.col("stocn").cast(pl.Utf8).cast(pl.Categorical),
    pl.col("scity").cast(pl.Utf8).cast(pl.Categorical),
    # pl.col("stscd").cast(pl.Utf8).cast(pl.Categorical),
    pl.col("hcefg").cast(pl.Utf8).cast(pl.Categorical),
    pl.col("csmcu").cast(pl.Utf8).cast(pl.Categorical),
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
result_df = (
    df.group_by("chid")
    .agg(pl.col("loctm").diff().mean().alias("prev_loctm"))
    .fill_null(0)
)
df = df.join(result_df, on="chid")

# Com
df = df.with_columns((df["conam"] - df["flam1"]).alias("pay_diff"))

# Count the times of each categorical column
for col in ["chid", "cano", "mchno", "acqic", "stocn", "scity"]:
    result_df = df.group_by(col).agg(pl.count().alias(f"{col}_count"))
    df = df.join(result_df, on=col)

# Count the number of cities in each country
result_df = df.group_by("stocn").agg(pl.count("scity").alias(f"scity_in_stocn_count"))
df = df.join(result_df, on="stocn")

# Compute the number of NA in NA existed columns
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
        if col == bin:
            continue
        result_df = df.group_by(col).agg(pl.col(bc).sum().alias(f"{bc}_in_{col}_1"))
        df = df.join(result_df, on=col)

# city in country
# df = df.with_columns(
#     ((df["conam"] - df[f"{col}_conam_mean"]) / df[f"{col}_conam_std"]).alias(
#         f"{col}_conam_dev"
#     )
# )

# Drop id columns
# Drop too many classes columns "stocn", "scity"
# Drop insfg -> 看分期期數就好
# Drop timestamps (assumed to be irrelavant)
# Drop "flam1", "csmcu" (highly relative to "conam")
df = df.drop(
    [
        "locdt",
        "loctm",
        "chid",
        "cano",
        "mchno",
        "acqic",
        "insfg",
        "mcc",
        "flam1",
        "stocn",
        "scity",
        "csmcu",
    ]
)

# Remove variables that Spearman correlation <= THRESHOLD
# Remove variables corr <= 0
df = df.drop(
    [
        "flbmk_conam_mean",
        "mcc_conam_std",
        "flbmk_conam_std",
        "chid_conam_dev",
        "cano_conam_dev",
        "stscd_conam_dev",
        "etymd_in_contp_NAs",
        "etymd_in_flbmk_NAs",
        "mcc_in_chid_NAs",
        "mcc_in_cano_NAs",
        "mcc_in_mchno_NAs",
        "mcc_in_contp_NAs",
        "mcc_in_mcc_NAs",
        "mcc_in_flbmk_NAs",
        "stocn_in_chid_NAs",
        "stocn_in_cano_NAs",
        "stocn_in_contp_NAs",
        "stocn_in_flbmk_NAs",
        "scity_in_contp_NAs",
        "scity_in_csmcu_NAs",
        "scity_in_flbmk_NAs",
        "stscd_in_contp_NAs",
        "stscd_in_etymd_NAs",
        "stscd_in_flbmk_NAs",
        "hcefg_in_mchno_NAs",
        "hcefg_in_contp_NAs",
        "hcefg_in_flbmk_NAs",
        "csmcu_in_contp_NAs",
        "csmcu_in_flbmk_NAs",
        "ecfg_in_contp_1",
        "ecfg_in_flbmk_1",
        "insfg_in_contp_1",
        "insfg_in_flbmk_1",
        "bnsfg_in_contp_1",
        "bnsfg_in_flbmk_1",
        "ovrlt_in_mchno_1",
        "ovrlt_in_contp_1",
        "ovrlt_in_etymd_1",
        "ovrlt_in_flbmk_1",
        "flbmk_in_chid_1",
        "flbmk_in_cano_1",
        "flbmk_in_contp_1",
        "flbmk_in_etymd_1",
        "flbmk_in_flbmk_1",
        "flg_3dsmk_in_contp_1",
        "flg_3dsmk_in_flbmk_1",
    ]
)

# Remove variables |corr| <= 0.01
df = df.drop(
    [
        "mchno_count",
        "mchno_conam_mean",
        "acqic_conam_mean",
        "contp_conam_mean",
        "etymd_conam_mean",
        "mcc_conam_mean",
        "insfg_conam_mean",
        "flg_3dsmk_conam_mean",
        "mchno_conam_std",
        "contp_conam_std",
        "etymd_conam_std",
        "insfg_conam_std",
        "flg_3dsmk_conam_std",
        "etymd_in_mchno_NAs",
        "etymd_in_etymd_NAs",
        "etymd_in_mcc_NAs",
        "etymd_in_insfg_NAs",
        "etymd_in_flg_3dsmk_NAs",
        "mcc_in_acqic_NAs",
        "mcc_in_etymd_NAs",
        "mcc_in_insfg_NAs",
        "mcc_in_flg_3dsmk_NAs",
        "stocn_in_stocn_NAs",
        "stocn_in_insfg_NAs",
        "stocn_in_flg_3dsmk_NAs",
        "scity_in_mchno_NAs",
        "scity_in_acqic_NAs",
        "scity_in_etymd_NAs",
        "scity_in_insfg_NAs",
        "scity_in_flg_3dsmk_NAs",
        "stscd_in_mchno_NAs",
        "stscd_in_insfg_NAs",
        "stscd_in_flg_3dsmk_NAs",
        "hcefg_in_acqic_NAs",
        "hcefg_in_insfg_NAs",
        "hcefg_in_flg_3dsmk_NAs",
        "csmcu_in_mchno_NAs",
        "csmcu_in_acqic_NAs",
        "csmcu_in_etymd_NAs",
        "csmcu_in_csmcu_NAs",
        "csmcu_in_insfg_NAs",
        "csmcu_in_flg_3dsmk_NAs",
        "ecfg_in_insfg_1",
        "ecfg_in_flg_3dsmk_1",
        "insfg_in_mchno_1",
        "insfg_in_etymd_1",
        "insfg_in_insfg_1",
        "insfg_in_flg_3dsmk_1",
        "bnsfg_in_chid_1",
        "bnsfg_in_cano_1",
        "bnsfg_in_mchno_1",
        "bnsfg_in_insfg_1",
        "bnsfg_in_flg_3dsmk_1",
        "ovrlt_in_chid_1",
        "ovrlt_in_cano_1",
        "ovrlt_in_insfg_1",
        "ovrlt_in_flg_3dsmk_1",
        "flbmk_in_mchno_1",
        "flbmk_in_insfg_1",
        "flbmk_in_flg_3dsmk_1",
        "flg_3dsmk_in_insfg_1",
        "flg_3dsmk_in_flg_3dsmk_1",
    ]
)

# Remove variables |corr| <= 0.02
df = df.drop(
    [
        "ovrlt_conam_mean",
        "chid_conam_std",
        "cano_conam_std",
        "acqic_conam_std",
        "ovrlt_conam_std",
        "acqic_conam_dev",
        "ovrlt_conam_dev",
        "flbmk_conam_dev",
        "etymd_in_cano_NAs",
        "etymd_in_acqic_NAs",
        "etymd_in_ovrlt_NAs",
        "mcc_in_ovrlt_NAs",
        "stocn_in_mchno_NAs",
        "stocn_in_etymd_NAs",
        "stocn_in_ovrlt_NAs",
        "scity_in_chid_NAs",
        "scity_in_cano_NAs",
        "scity_in_ovrlt_NAs",
        "stscd_in_ovrlt_NAs",
        "hcefg_in_chid_NAs",
        "hcefg_in_cano_NAs",
        "hcefg_in_ovrlt_NAs",
        "csmcu_in_ovrlt_NAs",
        "ecfg_in_mchno_1",
        "ecfg_in_ovrlt_1",
        "insfg_in_chid_1",
        "insfg_in_cano_1",
        "insfg_in_ovrlt_1",
        "bnsfg_in_etymd_1",
        "bnsfg_in_ovrlt_1",
        "ovrlt_in_ovrlt_1",
        "flbmk_in_ovrlt_1",
        "flg_3dsmk_in_mchno_1",
        "flg_3dsmk_in_ovrlt_1",
    ]
)

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
df_train.write_csv("train.csv", separator=",")

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
df_valid.write_csv("valid.csv", separator=",")

# Filter test data from preprocessed data
df_test = df.filter(pl.col("set") == "test")
# Drop the "set" column
df_test = df_test.drop(["set"])
# Check the shape (should be -1 with training/validation data)
print(df_test.shape)
# Check the columns (usually for "label")
print(df_valid.columns)
# Save test data as .csv
df_test.write_csv("test.csv", separator=",")
