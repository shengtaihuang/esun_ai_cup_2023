import polars as pl


def load_data(path: str, mode: str, limit: int = None):
    df = pl.read_csv(path, ignore_errors=True)
    # Transform type of the following columns
    df = df.with_columns(
        pl.col("etymd").cast(int),
        pl.col("mcc").cast(int),
        pl.col("stocn").cast(int),
        pl.col("scity").cast(int),
        pl.col("stscd").cast(pl.Categorical),
        pl.col("hcefg").cast(int),
        pl.col("csmcu").cast(int),
    )
    # Add a column "set" for split the training/validation/test data after preprocessing
    df = df.with_columns(set=pl.lit(mode))

    if limit is not None:
        df = df[:limit]

    if mode in ["train", "valid"]:
        # Store the label in another variable
        label = df["label"]
        # Drop the label to align with test data
        df = df.drop("label")
        return df, label

    return df


# Remove variables that Spearman correlation <= THRESHOLD
def drop_columns(df: pl.DataFrame):
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
            "cano_span_dev",
            "mchno_span_dev",
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
            "loctm_span",
            "cano_span_mean",
            "cano_span_std",
            "acqic_span_dev",
            "stocn_span_dev",
            "scity_span_dev",
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
    return df
