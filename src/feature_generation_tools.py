import itertools

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MaxAbsScaler, OrdinalEncoder

from src.commons import *


# Check unique values in columns
def verify_unique_values(df, y_col_name):
    target_cols = list(df.columns)
    if y_col_name in target_cols:
        target_cols.remove(y_col_name)

    unique_values = df[target_cols].apply(np.unique, axis=0)
    output_df = pd.DataFrame(unique_values, columns=["unique_values"])
    output_df["total"] = unique_values.apply(np.shape).apply(np.unique, axis=0)

    # pd.concat([a,b], axis=1)

    return output_df


# Encode features into Ordinal Encoding
def encode_features(df):
    OE = OrdinalEncoder()
    df_e = OE.fit_transform(df)

    logging.debug(
        f"Encoded features: {pd.DataFrame(df_e, index=df.index, columns=df.columns)}"
    )

    return df_e


# Scale features with MaxAbsScaler and classes into Ordinal Encoding
def scale_features(df):
    max_abs_scaler = MaxAbsScaler()
    df_e = max_abs_scaler.fit_transform(df)

    logging.debug(
        f"Scaled features: {pd.DataFrame(df_e, index=df.index, columns=df.columns)}"
    )

    return df_e


# Split double features into separate features
def split_features(source_df, y_col_name):
    df = source_df.copy()

    target_cols = list(df.columns)
    if y_col_name in target_cols:
        df = df.drop([y_col_name], axis=1)
        target_cols.remove(y_col_name)

    rows = df.values.shape[0]
    logging.info(f"Transforming DF with shape {df.values.shape}")

    values = np.array(df.values).flatten().astype(str)
    values = np.char.replace(values, " ", "")
    values = np.char.replace(values, "-", "--")
    new_arr = np.array(list(itertools.chain.from_iterable(values))).reshape(rows, -1)
    logging.info(f"Resulting array with shape {new_arr.shape}")

    col_names = (
        np.array(list(map(lambda x: (x, x + ".1"), target_cols))).astype(str).flatten()
    )
    logging.info(
        f"Generated {len(col_names)} column names from {len(target_cols)} columns"
    )

    new_df = pd.DataFrame(new_arr, columns=col_names, index=source_df.index)
    logging.info(f"Resulting DF with shape {new_df.shape}")

    return new_df


# Select features by removing features with no variance
def select_by_variance(X_e):
    selector = VarianceThreshold()
    selector_f = selector.fit(X_e)
    X_e_transformed = selector_f.transform(X_e)
    X_e_transformed_feature_names = selector_f.get_support()

    logging.info(f"Original data shape: {X_e.shape}")
    logging.info(f"Selected data shape: {X_e_transformed.shape}")
    diff = X_e.shape[1] - X_e_transformed.shape[1]
    logging.info(
        f"Diff: {diff} features, Excluded proportion: {diff/X_e.shape[1]}, Resulting proportion: {1 - diff/X_e.shape[1]}"
    )

    return X_e_transformed, X_e_transformed_feature_names


def prepare_dataset(X, y, scale=None):
    logging.info(f"""Preparing datasets of shape: {X.shape}, {y.shape}""")

    numeric_cols = X.select_dtypes(np.number).columns
    object_cols = X.select_dtypes(object).columns

    if len(object_cols) > 0 or scale == False:
        # Encode
        X_e = encode_features(X)
        y_e = encode_features(y)
    else:
        X_e = scale_features(X)
        y_e = encode_features(y)

    # Remove stale features
    X_e_transformed, X_e_transformed_feature_names = select_by_variance(X_e)

    logging.info(f"""Current classes: {y["y"].unique()}""")

    feature_names = X.columns[X_e_transformed_feature_names]

    return X_e_transformed, y_e, feature_names
