import pandas as pd
import numpy as np
from datetime import datetime


def load_csv(path: str, date_col: str = "date") -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame and parse the date column.

    Args:
        path (str): Path to the CSV file.
        date_col (str): Column name representing the date or timestamp.

    Returns:
        pd.DataFrame: Loaded DataFrame with parsed dates.
    """
    df = pd.read_csv(path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning such as handling missing values and duplicates.

    Args:
        df (pd.DataFrame): Input raw dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing numeric columns with mean and categorical with mode
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].mean())
        elif df[col].dtype == "O":
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    return df


def basic_feature_engineering(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Create basic time-related features for forecasting models.

    Args:
        df (pd.DataFrame): Input dataframe with a date column.
        date_col (str): Name of the date column.

    Returns:
        pd.DataFrame: DataFrame with additional time-based features.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")

    df = df.sort_values(date_col).reset_index(drop=True)
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["weekday"] = df[date_col].dt.weekday
    df["weekofyear"] = df[date_col].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["dayofyear"] = df[date_col].dt.dayofyear
    return df


def add_time_idx(
    df: pd.DataFrame,
    group_col: str,
    date_col: str = "date",
    time_idx_col: str = "time_idx"
) -> pd.DataFrame:
    """
    Adds a continuous time index per group, ensuring sequential order.

    Args:
        df (pd.DataFrame): Input dataframe.
        group_col (str): Column name for grouping (e.g., store_id, product_id).
        date_col (str): Column name for the date.
        time_idx_col (str): Name of the new time index column.

    Returns:
        pd.DataFrame: DataFrame with an added 'time_idx' column.
    """
    all_groups = []
    for group_id, group_df in df.groupby(group_col):
        group_df = group_df.sort_values(date_col).reset_index(drop=True)
        group_df[time_idx_col] = np.arange(len(group_df))
        all_groups.append(group_df)
    return pd.concat(all_groups, ignore_index=True)


def add_external_features(df: pd.DataFrame, holiday_df: pd.DataFrame = None, date_col: str = "date") -> pd.DataFrame:
    """
    Optionally merges external features such as holidays or weather data.

    Args:
        df (pd.DataFrame): Main dataframe.
        holiday_df (pd.DataFrame, optional): DataFrame containing holiday dates (must include date_col).
        date_col (str): Column used for merging.

    Returns:
        pd.DataFrame: DataFrame merged with external features if provided.
    """
    if holiday_df is not None:
        holiday_df[date_col] = pd.to_datetime(holiday_df[date_col])
        df = df.merge(holiday_df, on=date_col, how="left")
        df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    return df


def preprocess_pipeline(csv_path: str, date_col: str, group_col: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Load CSV
    2. Clean data
    3. Generate time features
    4. Add continuous time index

    Args:
        csv_path (str): Path to input CSV.
        date_col (str): Name of the date column.
        group_col (str): Group identifier column.

    Returns:
        pd.DataFrame: Ready-to-train dataframe.
    """
    df = load_csv(csv_path, date_col)
    df = clean_data(df)
    df = basic_feature_engineering(df, date_col)
    df = add_time_idx(df, group_col, date_col)
    return df
