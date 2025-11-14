from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import pandas as pd


def create_dataset(
    df: pd.DataFrame,
    time_idx: str,
    target: str,
    group_id: str,
    min_encoder_length: int,
    max_encoder_length: int,
    max_prediction_length: int,
    static_categoricals=None,
    static_reals=None,
    time_varying_known_categoricals=None,
    time_varying_known_reals=None,
    time_varying_unknown_reals=None,
):
    """
    Create a PyTorch Forecasting TimeSeriesDataSet for Temporal Fusion Transformer (TFT).

    Args:
        df (pd.DataFrame): Input time series dataframe.
        time_idx (str): Column name for time index (integer, increasing).
        target (str): Target column name (the value to forecast).
        group_id (str): Column name for the series identifier (e.g., store_id, product_id).
        min_encoder_length (int): Minimum sequence length for encoder.
        max_encoder_length (int): Maximum sequence length for encoder.
        max_prediction_length (int): Number of time steps to forecast.
        static_categoricals (list, optional): List of static categorical feature names.
        static_reals (list, optional): List of static continuous feature names.
        time_varying_known_categoricals (list, optional): List of known categorical features.
        time_varying_known_reals (list, optional): List of known continuous features.
        time_varying_unknown_reals (list, optional): List of unknown continuous features (typically includes target).

    Returns:
        TimeSeriesDataSet: Ready-to-use dataset for model training or validation.
    """

    # Default empty lists if None
    static_categoricals = static_categoricals or []
    static_reals = static_reals or []
    time_varying_known_categoricals = time_varying_known_categoricals or []
    time_varying_known_reals = time_varying_known_reals or []
    time_varying_unknown_reals = time_varying_unknown_reals or [target]

    # Define the TimeSeriesDataSet
    dataset = TimeSeriesDataSet(
        df,
        time_idx=time_idx,
        target=target,
        group_ids=[group_id],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            group_ids=[group_id], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    return dataset


def create_dataloaders(
    dataset: TimeSeriesDataSet, df: pd.DataFrame, time_idx: str, max_prediction_length: int, batch_size: int = 64
):
    """
    Split the dataset into train and validation dataloaders.
    """
    from torch.utils.data import DataLoader

    train_cutoff = df[time_idx].max() - max_prediction_length
    training = TimeSeriesDataSet.from_parameters(
        dataset.get_parameters(), df[lambda x: x[time_idx] <= train_cutoff]
    )
    validation = TimeSeriesDataSet.from_parameters(
        dataset.get_parameters(), df[lambda x: x[time_idx] > train_cutoff]
    )

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

    return train_loader, val_loader
