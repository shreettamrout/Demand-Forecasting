import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pytorch_forecasting import TemporalFusionTransformer
from src.utils import load_config, print_section, log_metrics
from src.data_processing import preprocess_pipeline
from src.dataset import create_dataset, create_dataloaders


def evaluate(config_path: str = "config.yaml", model_path: str = None, save_plot: bool = True):
    """
    Evaluate a trained Temporal Fusion Transformer (TFT) model on validation data.

    Args:
        config_path (str): Path to the YAML configuration file.
        model_path (str): Path to model checkpoint.
        save_plot (bool): Whether to save a plot comparing actual vs. predicted values.
    """
    # ----------------------------------------------------------------------
    # 1. Load configuration and verify paths
    # ----------------------------------------------------------------------
    config = load_config(config_path)
    print_section("LOAD CONFIGURATION")

    if model_path is None:
        raise ValueError("Please provide model checkpoint path using --model_path argument")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f" Model checkpoint not found: {model_path}")

    # ----------------------------------------------------------------------
    # 2. Load and preprocess dataset
    # ----------------------------------------------------------------------
    print_section("PREPROCESSING DATA")
    df = preprocess_pipeline(
        csv_path=config["data"]["csv_path"],
        date_col=config["data"]["date_col"],
        group_col=config["data"]["group_id"],
    )
    print(f"Loaded {len(df)} rows from {config['data']['csv_path']}")

    # ----------------------------------------------------------------------
    # 3. Create dataset and dataloaders
    # ----------------------------------------------------------------------
    dataset = create_dataset(
        df=df,
        time_idx=config["data"]["time_idx"],
        target=config["data"]["target"],
        group_id=config["data"]["group_id"],
        min_encoder_length=config["data"]["min_encoder_length"],
        max_encoder_length=config["data"]["max_encoder_length"],
        max_prediction_length=config["data"]["max_prediction_length"],
        static_categoricals=config["data"].get("static_categoricals", []),
        static_reals=config["data"].get("static_reals", []),
        time_varying_known_categoricals=config["data"].get("time_varying_known_categoricals", []),
        time_varying_known_reals=config["data"].get("time_varying_known_reals", []),
        time_varying_unknown_reals=config["data"].get("time_varying_unknown_reals", [config["data"]["target"]]),
    )

    train_loader, val_loader = create_dataloaders(
        dataset,
        df=df,
        time_idx=config["data"]["time_idx"],
        max_prediction_length=config["data"]["max_prediction_length"],
        batch_size=config["training"]["batch_size"],
    )

    # ----------------------------------------------------------------------
    # 4. Load model checkpoint
    # ----------------------------------------------------------------------
    print_section("LOADING TRAINED MODEL")
    model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print(" Using GPU for evaluation")
    else:
        print("Using CPU for evaluation")

    # ----------------------------------------------------------------------
    # 5. Generate predictions on validation data
    # ----------------------------------------------------------------------
    print_section("RUNNING VALIDATION PREDICTIONS")
    actuals = torch.cat([y[0] for x, y in iter(val_loader)]).detach().cpu().numpy()
    predictions = model.predict(val_loader).detach().cpu().numpy().squeeze()

    # ----------------------------------------------------------------------
    # 6. Compute evaluation metrics
    # ----------------------------------------------------------------------
    mae = mean_absolute_error(actuals, predictions)
    rmse = mean_squared_error(actuals, predictions, squared=False)
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}
    log_metrics(metrics)

    # ----------------------------------------------------------------------
    # 7. Visualization
    # ----------------------------------------------------------------------
    if save_plot:
        print_section("PLOTTING RESULTS")
        plt.figure(figsize=(10, 5))
        plt.plot(actuals[:200], label="Actual", linewidth=2)
        plt.plot(predictions[:200], label="Predicted", linewidth=2)
        plt.title("TFT Forecasting — Actual vs Predicted (First 200 samples)")
        plt.xlabel("Sample index")
        plt.ylabel(config["data"]["target"].capitalize())
        plt.legend()
        plt.grid(True)

        output_dir = os.path.join(config["output"]["predictions_dir"], "evaluation_plot.png")
        os.makedirs(config["output"]["predictions_dir"], exist_ok=True)
        plt.savefig(output_dir, bbox_inches="tight")
        plt.close()
        print(f" Saved evaluation plot to {output_dir}")

    # ----------------------------------------------------------------------
    # 8. Save metrics
    # ----------------------------------------------------------------------
    metrics_path = os.path.join(config["output"]["predictions_dir"], "evaluation_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    # Usage:
    # python -m src.evaluate config.yaml models/checkpoints/tft-best.ckpt
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    evaluate(config_path=config_path, model_path=model_path)
