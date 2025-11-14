import os
import sys
import torch
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer
from src.utils import load_config, ensure_dir, print_section
from src.data_processing import preprocess_pipeline
from src.dataset import create_dataset
from torch.utils.data import DataLoader


def predict(config_path: str = "config.yaml", model_path: str = None, output_path: str = None):
    """
    Generate future demand forecasts using a trained Temporal Fusion Transformer (TFT) model.

    Args:
        config_path (str): Path to YAML configuration file.
        model_path (str): Path to trained model checkpoint.
        output_path (str): Optional output path for predictions CSV.
    """
    # ----------------------------------------------------------------------
    # 1. Load config and prepare directories
    # ----------------------------------------------------------------------
    config = load_config(config_path)
    print_section("LOAD CONFIGURATION")

    ensure_dir(config["output"]["predictions_dir"])

    if model_path is None:
        raise ValueError(" Please provide a valid model checkpoint path using --model_path argument")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # ----------------------------------------------------------------------
    # 2. Load and preprocess data
    # ----------------------------------------------------------------------
    print_section("PREPROCESSING DATA")
    df = preprocess_pipeline(
        csv_path=config["data"]["csv_path"],
        date_col=config["data"]["date_col"],
        group_col=config["data"]["group_id"],
    )
    print(f" Loaded {len(df)} rows from {config['data']['csv_path']}")

    # ----------------------------------------------------------------------
    # 3. Rebuild the dataset for prediction
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

    dataloader = dataset.to_dataloader(train=False, batch_size=config["training"]["batch_size"], num_workers=4)
    print(f" Created prediction DataLoader with {len(dataloader.dataset)} samples")

    # ----------------------------------------------------------------------
    # 4. Load model checkpoint
    # ----------------------------------------------------------------------
    print_section("LOADING TRAINED MODEL")
    model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print(" Using GPU for inference")
    else:
        print(" Using CPU for inference")

    # ----------------------------------------------------------------------
    # 5. Run inference
    # ----------------------------------------------------------------------
    print_section("RUNNING INFERENCE")
    predictions = model.predict(dataloader, return_x=True)

    # Extract forecasts and metadata
    y_pred = predictions.output.detach().cpu().numpy().squeeze()
    x_input = predictions.x

    # Extract identifiers and time info for mapping
    group_ids = [x_input["group_ids"][i][0] for i in range(len(x_input["group_ids"]))]
    time_idx = [x_input["decoder_time_idx"][i][-1].item() for i in range(len(x_input["decoder_time_idx"]))]

    pred_df = pd.DataFrame({
        config["data"]["group_id"]: group_ids,
        config["data"]["time_idx"]: time_idx,
        f"{config['data']['target']}_forecast": y_pred
    })

    # ----------------------------------------------------------------------
    # 6. Save predictions
    # ----------------------------------------------------------------------
    ensure_dir(config["output"]["predictions_dir"])
    output_file = output_path or os.path.join(
        config["output"]["predictions_dir"], "tft_predictions.csv"
    )
    pred_df.to_csv(output_file, index=False)
    print_section("PREDICTION RESULTS")
    print(f"Saved predictions to: {output_file}")
    print(pred_df.head())


if __name__ == "__main__":
    # Usage:
    # python -m src.predict config.yaml models/checkpoints/tft-best.ckpt
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    predict(config_path=config_path, model_path=model_path)
