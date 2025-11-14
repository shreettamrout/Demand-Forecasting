import os
import sys
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer
from src.utils import set_seed, ensure_dir, load_config
from src.data_processing import preprocess_pipeline
from src.dataset import create_dataset, create_dataloaders
from src.model import build_tft_model, train_tft_model


def main(config_path: str = "config.yaml"):
    """
    Train a Temporal Fusion Transformer (TFT) for Demand Forecasting.
    """
    # ----------------------------------------------------------------------
    # 1. Load configuration and set up environment
    # ----------------------------------------------------------------------
    config = load_config(config_path)
    set_seed(config["training"].get("seed", 42))

    print("Starting Demand Forecasting training using Temporal Fusion Transformer...")
    print(f"Loaded configuration from: {config_path}")

    # Ensure directories exist
    ensure_dir(config["output"]["models_dir"])
    ensure_dir(config["output"]["logs_dir"])

    # ----------------------------------------------------------------------
    # 2. Load and preprocess data
    # ----------------------------------------------------------------------
    df = preprocess_pipeline(
        csv_path=config["data"]["csv_path"],
        date_col=config["data"]["date_col"],
        group_col=config["data"]["group_id"],
    )

    print(f"Data loaded and preprocessed — {len(df)} records")

    # ----------------------------------------------------------------------
    # 3. Create TimeSeriesDataSet for PyTorch Forecasting
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

    # ----------------------------------------------------------------------
    # 4. Create dataloaders (train + validation)
    # ----------------------------------------------------------------------
    train_loader, val_loader = create_dataloaders(
        dataset,
        df=df,
        time_idx=config["data"]["time_idx"],
        max_prediction_length=config["data"]["max_prediction_length"],
        batch_size=config["training"]["batch_size"],
    )

    print(f"Training samples: {len(train_loader.dataset)} | Validation samples: {len(val_loader.dataset)}")

    # ----------------------------------------------------------------------
    # 5. Build model
    # ----------------------------------------------------------------------
    tft = build_tft_model(training_dataset=dataset, config=config)

    # ----------------------------------------------------------------------
    # 6. Train model
    # ----------------------------------------------------------------------
    trainer = train_tft_model(
        tft,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        output_dir=config["output"]["models_dir"],
    )

    print("Model training completed successfully!")

    # ----------------------------------------------------------------------
    # 7. Save final checkpoint
    # ----------------------------------------------------------------------
    best_model_path = trainer.checkpoint_callback.best_model_path
    if not best_model_path:
        best_model_path = os.path.join(config["output"]["models_dir"], "final_tft_model.ckpt")
        trainer.save_checkpoint(best_model_path)

    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    # Allow custom config path from CLI
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path)
