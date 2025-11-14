import os
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.metrics import MAE, RMSE
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


def build_tft_model(training_dataset, config):
    """
    Build and return a Temporal Fusion Transformer model from a given dataset.

    Args:
        training_dataset (TimeSeriesDataSet): Dataset object created by pytorch_forecasting.
        config (dict): Configuration dictionary loaded from config.yml.

    Returns:
        TemporalFusionTransformer: Untrained model ready for training.
    """

    # Define model
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config["training"].get("learning_rate", 1e-3),
        hidden_size=config["training"].get("hidden_size", 32),
        attention_head_size=config["training"].get("attention_head_size", 4),
        dropout=config["training"].get("dropout", 0.1),
        hidden_continuous_size=config["training"].get("hidden_continuous_size", 16),
        output_size=config["training"].get("output_size", 7),  # quantile outputs
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    print(f"TFT model initialized with {sum(p.numel() for p in tft.parameters() if p.requires_grad):,} trainable params.")
    return tft


def train_tft_model(tft, train_dataloader, val_dataloader, config, output_dir: str):
    """
    Train the Temporal Fusion Transformer model with logging and checkpointing.

    Args:
        tft (TemporalFusionTransformer): The model instance.
        train_dataloader (DataLoader): Training dataloader.
        val_dataloader (DataLoader): Validation dataloader.
        config (dict): Configuration settings.
        output_dir (str): Directory to store checkpoints and logs.

    Returns:
        Trainer: The PyTorch Lightning Trainer object after training.
    """

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["training"].get("early_stop_patience", 5),
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        ),
    ]

    # Logger
    logger = CSVLogger(save_dir=os.path.join(output_dir, "logs"), name="tft_training")

    # Trainer
    trainer = Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="gpu" if config["training"].get("gpus", 0) > 0 else "cpu",
        devices=config["training"].get("gpus", 0) or 1,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config["training"].get("grad_clip", 0.1),
        enable_checkpointing=True,
        log_every_n_steps=20,
    )

    print("Starting TFT model training...")
    trainer.fit(tft, train_dataloader, val_dataloader)
    print("Training complete!")

    return trainer


def evaluate_tft_model(trainer, model, val_dataloader):
    """
    Evaluate a trained TFT model on the validation set.

    Args:
        trainer (Trainer): PyTorch Lightning Trainer instance.
        model (TemporalFusionTransformer): Trained model.
        val_dataloader (DataLoader): Validation dataloader.

    Returns:
        dict: Evaluation metrics (MAE, RMSE).
    """
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    predictions = model.predict(val_dataloader)
    mae = MAE()(predictions, actuals)
    rmse = RMSE()(predictions, actuals)
    print(f"Validation MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return {"MAE": mae.item(), "RMSE": rmse.item()}
