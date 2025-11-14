import argparse
import sys
import os

# Import internal project modules
from src.train import main as train_main
from src.predict import predict
from src.evaluate import evaluate
from src.utils import print_section


def main():
    """
    Main entry point for the Demand Forecasting project using Temporal Fusion Transformer (TFT).
    Provides a unified command-line interface for:
        - Training
        - Prediction
        - Evaluation
    """

    print_section("TEMPORAL FUSION TRANSFORMER PIPELINE")

    # ------------------------------------------------------------
    # Argument parser setup
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Demand Forecasting using Temporal Fusion Transformer (TFT)"
    )

    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "predict", "evaluate"],
        help="Choose operation mode: train | predict | evaluate",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.ckpt) for prediction/evaluation",
    )

    parser.add_argument(
        "--save_plot",
        action="store_true",
        help="Whether to save plots during evaluation (default: False)",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Dispatch actions based on mode
    # ------------------------------------------------------------
    if args.mode == "train":
        print_section("TRAINING MODE")
        train_main(args.config)

    elif args.mode == "predict":
        print_section("PREDICTION MODE")
        if args.model is None:
            print(" Please provide a model checkpoint using --model argument")
            sys.exit(1)
        predict(config_path=args.config, model_path=args.model)

    elif args.mode == "evaluate":
        print_section("EVALUATION MODE")
        if args.model is None:
            print(" Please provide a model checkpoint using --model argument")
            sys.exit(1)
        evaluate(config_path=args.config, model_path=args.model, save_plot=args.save_plot)

    else:
        print(" Invalid mode. Choose from: train | predict | evaluate")
        sys.exit(1)


if __name__ == "__main__":
    main()
