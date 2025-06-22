#!/usr/bin/env python3
"""
TorchFreya CLI Scripts
Command-line interfaces for training SSRL and downstream models.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch


def downstream_train_cli():
    """Command-line interface for downstream task training."""
    parser = argparse.ArgumentParser(
        description="TorchFreya Downstream Task Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to downstream training configuration file",
    )

    parser.add_argument(
        "--data", "-d", type=str, required=True, help="Path to dataset directory"
    )

    parser.add_argument(
        "--ssrl-checkpoint",
        type=str,
        required=True,
        help="Path to SSRL pre-trained checkpoint",
    )

    # Optional arguments
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided)",
    )

    parser.add_argument(
        "--save-dir",
        "-s",
        type=str,
        default="./experiments",
        help="Directory to save experiments",
    )

    parser.add_argument(
        "--logger",
        type=str,
        choices=["tensorboard", "wandb", "none"],
        default="tensorboard",
        help="Logger type",
    )

    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    parser.add_argument(
        "--devices", type=int, default=1, help="Number of devices to use"
    )

    parser.add_argument(
        "--precision",
        type=str,
        choices=["16", "32", "bf16"],
        default="32",
        help="Training precision",
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum number of epochs (overrides config)",
    )

    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size (overrides config)"
    )

    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )

    parser.add_argument(
        "--freeze-backbone", action="store_true", help="Freeze backbone during training"
    )

    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=0,
        help="Number of epochs to freeze backbone",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="torchfreya-downstream",
        help="Weights & Biases project name",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loader workers"
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Validate SSRL checkpoint exists
    if not Path(args.ssrl_checkpoint).exists():
        print(f"Error: SSRL checkpoint not found: {args.ssrl_checkpoint}")
        sys.exit(1)

    # Load and modify config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Update config with CLI arguments
    config["data"]["root_dir"] = args.data
    config["data"]["num_workers"] = args.num_workers

    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size

    if args.max_epochs:
        config["training"]["max_epochs"] = args.max_epochs

    if args.learning_rate:
        config["model"]["learning_rate"] = args.learning_rate

    config["training"]["devices"] = args.devices
    config["training"]["precision"] = args.precision
    config["model"]["freeze_backbone"] = args.freeze_backbone
    config["model"]["freeze_epochs"] = args.freeze_epochs

    # Import here to avoid circular imports
    from torchfreya.training.downstream_trainer import DownstreamTrainer

    # Create trainer
    trainer = DownstreamTrainer(
        config=config,
        ssrl_checkpoint_path=args.ssrl_checkpoint,
        logger_type=args.logger,
        experiment_name=args.name,
        project_name=args.wandb_project,
        save_dir=args.save_dir,
        resume_from_checkpoint=args.resume,
    )

    # Start training
    try:
        best_checkpoint = trainer.fit()
        print(f"\nDownstream training completed successfully!")
        print(f"Best checkpoint: {best_checkpoint}")

        # Run test if test data exists
        if Path(args.data).joinpath("images", "test").exists():
            print("\nRunning test evaluation...")
            test_results = trainer.test()
            print(f"Test results: {test_results}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        sys.exit(1)
