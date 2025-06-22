"""
TorchFreya SSRL Training Pipeline
SSRL pretraining pipeline.
"""

import os
import yaml
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torchvision.models import resnet101
from torchvision.models import resnet50

from .data.datasets import FreyaDataModule
from .models.ssrl.tribyol import TriBYOL
from .models.ssrl.simclr import SimCLR
from .models.downstream.segmentation import DeepLabV3


class SSRLTrainer:
    """
    Self-Supervised Learning training pipeline for TorchFreya.
    Handles pretext task training with various SSRL methods.
    """

    def __init__(
        self,
        config: Union[str, Dict[str, Any]],
        model: Optional[L.LightningModule] = None,
        datamodule: Optional[L.LightningDataModule] = None,
        logger_type: str = "tensorboard",
        experiment_name: Optional[str] = None,
        project_name: str = "torchfreya-ssrl",
        save_dir: str = "./experiments",
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Args:
            config: Configuration dict or path to config file
            model: Pre-initialized model (optional)
            datamodule: Pre-initialized datamodule (optional)
            logger_type: Type of logger ("tensorboard", "wandb", "none")
            experiment_name: Name for this experiment
            project_name: Project name for logging
            save_dir: Directory to save experiments
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        self.config = self._load_config(config)
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name or f"ssrl_{self.config['model']['name']}"
        self.project_name = project_name
        self.resume_from_checkpoint = resume_from_checkpoint

        # Initialize components
        self.model = model or self._create_model()
        self.datamodule = datamodule or self._create_datamodule()
        self.logger = self._create_logger(logger_type)
        self.callbacks = self._create_callbacks()
        self.trainer = None

    def _load_config(self, config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from file or dict."""
        if isinstance(config, str):
            with open(config, "r") as f:
                return yaml.safe_load(f)
        return config

    def _create_model(self) -> L.LightningModule:
        """Create SSRL model from config."""
        model_config = self.config["model"]
        model_name = model_config["name"].lower()

        if model_name == "tribyol":
            # Create backbone
            backbone = self._create_backbone(model_config.get("backbone", "resnet101"))

            return TriBYOL(
                backbone=backbone,
                projector_dim=model_config.get("projector_dim", 2048),
                hidden_dim=model_config.get("hidden_dim", 512),
                output_dim=model_config.get("output_dim", 128),
                lr=model_config.get("lr", 0.03),
                momentum=model_config.get("momentum", 0.9),
                weight_decay=model_config.get("weight_decay", 0.0004),
                tau=model_config.get("tau", 0.996),
            )

        elif model_name == "simclr":
            backbone = self._create_backbone(model_config.get("backbone", "resnet101"))

            return SimCLR(
                backbone=backbone,
                projector_dim=model_config.get("projector_dim", 2048),
                hidden_dim=model_config.get("hidden_dim", 512),
                output_dim=model_config.get("output_dim", 128),
                temperature=model_config.get("temperature", 0.5),
                lr=model_config.get("lr", 1e-3),
            )

        else:
            raise ValueError(f"Unknown SSRL model: {model_name}")

    def _create_backbone(self, backbone_name: str):
        """Create backbone model."""
        if backbone_name == "resnet101":

            model = resnet101(pretrained=True)
            # Remove final classification layers for feature extraction
            model.fc = torch.nn.Identity()
            model.avgpool = torch.nn.Identity()
            return model
        elif backbone_name == "resnet50":

            model = resnet50(pretrained=True)
            model.fc = torch.nn.Identity()
            model.avgpool = torch.nn.Identity()
            return model
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

    def _create_datamodule(self) -> FreyaDataModule:
        """Create datamodule from config."""
        data_config = self.config["data"]

        return FreyaDataModule(
            root_dir=data_config["root_dir"],
            task_type="ssrl",
            batch_size=data_config.get("batch_size", 32),
            num_workers=data_config.get("num_workers", 4),
            image_size=tuple(data_config.get("image_size", [224, 224])),
            pin_memory=data_config.get("pin_memory", True),
            persistent_workers=data_config.get("persistent_workers", True),
        )

    def _create_logger(self, logger_type: str):
        """Create experiment logger."""
        if logger_type == "tensorboard":
            return TensorBoardLogger(
                save_dir=self.save_dir,
                name=self.experiment_name,
                version=None,  # Auto-increment
            )
        elif logger_type == "wandb":
            return WandbLogger(
                project=self.project_name,
                name=self.experiment_name,
                save_dir=self.save_dir,
            )
        elif logger_type == "none":
            return None
        else:
            raise ValueError(f"Unknown logger type: {logger_type}")

    def _create_callbacks(self) -> List[L.Callback]:
        """Create training callbacks."""
        callbacks = []

        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.save_dir / self.experiment_name / "checkpoints",
            filename="{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=10, mode="min", verbose=True
        )
        callbacks.append(early_stop_callback)

        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

        # Progress bar
        progress_bar = RichProgressBar()
        callbacks.append(progress_bar)

        # Model summary
        model_summary = RichModelSummary(max_depth=2)
        callbacks.append(model_summary)

        return callbacks

    def _create_trainer(self) -> L.Trainer:
        """Create Lightning trainer."""
        training_config = self.config.get("training", {})

        # Set up strategy for multi-GPU training
        strategy = None
        if training_config.get("devices", 1) > 1:
            strategy = DDPStrategy(find_unused_parameters=False)

        trainer = L.Trainer(
            max_epochs=training_config.get("max_epochs", 100),
            accelerator=training_config.get("accelerator", "auto"),
            devices=training_config.get("devices", 1),
            precision=training_config.get("precision", 32),
            logger=self.logger,
            callbacks=self.callbacks,
            strategy=strategy,
            log_every_n_steps=training_config.get("log_every_n_steps", 50),
            val_check_interval=training_config.get("val_check_interval", 1.0),
            gradient_clip_val=training_config.get("gradient_clip_val", 0.0),
            accumulate_grad_batches=training_config.get("accumulate_grad_batches", 1),
            enable_progress_bar=True,
            enable_model_summary=True,
            deterministic=training_config.get("deterministic", False),
            benchmark=training_config.get("benchmark", True),
        )

        return trainer

    def fit(self):
        """Start SSRL pretraining."""
        print(f"Starting SSRL pretraining with {self.config['model']['name']}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Save directory: {self.save_dir}")

        # Create trainer
        self.trainer = self._create_trainer()

        # Setup data
        self.datamodule.setup()

        # Start training
        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.resume_from_checkpoint,
        )

        print("SSRL pretraining completed!")
        return self.trainer.checkpoint_callback.best_model_path

    def test(self, checkpoint_path: Optional[str] = None):
        """Test the trained SSRL model."""
        if checkpoint_path:
            self.model = self.model.load_from_checkpoint(checkpoint_path)

        if self.trainer is None:
            self.trainer = self._create_trainer()

        self.datamodule.setup("test")
        results = self.trainer.test(model=self.model, datamodule=self.datamodule)
        return results


# Utility functions for quick training setup
def train_ssrl_model(
    config_path: str,
    data_dir: str,
    experiment_name: Optional[str] = None,
    logger_type: str = "tensorboard",
    resume_from: Optional[str] = None,
) -> str:
    """Quick function to train an SSRL model."""

    # Load and update config with data directory
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["data"]["root_dir"] = data_dir

    # Create and run trainer
    trainer = SSRLTrainer(
        config=config,
        logger_type=logger_type,
        experiment_name=experiment_name,
        resume_from_checkpoint=resume_from,
    )

    best_checkpoint = trainer.fit()
    return best_checkpoint
