import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelSummary,
    ModelCheckpoint,
)
from utils.misc import get_device_params
from pyL_modules import PyLDataModule, PyLModel


# Global config
with open("./configs/config_training.yaml", "r") as file:
    config_training = yaml.safe_load(file)
    config_training = {k: v["value"] for k, v in config_training.items()}


def train(use_wandb=True):
    # Create required directories
    model_dir = os.path.join(
        config_training["experiment_details"]["model_dir"],
        config_training["experiment_details"]["experiment_name"],
    )
    print(f"Model directory: {model_dir}")
    model_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(model_dir, exist_ok=False)

    if use_wandb:
        wandb_config = config_training["wandb_config"]
        wandb_config["notes"] = config_training["experiment_details"]["experiment_name"]
        wandb_logger = WandbLogger(**wandb_config)
        model_name = wandb_logger.experiment.name
        wandb_logger.experiment.log_code("..")
        lr_monitor = LearningRateMonitor(logging_interval="step")
    else:
        model_name = "local-test"
        wandb_logger = None
        callbacks = None

    # Create Data Module
    print("Creating Data Module")
    dataset = PyLDataModule()

    # Create Model
    print("Creating Model")
    model = PyLModel()
    model_summary = ModelSummary(max_depth=2)

    # Create callbacks
    callbacks = define_all_callbacks(model_dir, model_name)
    if use_wandb:
        callbacks.append(model_summary)
        callbacks.append(lr_monitor)

    device_params = get_device_params()
    print(f"Device parameters: {device_params}")

    # Initialize Trainer
    trainer = pl.Trainer(
        accelerator=device_params["accelerator"],
        strategy=device_params["strategy"],
        devices=device_params["devices"],
        precision=32,
        max_epochs=config_training["training_hyperparameters"]["num_epochs"],
        callbacks=callbacks,
        gradient_clip_val=1.0,
        # limit_train_batches=0.02,
        sync_batchnorm=True,
        logger=wandb_logger,
    )

    # Train the model
    trainer.fit(model, dataset)

    # Finish the experiment
    if use_wandb:
        wandb_logger.experiment.finish()


def define_all_callbacks(model_dir, model_name):
    # Create callbacks
    checkpoint_callback_1 = ModelCheckpoint(
        dirpath=model_dir,
        filename=f"{model_name}_latest",
        verbose=True,
    )
    checkpoint_callback_2 = ModelCheckpoint(
        dirpath=model_dir,
        filename=f"{model_name}_best_val_loss",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        verbose=True,
    )
    checkpoint_callback_3 = ModelCheckpoint(
        dirpath=model_dir,
        filename=f"{model_name}_best_train_loss",
        monitor="train/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        verbose=True,
    )
    checkpoint_callback_4 = ModelCheckpoint(
        dirpath=model_dir,
        filename=f"{model_name}_best_val_balanced_accuracy",
        monitor="val/balanced_accuracy",
        mode="max",
        save_top_k=1,
        save_last=False,
        verbose=True,
    )

    callbacks = [
        checkpoint_callback_1,
        checkpoint_callback_2,
        checkpoint_callback_3,
        checkpoint_callback_4,
    ]

    return callbacks


if __name__ == "__main__":
    train(use_wandb=config_training["experiment_details"]["use_wandb"])
