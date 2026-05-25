import yaml
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import Accuracy

from custom_dataset import PlantPathologyDataset
from model import BaselineModel
from utils.custom_scheduluer import NoamScheduler


# Global config
with open("./configs/config_training.yaml", "r") as file:
    config_training = yaml.safe_load(file)
    config_training = {k: v["value"] for k, v in config_training.items()}


class PyLDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.base_dataset_path = config_training["dataset_configs"]["base_dataset_path"]
        self.dataset_name = config_training["dataset_configs"]["plantpathology"]

    def setup(self, stage=None):
        self.train_set = PlantPathologyDataset(
            stage="train", base_dataset_path=self.base_dataset_path, dataset_name=self.dataset_name
        )
        self.val_set = PlantPathologyDataset(
            stage="val", base_dataset_path=self.base_dataset_path, dataset_name=self.dataset_name
        )

    def train_dataloader(self):
        sampler = WeightedRandomSampler(
            weights=self.train_set.get_class_weights(),
            num_samples=len(self.train_set),
            replacement=True,
        )

        return DataLoader(
            self.train_set,
            batch_size=config_training["training_hyperparameters"]["batch_size"],
            sampler=sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=config_training["training_hyperparameters"]["batch_size"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=4,
            persistent_workers=True,
        )


class PyLModel(pl.LightningModule):
    def __init__(self, wandb_logger=None):
        super().__init__()
        self.save_hyperparameters()
        self.wandb_logger = wandb_logger

        self.LABEL_ENCODING = config_training["plant_label_encoding"]
        self.LABEL_DECODING = {v: k for k, v in self.LABEL_ENCODING.items()}

        num_classes = len(self.LABEL_ENCODING)
        self.model = BaselineModel(num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.val_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="micro"
        )
        self.val_balanced_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_per_class_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )

    def training_step(self, batch, batch_idx):
        inputs = batch["image"]
        labels = batch["label"]

        _, outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels).float() / len(labels)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/accuracy", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        labels = batch["label"]

        _, outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        self.val_acc.update(preds, labels)
        self.val_balanced_acc.update(preds, labels)
        self.val_per_class_acc.update(preds, labels)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        balanced_acc = self.val_balanced_acc.compute()
        per_class_acc = self.val_per_class_acc.compute()

        self.log("val/accuracy", val_acc, prog_bar=True)
        self.log("val/balanced_accuracy", balanced_acc, prog_bar=True)

        for class_idx, acc in enumerate(per_class_acc):
            class_name = self.LABEL_DECODING[class_idx]
            self.log(f"val/accuracy_{class_name}", acc)

        # Reset metrics for next epoch
        self.val_acc.reset()
        self.val_balanced_acc.reset()
        self.val_per_class_acc.reset()

    def predict_step(self, batch, batch_idx):
        # Get inputs
        inputs = batch["image"]
        labels = batch["label"]
        ids = batch["id"]

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            emb, logits = self.model(inputs)

        outputs = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(outputs, dim=1)

        true_label_name = [
            self.LABEL_DECODING[label.item()]
            if label.item() in self.LABEL_DECODING
            else "Unknown"
            for label in labels
        ]
        pred_label_name = [self.LABEL_DECODING[pred.item()] for pred in preds]

        return {
            "id": ids,
            "predicted_label": pred_label_name,
            "true_label": true_label_name,
            "embedding": emb,
            "outputs": outputs,
            "logits": logits,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config_training["training_hyperparameters"]["learning_rate"],
            weight_decay=config_training["training_hyperparameters"]["weight_decay"],
        )
        scheduler = NoamScheduler(
            optimizer,
            warmup_steps=config_training["training_hyperparameters"]["warmup_steps"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    dataset = PyLDataModule(dataset_path="./dataset/")
    model = PyLModel()
    breakpoint()
