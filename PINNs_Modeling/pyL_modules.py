import yaml
import wandb
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from custom_dataset import Cylinder2DDataset
from model import BaselineModel


# Global config
with open("./configs/config_training.yaml", "r") as file:
    config_training = yaml.safe_load(file)
    config_training = {k: v["value"] for k, v in config_training.items()}


def collate_single(batch):
    return batch[0]


class PyLDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset_configs = config_training["dataset_configs"]

    def setup(self, stage=None):
        self.train_set = Cylinder2DDataset(**self.dataset_configs)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=config_training["training_hyperparameters"]["batch_size"],
            pin_memory=False,
            drop_last=True,
            num_workers=4,
            persistent_workers=True,
            collate_fn=collate_single,
        )


class PyLModel(pl.LightningModule):
    def __init__(self, wandb_logger=None):
        super().__init__()
        self.save_hyperparameters(ignore=["wandb_logger"])
        self.wandb_logger = wandb_logger

        self.model = BaselineModel(
            domain_bounds=config_training["dataset_configs"]["domain_bounds"],
            **config_training["model_architecture_hyperparameters"],
            **config_training.get("exploratory_variables", {}),
        )
        self.lambda_physics = config_training["loss_weights"]["lambda_physics"]
        self.lambda_bc = config_training["loss_weights"]["lambda_bc"]
        self.lambda_ic = config_training["loss_weights"]["lambda_ic"]
        self.nu = config_training["dataset_configs"]["nu"]

    def bc_loss(self, bc_points):
        # inlet (ui = Schäfer-Turek parabolic, vi = 0)
        xi, yi, ti = bc_points["inlet"]
        ui_pred, vi_pred, _ = self.model(xi, yi, ti)
        ui_true = bc_points["inlet_u"]
        inlet_loss = torch.mean((ui_pred - ui_true) ** 2) + torch.mean(vi_pred**2)

        # outlet (po = 0)
        xo, yo, to = bc_points["outlet"]
        _, _, po_pred = self.model(xo, yo, to)
        outlet_loss = torch.mean(po_pred**2)

        # no-slip walls (ui = vi = 0)
        xw, yw, tw = bc_points["noslip"]
        ui_pred_w, vi_pred_w, _ = self.model(xw, yw, tw)
        noslip_loss = torch.mean(ui_pred_w**2) + torch.mean(vi_pred_w**2)

        return inlet_loss, outlet_loss, noslip_loss

    def ic_loss(self, ic_points):
        # initial condition: ui = vi = 0 at t=0
        x, y, t = ic_points
        ui_pred, vi_pred, _ = self.model(x, y, t)
        ic_loss = torch.mean(ui_pred**2) + torch.mean(vi_pred**2)
        return ic_loss

    def physics_loss(self, collocation_points):
        x, y, t = collocation_points
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)
        u, v, p = self.model(x, y, t)

        nu = self.nu

        def grad(f, var):
            return torch.autograd.grad(
                f, var, grad_outputs=torch.ones_like(f), create_graph=True
            )[0]

        # first order
        u_x = grad(u, x)
        u_y = grad(u, y)
        u_t = grad(u, t)
        v_x = grad(v, x)
        v_y = grad(v, y)
        v_t = grad(v, t)
        p_x = grad(p, x)
        p_y = grad(p, y)

        # second order
        u_xx = grad(u_x, x)
        u_yy = grad(u_y, y)
        v_xx = grad(v_x, x)
        v_yy = grad(v_y, y)

        continuity = u_x + v_y
        momentum_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        momentum_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

        return (
            torch.mean(continuity**2)
            + torch.mean(momentum_u**2)
            + torch.mean(momentum_v**2)
        )

    def training_step(self, batch, batch_idx):
        collocation_points = batch["collocation"]
        bc_points = batch["boundary"]
        ic_points = batch["ic"]

        inlet_loss, outlet_loss, noslip_loss = self.bc_loss(bc_points)
        bc_loss = inlet_loss + outlet_loss + noslip_loss
        ic_loss = self.ic_loss(ic_points)
        physics_loss = self.physics_loss(collocation_points)

        loss = (
            self.lambda_physics * physics_loss
            + self.lambda_bc * bc_loss
            + self.lambda_ic * ic_loss
        )

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1
        )

        def log_loss(name, value):
            self.log(
                name,
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=1,
            )

        log_loss("train/physics_loss", physics_loss)
        log_loss("train/bc_loss", bc_loss)
        log_loss("train/ic_loss", ic_loss)
        log_loss("train/inlet_loss", inlet_loss)
        log_loss("train/outlet_loss", outlet_loss)
        log_loss("train/noslip_loss", noslip_loss)

        # Log histograms of predictions and losses to W&B
        if self.wandb_logger is not None and batch_idx % 10 == 0:
            x, y, t = collocation_points
            with torch.no_grad():
                u_pred, v_pred, p_pred = self.model(x, y, t)

            self.wandb_logger.experiment.log(
                {
                    "distributions/u_pred": wandb.Histogram(
                        u_pred.detach().cpu().numpy()
                    ),
                    "distributions/v_pred": wandb.Histogram(
                        v_pred.detach().cpu().numpy()
                    ),
                    "distributions/p_pred": wandb.Histogram(
                        p_pred.detach().cpu().numpy()
                    ),
                    "distributions/u_mean": u_pred.mean().item(),
                    "distributions/u_std": u_pred.std().item(),
                    "distributions/v_mean": v_pred.mean().item(),
                    "distributions/v_std": v_pred.std().item(),
                    "distributions/p_mean": p_pred.mean().item(),
                    "distributions/p_std": p_pred.std().item(),
                    "distributions/u_abs_max": u_pred.abs().max().item(),
                    "distributions/v_abs_max": v_pred.abs().max().item(),
                    "distributions/p_abs_max": p_pred.abs().max().item(),
                },
                step=self.global_step,
            )

            xi, yi, ti = bc_points["inlet"]
            with torch.no_grad():
                ui_pred, _, _ = self.model(xi, yi, ti)
            ui_true = bc_points["inlet_u"]
            self.wandb_logger.experiment.log(
                {
                    "distributions/inlet_u_pred_mean": ui_pred.mean().item(),
                    "distributions/inlet_u_true_mean": ui_true.mean().item(),
                },
                step=self.global_step,
            )

        return loss

    def predict_step(self, batch, batch_idx):
        # Get inputs
        x, y, t, id = batch

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            u, v, p = self.model(x, y, t)

        return {
            "test_point_id": id,
            "u_pred": u,
            "v_pred": v,
            "p_pred": p,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config_training["training_hyperparameters"]["learning_rate"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config_training["training_hyperparameters"]["num_epochs"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    dataset = PyLDataModule(dataset_path="./dataset/")
    model = PyLModel()
    breakpoint()
