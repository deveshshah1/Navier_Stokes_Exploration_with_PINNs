import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from custom_dataset import Cylinder2DDataset
from model import BaselineModel


# Global config
with open("./configs/config_training.yaml", "r") as file:
    config_training = yaml.safe_load(file)
    config_training = {k: v["value"] for k, v in config_training.items()}


class PyLDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset_configs = config_training["dataset_configs"]
        bounds = self.dataset_configs.pop("domain_bounds")
        self.dataset_configs["domain"] = (
            bounds["x_min"],
            bounds["x_max"],
            bounds["y_min"],
            bounds["y_max"],
            bounds["t_min"],
            bounds["t_max"],
        )

    def setup(self, stage=None):
        self.train_set = Cylinder2DDataset(**self.dataset_configs)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=config_training["training_hyperparameters"]["batch_size"],
            pin_memory=True,
            drop_last=True,
            num_workers=4,
            persistent_workers=True,
        )

class PyLModel(pl.LightningModule):
    def __init__(self, wandb_logger=None):
        super().__init__()
        self.save_hyperparameters()
        self.wandb_logger = wandb_logger

        self.model = BaselineModel(**config_training["model_architecture_hyperparameters"])
        self.lambda_physics = config_training["loss_weights"]["lambda_physics"]
        self.lambda_bc = config_training["loss_weights"]["lambda_bc"]
        self.lambda_ic = config_training["loss_weights"]["lambda_ic"]
        self.Re = config_training["dataset_configs"]["reynolds_number"]

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

        return (inlet_loss + outlet_loss + noslip_loss)
    
    def ic_loss(self, ic_points):
        # initial condition: ui = vi = 0 at t=0
        x, y, t = ic_points
        ui_pred, vi_pred, _ = self.model(x, y, t)
        ic_loss = torch.mean(ui_pred**2) + torch.mean(vi_pred**2)
        return ic_loss
    
    def physics_loss(self, collocation_points):
        # physics loss: Navier-Stokes residuals at collocation points
        x, y, t = collocation_points
        u, v, p = self.model(x, y, t)

        # first order derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        # second order derivatives
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        # Navier-Stokes residuals
        continuity = u_x + v_y
        momentum_u = u_t + (u * u_x) + (v * u_y) + p_x - (1.0 / self.Re) * (u_xx + u_yy)
        momentum_v = v_t + (u * v_x) + (v * v_y) + p_y - (1.0 / self.Re) * (v_xx + v_yy)

        physics_loss = torch.mean(continuity**2) + torch.mean(momentum_u**2) + torch.mean(momentum_v**2)
        return physics_loss

    def training_step(self, batch, batch_idx):
        collocation_points = batch["collocation"]
        bc_points = batch["boundary"]
        ic_points = batch["ic"]

        bc_loss = self.bc_loss(bc_points)
        ic_loss = self.ic_loss(ic_points)
        physics_loss = self.physics_loss(collocation_points)

        loss = self.lambda_physics * physics_loss + self.lambda_bc * bc_loss + self.lambda_ic * ic_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("physics_loss", physics_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("bc_loss", bc_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("ic_loss", ic_loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss
    
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
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
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
