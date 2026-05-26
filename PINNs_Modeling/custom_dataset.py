import torch
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt


# Global config
with open("./configs/config_training.yaml", "r") as file:
    config_training = yaml.safe_load(file)
    config_training = {k: v["value"] for k, v in config_training.items()}


class Cylinder2DDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_collocation_points=1000,
        num_bc_points=100,
        domain=(0.0, 2.2, 0.0, 0.41, 0.0, 8.0),
        H=0.41,
    ):
        """
        Initializes the sampling of collacation points
        """
        super().__init__()
        self.num_collocation_points = num_collocation_points
        self.num_bc_points = num_bc_points
        self.H = H
        self.x_min, self.x_max, self.y_min, self.y_max, self.t_min, self.t_max = domain

    def __len__(self):
        return 1

    def make_collocation_points(self):
        """
        Random collocation points inside the fluid domain.
        Domain defaults to the ST 2D-2 bounding box (excluding cylinder interior).
        domain = (x_min, x_max, y_min, y_max, t_min, t_max)
        """
        x = torch.empty(self.num_collocation_points).uniform_(self.x_min, self.x_max)
        y = torch.empty(self.num_collocation_points).uniform_(self.y_min, self.y_max)
        t = torch.empty(self.num_collocation_points).uniform_(self.t_min, self.t_max)

        # crude cylinder mask  (centre (0.2, 0.2), radius 0.05)
        cx, cy, r = 0.2, 0.2, 0.05
        inside = ((x - cx) ** 2 + (y - cy) ** 2) < r**2
        x, y, t = x[~inside], y[~inside], t[~inside]

        return (x, y, t)

    def make_boundary_points(self):
        # Inlet (Left edge); x = x_min, y in [0, H], t in [t_min, t_max]
        xi = torch.full((self.num_bc_points,), self.x_min)
        yi = torch.empty(self.num_bc_points).uniform_(0, self.H)
        ti = torch.empty(self.num_bc_points).uniform_(self.t_min, self.t_max)

        # Outlet (Right edge); x = x_max, y in [0, H], t in [t_min, t_max]
        xo = torch.full((self.num_bc_points,), self.x_max)
        yo = torch.empty(self.num_bc_points).uniform_(0, self.H)
        to = torch.empty(self.num_bc_points).uniform_(self.t_min, self.t_max)

        # No-slip walls; x in [x_min, x_max], y = {y_min, y_max}, t in [t_min, t_max]
        # Bottom wall
        x_bottom = torch.empty(self.num_bc_points // 3).uniform_(self.x_min, self.x_max)
        y_bottom = torch.full((self.num_bc_points // 3,), self.y_min)
        # Top wall
        x_top = torch.empty(self.num_bc_points // 3).uniform_(self.x_min, self.x_max)
        y_top = torch.full((self.num_bc_points // 3,), self.y_max)
        # Cylinder surface (parametric)
        theta = torch.empty(self.num_bc_points // 3).uniform_(0, 2 * torch.pi)
        cx, cy, r = 0.2, 0.2, 0.05
        x_cyl = cx + r * torch.cos(theta)
        y_cyl = cy + r * torch.sin(theta)
        # Combine
        xn = torch.cat([x_bottom, x_top, x_cyl])
        yn = torch.cat([y_bottom, y_top, y_cyl])
        tn = torch.empty(self.num_bc_points).uniform_(self.t_min, self.t_max)

        return {"inlet": (xi, yi, ti), "outlet": (xo, yo, to), "noslip": (xn, yn, tn)}

    def __getitem__(self, idx):
        collocation_points = self.make_collocation_points()
        bc_points = self.make_boundary_points()

        return {"collocation_points": collocation_points, "boundary_points": bc_points}


if __name__ == "__main__":
    data_set = Cylinder2DDataset()
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=9, shuffle=False)
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    batch = next(iter(data_loader))
    collocation_points_batch = batch["collocation_points"]
    for i in range(len(collocation_points_batch[0])):
        row, col = i // 3, i % 3
        ax[row, col].scatter(
            collocation_points_batch[0][i],
            collocation_points_batch[1][i],
            c=collocation_points_batch[2][i],
        )
        ax[row, col].set_title(f"Point {i}")
        ax[row, col].axis("off")
    fig.tight_layout()
    plt.show()

    idx = 0
    first_item = data_set[idx]
    print(first_item["collocation_points"])

    print()
    print("Dataset Loaded Successfully")
