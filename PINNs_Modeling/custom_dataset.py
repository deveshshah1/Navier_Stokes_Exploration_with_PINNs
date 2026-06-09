import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt


with open("./configs/config_training.yaml", "r") as f:
    config_training = yaml.safe_load(f)
    config_training = {k: v["value"] for k, v in config_training.items()}


class Cylinder2DDataset(torch.utils.data.Dataset):
    """
    Physics-only dataset for the 2-D cylinder PINN.
    """

    def __init__(
        self,
        num_collocation_points: int = 2000,
        num_bc_points: int = 200,  # per boundary (inlet, outlet)
        num_noslip_points: int = 300,  # per no-slip surface (bottom, top, cyl)
        num_ic_points: int = 500,
        domain_bounds: dict = {
            "x_min": 0.0,
            "x_max": 2.2,
            "y_min": 0.0,
            "y_max": 0.41,
            "t_min": 0.0,
            "t_max": 10.0,
        },
        U_mean: float = 0.2,  # mean inlet velocity (for ICs and inlet BC)
        cylinder_geometry: dict = {"cx": 0.2, "cy": 0.2, "r": 0.05},
        num_near_cylinder_points: int = 1000,
        near_cylinder_radius_factor: float = 4.0,
        steps_per_epoch: int = 100,
        ground_truth_dataset_path: str = None,
        use_ground_truth_dataset: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_collocation_points = num_collocation_points
        self.num_bc_points = num_bc_points
        self.num_noslip_points = num_noslip_points
        self.num_ic_points = num_ic_points
        self.steps_per_epoch = steps_per_epoch

        self.x_min = domain_bounds["x_min"]
        self.x_max = domain_bounds["x_max"]
        self.y_min = domain_bounds["y_min"]
        self.y_max = domain_bounds["y_max"]
        self.t_min = domain_bounds["t_min"]
        self.t_max = domain_bounds["t_max"]
        self.U_mean = U_mean

        self.cyl_cx = cylinder_geometry["cx"]
        self.cyl_cy = cylinder_geometry["cy"]
        self.cyl_r = cylinder_geometry["r"]
        self.H = self.y_max - self.y_min
        self.num_near_cylinder_points = num_near_cylinder_points
        self.near_cylinder_radius_factor = near_cylinder_radius_factor

        self.use_ground_truth_dataset = use_ground_truth_dataset
        if self.use_ground_truth_dataset:
            self.dataset_df = pd.read_parquet(ground_truth_dataset_path)

    def __len__(self):
        if self.use_ground_truth_dataset:
            return len(self.dataset_df)
        return self.steps_per_epoch

    def _remove_cylinder_interior(self, x, y, *rest):
        """Drop any points that fall inside the cylinder."""
        inside = ((x - self.cyl_cx) ** 2 + (y - self.cyl_cy) ** 2) < self.cyl_r**2
        mask = ~inside
        return (x[mask], y[mask]) + tuple(t[mask] for t in rest)

    def inlet_u(self, y):
        """Schäfer-Turek parabolic inlet: u = 6 U_mean y(H-y) / H²"""
        return 6.0 * self.U_mean * y * (self.H - y) / self.H**2

    def make_collocation_points(self):
        """
        Uniform random points over the fluid domain, plus concentrated points
        near the cylinder where gradients are steepest.
        """
        # Background uniform points
        x = torch.empty(self.num_collocation_points).uniform_(self.x_min, self.x_max)
        y = torch.empty(self.num_collocation_points).uniform_(self.y_min, self.y_max)
        t = torch.empty(self.num_collocation_points).uniform_(self.t_min, self.t_max)
        x, y, t = self._remove_cylinder_interior(x, y, t)

        if self.num_near_cylinder_points > 0:
            r_near = self.near_cylinder_radius_factor * self.cyl_r
            # Rejection-sample from bounding box clipped to domain, keep annular region
            n_draw = self.num_near_cylinder_points * 3
            xn = torch.empty(n_draw).uniform_(
                max(self.x_min, self.cyl_cx - r_near),
                min(self.x_max, self.cyl_cx + r_near),
            )
            yn = torch.empty(n_draw).uniform_(
                max(self.y_min, self.cyl_cy - r_near),
                min(self.y_max, self.cyl_cy + r_near),
            )
            tn = torch.empty(n_draw).uniform_(self.t_min, self.t_max)
            dist2 = (xn - self.cyl_cx) ** 2 + (yn - self.cyl_cy) ** 2
            # Keep points in the annulus (outside cylinder, inside r_near)
            mask = (dist2 >= self.cyl_r ** 2) & (dist2 <= r_near ** 2)
            xn, yn, tn = xn[mask][: self.num_near_cylinder_points], \
                         yn[mask][: self.num_near_cylinder_points], \
                         tn[mask][: self.num_near_cylinder_points]
            x = torch.cat([x, xn])
            y = torch.cat([y, yn])
            t = torch.cat([t, tn])

        return x, y, t

    def make_boundary_points(self):
        """
        Sample boundary points for inlet, outlet, and all no-slip surfaces.
        No-slip covers three surfaces (bottom wall, top wall, cylinder).
        """
        n = self.num_bc_points
        n_s = self.num_noslip_points

        def rand_t(size):
            return torch.empty(size).uniform_(self.t_min, self.t_max)

        # Inlet: x = x_min, y ∈ [0, H]
        xi = torch.full((n,), self.x_min)
        yi = torch.empty(n).uniform_(self.y_min, self.y_max)
        ti = rand_t(n)
        u_true = self.inlet_u(yi)

        # Outlet: x = x_max, y ∈ [0, H]
        xo = torch.full((n,), self.x_max)
        yo = torch.empty(n).uniform_(self.y_min, self.y_max)
        to = rand_t(n)

        # No-slip: bottom wall
        x_bot = torch.empty(n_s).uniform_(self.x_min, self.x_max)
        y_bot = torch.full((n_s,), self.y_min)

        # No-slip: top wall
        x_top = torch.empty(n_s).uniform_(self.x_min, self.x_max)
        y_top = torch.full((n_s,), self.y_max)

        # No-slip: cylinder surface (parametric)
        n_s_cyl = n_s * 1
        theta = torch.empty(n_s_cyl).uniform_(0.0, 2.0 * torch.pi)
        x_cyl = self.cyl_cx + self.cyl_r * torch.cos(theta)
        y_cyl = self.cyl_cy + self.cyl_r * torch.sin(theta)

        # combine no-slip
        xn = torch.cat([x_bot, x_top, x_cyl])
        yn = torch.cat([y_bot, y_top, y_cyl])
        tn = rand_t(len(xn))

        return {
            "inlet": (xi, yi, ti),
            "outlet": (xo, yo, to),
            "noslip": (xn, yn, tn),
            "inlet_u": u_true,  # for BC loss calculation
        }

    def make_ic_points(self):
        """
        Initial condition points at t = t_min over the fluid domain.
        u=0, v=0 everywhere (fluid starts at rest).
        """
        x = torch.empty(self.num_ic_points).uniform_(self.x_min, self.x_max)
        y = torch.empty(self.num_ic_points).uniform_(self.y_min, self.y_max)
        t = torch.full((self.num_ic_points,), self.t_min)
        x, y, t = self._remove_cylinder_interior(x, y, t)

        return x, y, t

    def get_ground_truth_points(self, idx):
        test_point = self.dataset_df.loc[idx]
        test_point_id = torch.tensor(test_point["test_point_id"], dtype=torch.int64)
        x = torch.tensor(test_point["x"], dtype=torch.float32)
        y = torch.tensor(test_point["y"], dtype=torch.float32)
        t = torch.tensor(test_point["t"], dtype=torch.float32)
        return x, y, t, test_point_id

    def __getitem__(self, idx):
        if self.use_ground_truth_dataset:
            return self.get_ground_truth_points(idx)
        else:
            return {
                "collocation": self.make_collocation_points(),
                "boundary": self.make_boundary_points(),
                "ic": self.make_ic_points(),
            }


if __name__ == "__main__":
    dataset = Cylinder2DDataset(
        num_collocation_points=2000,
        num_bc_points=200,
        num_noslip_points=300,
        num_ic_points=500,
        steps_per_epoch=4,
        cylinder_geometry=config_training["dataset_configs"]["cylinder_geometry"],
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0],
    )
    batch = next(iter(loader))

    # print shapes
    xc, yc, tc = batch["collocation"]
    print(f"Collocation : {xc.shape} points (after cylinder mask)")

    for name, pts in batch["boundary"].items():
        print(f"BC {name:<8}: {pts[0].shape} points")

    x_ic, y_ic, t_ic = batch["ic"]
    print(f"IC          : {x_ic.shape} points  (t={t_ic.unique().tolist()})")

    # plot collocation + BC points
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    sc = ax.scatter(xc, yc, c=tc, s=3, cmap="viridis", label="collocation")
    plt.colorbar(sc, ax=ax, label="t")
    cyl = plt.Circle(
        (dataset.cyl_cx, dataset.cyl_cy), dataset.cyl_r, color="gray", zorder=5
    )
    ax.add_patch(cyl)
    ax.set_title("Collocation points (coloured by t)")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = axes[1]
    colours = {"inlet": "tab:blue", "outlet": "tab:orange", "noslip": "tab:red"}
    for name, (x, y, _) in [
        ("inlet", batch["boundary"]["inlet"]),
        ("outlet", batch["boundary"]["outlet"]),
        ("noslip", batch["boundary"]["noslip"]),
    ]:
        ax.scatter(x, y, s=6, label=name, color=colours[name], alpha=0.7)
    ax.scatter(x_ic, y_ic, s=3, label="IC (t=0)", color="tab:green", alpha=0.5)
    cyl2 = plt.Circle(
        (dataset.cyl_cx, dataset.cyl_cy), dataset.cyl_r, color="gray", zorder=5
    )
    ax.add_patch(cyl2)
    ax.set_title("Boundary & IC points")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(markerscale=2, fontsize=8)

    fig.tight_layout()
    plt.show()
    print("\nDataset loaded successfully.")
