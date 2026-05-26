import torch
import yaml
import matplotlib.pyplot as plt


with open("./configs/config_training.yaml", "r") as f:
    config_training = yaml.safe_load(f)
    config_training = {k: v["value"] for k, v in config_training.items()}


# Schäfer-Turek geometry constants
CYL_CX, CYL_CY, CYL_R = 0.2, 0.2, 0.05  # cylinder center and radius
H = 0.41  # channel height
U_MEAN = 0.2  # mean inlet velocity


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
        domain: tuple = (
            0.0,
            2.2,
            0.0,
            0.41,
            0.0,
            10.0,
        ),  # (x_min, x_max, y_min, y_max, t_min, t_max)
        steps_per_epoch: int = 100,
    ):
        super().__init__()
        self.num_collocation_points = num_collocation_points
        self.num_bc_points = num_bc_points
        self.num_noslip_points = num_noslip_points
        self.num_ic_points = num_ic_points
        self.steps_per_epoch = steps_per_epoch

        self.x_min, self.x_max, self.y_min, self.y_max, self.t_min, self.t_max = domain

    def __len__(self):
        return self.steps_per_epoch
    
    def _remove_cylinder_interior(self, x, y, *rest):
        """Drop any points that fall inside the cylinder."""
        inside = ((x - CYL_CX) ** 2 + (y - CYL_CY) ** 2) < CYL_R**2
        mask = ~inside
        return (x[mask], y[mask]) + tuple(t[mask] for t in rest)

    def inlet_u(self, y, H=H, U_mean=U_MEAN):
        """Schäfer-Turek parabolic inlet: u = 6 U_mean y(H-y) / H²"""
        return 6.0 * U_mean * y * (H - y) / H**2

    def make_collocation_points(self):
        """Uniform random points over the fluid domain"""
        x = torch.empty(self.num_collocation_points).uniform_(self.x_min, self.x_max)
        y = torch.empty(self.num_collocation_points).uniform_(self.y_min, self.y_max)
        t = torch.empty(self.num_collocation_points).uniform_(self.t_min, self.t_max)
        x, y, t = self._remove_cylinder_interior(x, y, t)
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
        theta = torch.empty(n_s).uniform_(0.0, 2.0 * torch.pi)
        x_cyl = CYL_CX + CYL_R * torch.cos(theta)
        y_cyl = CYL_CY + CYL_R * torch.sin(theta)

        # combine no-slip
        xn = torch.cat([x_bot, x_top, x_cyl])
        yn = torch.cat([y_bot, y_top, y_cyl])
        tn = rand_t(len(xn)) 

        return {
            "inlet": (xi, yi, ti),
            "outlet": (xo, yo, to),
            "noslip": (xn, yn, tn),
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

        ic_u = torch.zeros_like(x)
        ic_v = torch.zeros_like(x)

        return x, y, t, ic_u, ic_v

    def __getitem__(self, idx):
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

    x_ic, y_ic, t_ic, u_ic, v_ic = batch["ic"]
    print(
        f"IC          : {x_ic.shape} points  (u={u_ic.unique().tolist()}, v={v_ic.unique().tolist()})"
    )

    # plot collocation + BC points
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    sc = ax.scatter(xc, yc, c=tc, s=3, cmap="viridis", label="collocation")
    plt.colorbar(sc, ax=ax, label="t")
    cyl = plt.Circle((CYL_CX, CYL_CY), CYL_R, color="gray", zorder=5)
    ax.add_patch(cyl)
    ax.set_title("Collocation points (coloured by t)")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = axes[1]
    colours = {"inlet": "tab:blue", "outlet": "tab:orange", "noslip": "tab:red"}
    for name, (x, y, _) in batch["boundary"].items():
        ax.scatter(x, y, s=6, label=name, color=colours[name], alpha=0.7)
    ax.scatter(x_ic, y_ic, s=3, label="IC (t=0)", color="tab:green", alpha=0.5)
    cyl2 = plt.Circle((CYL_CX, CYL_CY), CYL_R, color="gray", zorder=5)
    ax.add_patch(cyl2)
    ax.set_title("Boundary & IC points")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(markerscale=2, fontsize=8)

    fig.tight_layout()
    plt.show()
    print("\nDataset loaded successfully.")
