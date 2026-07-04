import warnings
import torch
import torch.nn as nn
import math


class Sine(nn.Module):
    def __init__(self, omega: float = 1.0):
        super().__init__()
        self.omega = omega

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)


class FourierFeatureEncoder(nn.Module):
    """
    Maps input coordinates through random Fourier features to combat spectral bias.
    B is sampled from N(0, sigma^2) and fixed (not learned).
    Output: [sin(2π B x), cos(2π B x)], shape (..., 2 * num_features).
    """

    def __init__(self, input_dim: int, num_features: int, sigma: float):
        super().__init__()
        B = torch.randn(num_features, input_dim) * sigma
        self.register_buffer("B", B)

    @property
    def output_dim(self) -> int:
        return 2 * self.B.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class BaselineModel(nn.Module):
    def __init__(
        self,
        hidden_layers=6,
        hidden_width=64,
        domain_bounds={
            "x_min": 0.0,
            "x_max": 2.2,
            "y_min": 0.0,
            "y_max": 0.41,
            "t_min": 0.0,
            "t_max": 10.0,
        },
        use_fourier_features=False,
        num_fourier_features=64,
        sigma_spatial=10.0,
        sigma_time=5.0,
        use_hard_bc_cylinder=False,
        cylinder_geometry={"cx": 0.2, "cy": 0.2, "r": 0.05},
        use_siren=False,
        siren_omega_0=30.0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            warnings.warn(
                f"BaselineModel received unexpected kwargs (possible config typo?): {list(kwargs.keys())}"
            )

        self.use_fourier_features = use_fourier_features
        self.use_hard_bc_cylinder = use_hard_bc_cylinder
        self.use_siren = use_siren
        self.siren_omega_0 = siren_omega_0
        if use_hard_bc_cylinder:
            self.register_buffer("cyl_cx", torch.tensor(cylinder_geometry["cx"], dtype=torch.float32))
            self.register_buffer("cyl_cy", torch.tensor(cylinder_geometry["cy"], dtype=torch.float32))
            self.register_buffer("cyl_r",  torch.tensor(cylinder_geometry["r"],  dtype=torch.float32))

        if use_fourier_features:
            self.xy_encoder = FourierFeatureEncoder(
                2, num_fourier_features, sigma_spatial
            )
            self.t_encoder = FourierFeatureEncoder(1, num_fourier_features, sigma_time)
            input_dim = self.xy_encoder.output_dim + self.t_encoder.output_dim
        else:
            input_dim = 3

        if self.use_siren:
            activation_first = Sine(omega=siren_omega_0)
            activation_hidden = Sine(omega=1.0)
        else:
            activation_first = nn.Tanh()
            activation_hidden = nn.Tanh()
        
        layers = [nn.Linear(input_dim, hidden_width), activation_first]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(activation_hidden)
        layers.append(nn.Linear(hidden_width, 3))

        self.net = nn.Sequential(*layers)
        self._init_weights()

        # Set normalization constants as buffers so they move with the model/device
        self.register_buffer("x_min", torch.tensor(domain_bounds["x_min"]))
        self.register_buffer("x_max", torch.tensor(domain_bounds["x_max"]))
        self.register_buffer("y_min", torch.tensor(domain_bounds["y_min"]))
        self.register_buffer("y_max", torch.tensor(domain_bounds["y_max"]))
        self.register_buffer("t_min", torch.tensor(domain_bounds["t_min"]))
        self.register_buffer("t_max", torch.tensor(domain_bounds["t_max"]))

    def _cylinder_wall_distance(self, x, y):
        """
        Smooth mask in [0, 1]: 0 on the cylinder surface, →1 far away.
        Multiplying u and v by this enforces no-slip structurally.
        Uses physical (unnormalized) coordinates since cylinder geometry is in physical space.
        """
        r_dist = torch.sqrt((x - self.cyl_cx) ** 2 + (y - self.cyl_cy) ** 2)
        # alpha=10 confines suppression to ~0.2r from the surface, leaving the near-wake unaffected
        return torch.tanh(10.0 * (r_dist - self.cyl_r) / self.cyl_r)

    def _normalize(self, x, y, t):
        x_norm = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_norm = 2.0 * (y - self.y_min) / (self.y_max - self.y_min) - 1.0
        t_norm = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        return x_norm, y_norm, t_norm

    def _init_weights(self):
        if self.use_siren:
            is_first = True
            for m in self.net:
                if isinstance(m, nn.Linear):
                    fan_in = m.weight.shape[1]
                    if is_first:
                        # First layer: uniform(-1/fan_in, 1/fan_in), scaled by omega_0
                        bound = 1.0 / fan_in
                        nn.init.uniform_(m.weight, -bound, bound)
                        is_first = False
                    else:
                        # Hidden layers: uniform(-sqrt(6/fan_in), sqrt(6/fan_in))
                        bound = math.sqrt(6.0 / fan_in)
                        nn.init.uniform_(m.weight, -bound, bound)
                    nn.init.zeros_(m.bias)
        else:
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        x_norm, y_norm, t_norm = self._normalize(x, y, t)

        if self.use_fourier_features:
            xy = torch.stack([x_norm, y_norm], dim=-1)
            t_in = t_norm.unsqueeze(-1)
            encoded = torch.cat([self.xy_encoder(xy), self.t_encoder(t_in)], dim=-1)
        else:
            encoded = torch.stack([x_norm, y_norm, t_norm], dim=-1)

        output = self.net(encoded)
        u_raw, v_raw, p = output[:, 0], output[:, 1], output[:, 2]

        if self.use_hard_bc_cylinder:
            d = self._cylinder_wall_distance(x, y)
            u, v = d * u_raw, d * v_raw
        else:
            u, v = u_raw, v_raw

        return u, v, p


if __name__ == "__main__":
    model = BaselineModel()
    print(model)

    x = torch.randn(2)  # Example input tensor for x
    y = torch.randn(2)  # Example input tensor for y
    t = torch.randn(2)  # Example input tensor for t
    u, v, p = model(x, y, t)
    print("u shape:", u.shape)  # Should be (2,)
    print("v shape:", v.shape)  # Should be (2,)
    print("p shape:", p.shape)  # Should be (2,)
