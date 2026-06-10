import warnings
import torch
import torch.nn as nn
import math


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
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            warnings.warn(
                f"BaselineModel received unexpected kwargs (possible config typo?): {list(kwargs.keys())}"
            )

        self.use_fourier_features = use_fourier_features

        if use_fourier_features:
            self.xy_encoder = FourierFeatureEncoder(
                2, num_fourier_features, sigma_spatial
            )
            self.t_encoder = FourierFeatureEncoder(1, num_fourier_features, sigma_time)
            input_dim = self.xy_encoder.output_dim + self.t_encoder.output_dim
        else:
            input_dim = 3

        layers = [nn.Linear(input_dim, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.Tanh())
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

    def _normalize(self, x, y, t):
        x_norm = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_norm = 2.0 * (y - self.y_min) / (self.y_max - self.y_min) - 1.0
        t_norm = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        return x_norm, y_norm, t_norm

    def _init_weights(self):
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
        u, v, p = output[:, 0], output[:, 1], output[:, 2]
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
