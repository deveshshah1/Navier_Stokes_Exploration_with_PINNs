import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(
        self,
        hidden_layers=6,
        hidden_width=64,
        domain=(
            0.0,
            2.2,
            0.0,
            0.41,
            0.0,
            10.0,
        ),
    ):
        super().__init__()
        layers = [nn.Linear(3, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_width, 3))

        self.net = nn.Sequential(*layers)
        self._init_weights()

        # Set normalization constants as buffers so they move with the model/device
        self.register_buffer("x_min", torch.tensor(domain[0]))
        self.register_buffer("x_max", torch.tensor(domain[1]))
        self.register_buffer("y_min", torch.tensor(domain[2]))
        self.register_buffer("y_max", torch.tensor(domain[3]))
        self.register_buffer("t_min", torch.tensor(domain[4]))
        self.register_buffer("t_max", torch.tensor(domain[5]))

    def _normalize(self, x, y, t):
        """Normalize inputs to [-1, 1] range based on domain limits."""
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
        input = torch.stack([x_norm, y_norm, t_norm], dim=-1)  # Shape: (batch_size, 3)
        output = self.net(input)  # Shape: (batch_size, 3)
        u, v, p = output[:, 0], output[:, 1], output[:, 2]  # Split into components
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
