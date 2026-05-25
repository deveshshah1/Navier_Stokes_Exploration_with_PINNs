import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, hidden_layers=6, hidden_width=64):
        super().__init__()
        layers = [nn.Linear(3, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_width, 3))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        input = torch.stack([x, y, t], dim=-1)  # Shape: (batch_size, 3)
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
    print("u shape:", u.shape) # Should be (2,)
    print("v shape:", v.shape) # Should be (2,)
    print("p shape:", p.shape) # Should be (2,)
