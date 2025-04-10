import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRFModel(nn.Module):
    """
    Positional Encoding for 3D coordinates.
    MLP to obtain color and density values for given input coordinates and direction vectors.
    """
    def __init__(self, embedding_dim_pos=10, embedding_dim_dirxn=4, hidden_dim_pos=384, hidden_dim_dir=128, D=8):
        super(NeRFModel, self).__init__()
        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_dir = embedding_dim_dirxn
        self.hidden_dim_dir = embedding_dim_dirxn
        self.hidden_dim_pos = embedding_dim_pos
        self.D = D

        self.relu = nn.ReLU()
        # print(f"Embedding Dim Pos: {self.embedding_dim_pos}, Type: {type(self.embedding_dim_pos)}")
        # print(f"Embedding Dim Dirxn: {self.embedding_dim_dir}, Type: {type(self.embedding_dim_dir)}")


        ## Figure 2 of new paper
        self.block1_pos = nn.Sequential(
            nn.Linear(self.embedding_dim_pos*6 + 3, hidden_dim_pos),
            nn.ReLU(),
            nn.Linear(hidden_dim_pos, hidden_dim_pos),
            nn.ReLU(),
            nn.Linear(hidden_dim_pos, hidden_dim_pos),
            nn.ReLU(),
            nn.Linear(hidden_dim_pos, hidden_dim_pos),
            nn.ReLU(),
            nn.Linear(hidden_dim_pos, hidden_dim_pos),
            nn.ReLU(),
            nn.Linear(hidden_dim_pos, hidden_dim_pos),
            nn.ReLU(),
            nn.Linear(hidden_dim_pos, hidden_dim_pos),
            nn.ReLU(),
            nn.Linear(hidden_dim_pos, 3 * D + 1),
        )

        # we need to add skip connections too
        self.block2_dir = nn.Sequential(
            nn.Linear(self.embedding_dim_dir * 6 + 3, hidden_dim_dir),    # `+3` is important
            nn.ReLU(),
            nn.Linear(hidden_dim_dir, hidden_dim_dir),
            nn.ReLU(),
            nn.Linear(hidden_dim_dir, hidden_dim_dir),
            nn.ReLU(),
            nn.Linear(hidden_dim_dir, hidden_dim_dir),
            nn.ReLU(),
            nn.Linear(hidden_dim_dir, D),  # the extra 1 for the density
        )


    def positional_encoding(self, x, num_encoding_functions=6):
        """
        Standard Positional Encoding for 3D coordinates - helps us to learn high-frequency signals in the scenes
        3D Input -> 63-dimension for the position, 24-dimension for the direction output
        """
        encoding = [x]
        for i in range(num_encoding_functions):
            encoding.append(torch.sin(2.0 ** i * x))
            encoding.append(torch.cos(2.0 ** i * x))
        return torch.cat(encoding, dim=1)

    def forward(self, x, d):
        """
        x: 3D coordinates
        d: direction vector
        For every position and direction, we return predicted color and density values
        """
        embedding_x = self.positional_encoding(x, self.embedding_dim_pos)
        embedding_d = self.positional_encoding(d, self.embedding_dim_dir)

        h1 = self.block1_pos(embedding_x)
        sigma = F.softplus(h1[:, 0][..., None])
        h2 = h1[:, 1:].reshape(-1, 3, self.D)
        final_pos = F.sigmoid(h2)   # [B, 3, D]

        h3 = self.block2_dir(embedding_d)
        betas = F.softmax(h3, -1)   # [B, 3]
        color = (betas.unsqueeze(1) * final_pos).sum(-1)  # weigted sum

        return color, sigma


################################################# Start of compatible borrowed code #################################################

class Cache(nn.Module):
    """
    To ensure high fps rendering of rays, I'm borrowing the implementation of class `Cache` from
    https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/tree/main/FastNeRF_High_Fidelity_Neural_Rendering_at_200FPS 
    """
    def __init__(self, model, scale, device, Np, Nd):
        super(Cache, self).__init__()

        with torch.no_grad():
            # Position
            x, y, z = torch.meshgrid([torch.linspace(-scale / 2, scale / 2, Np).to(device),
                                      torch.linspace(-scale / 2, scale / 2, Np).to(device),
                                      torch.linspace(-scale / 2, scale / 2, Np).to(device)])
            xyz = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1)
            sigma_uvw = model.block1_pos(model.positional_encoding(xyz, model.embedding_dim_pos))
            self.sigma_uvw = sigma_uvw.reshape((Np, Np, Np, -1))
            # Direction
            xd, yd = torch.meshgrid([torch.linspace(-scale / 2, scale / 2, Nd).to(device),
                                     torch.linspace(-scale / 2, scale / 2, Nd).to(device)])
            xyz_d = torch.cat((xd.reshape(-1, 1), yd.reshape(-1, 1),
                               torch.sqrt((1 - xd ** 2 - yd ** 2).clip(0, 1)).reshape(-1, 1)), dim=1)
            beta = model.block2_dir(model.positional_encoding(xyz_d, model.embedding_dim_dir))
            self.beta = beta.reshape((Nd, Nd, -1))

        self.scale = scale
        self.Np = Np
        self.Nd = Nd
        self.D = model.D

    def forward(self, x, d):
        color = torch.zeros_like(x)
        sigma = torch.zeros((x.shape[0], 1), device=x.device)

        mask = (x[:, 0].abs() < (self.scale / 2)) & (x[:, 1].abs() < (self.scale / 2)) & (x[:, 2].abs() < (self.scale / 2))
        # Position
        idx = (x[mask] / (self.scale / self.Np) + self.Np / 2).long().clip(0, self.Np - 1)
        sigma_uvw = self.sigma_uvw[idx[:, 0], idx[:, 1], idx[:, 2]]
        # Direction
        idx = (d[mask] * self.Nd).long().clip(0, self.Nd - 1)
        beta = torch.softmax(self.beta[idx[:, 0], idx[:, 1]], -1)

        sigma[mask] = torch.nn.functional.softplus(sigma_uvw[:, 0][..., None])  # [batch_size, 1]
        uvw = torch.sigmoid(sigma_uvw[:, 1:].reshape(-1, 3, self.D))  # [batch_size, 3, D]
        color[mask] = (beta.unsqueeze(1) * uvw).sum(-1)  # [batch_size, 3]
        return color, sigma

################################################# End of borrowed code #################################################

if __name__ == "__main__":
    model = NeRFModel()
    print(model)
    # random input
    x = torch.randn(1, 3)
    d = torch.randn(1, 3)
    color, sigma = model(x, d)
    print(f"Color: {color}")
    print(f"Sigma: {sigma}")