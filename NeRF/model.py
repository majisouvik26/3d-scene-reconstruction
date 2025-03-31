import torch
import torch.nn as nn

class NeRFModel(nn.Module):
    """
    Positional Encoding for 3D coordinates.
    MLP to obtain color and density values for given input coordinates and direction vectors.
    """
    def __init__(self, embedding_dim_pos=10, embedding_dim_dirxn=4, hidden_dim=128):
        super(NeRFModel, self).__init__()
        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_dirxn = embedding_dim_dirxn
        self.relu = nn.ReLU()
        print(f"Embedding Dim Pos: {self.embedding_dim_pos}, Type: {type(self.embedding_dim_pos)}")
        print(f"Embedding Dim Dirxn: {self.embedding_dim_dirxn}, Type: {type(self.embedding_dim_dirxn)}")

        self.block1 = nn.Sequential(
            nn.Linear(self.embedding_dim_pos*6 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # we need to add skip connections too
        self.block2 = nn.Sequential(
            nn.Linear(self.embedding_dim_pos*6 + hidden_dim + 3, hidden_dim),    # `+3` is important
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1),  # the extra 1 for the density
        )

        # now, we code up the tail of the network which firstly predicts the volume density sigma, then the RGB values
        self.tail1 = nn.Sequential(
            nn.Linear(self.embedding_dim_dirxn*6 + hidden_dim + 3, hidden_dim // 2),
            nn.ReLU(),
        )

        self.final_tail = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

    def positional_encoding(self, x, num_encoding_functions=6):
        """
        Positional Encoding for 3D coordinates - helps us to learn high-frequency signals in the scenes
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
        embedding_d = self.positional_encoding(d, self.embedding_dim_dirxn)

        h1 = self.block1(embedding_x)
        x2 = torch.cat([h1, embedding_x], dim=1)   # add skip connections
        h2 = self.block2(x2)

        h3, sigma = h2[:, :-1], self.relu(h2[:, -1])  # split the output into RGB and density

        h4 = self.tail1(torch.cat([embedding_d, h3], dim=1))
        color = self.final_tail(h4)

        return color, sigma
    

if __name__ == "__main__":
    model = NeRFModel()
    print(model)
    # random input
    x = torch.randn(1, 3)
    d = torch.randn(1, 3)
    color, sigma = model(x, d)
    print(f"Color: {color}")
    print(f"Sigma: {sigma}")