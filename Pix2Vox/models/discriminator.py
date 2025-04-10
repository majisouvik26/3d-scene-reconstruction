import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, cfg=None):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # Input: (1, 32, 32, 32)
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 16, 16, 16)
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 8, 8, 8)
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 4, 4, 4)
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),  # -> (256, 2, 2, 2)
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),            # -> (256 * 2 * 2 * 2)
            nn.Linear(256 * 2 * 2 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        print("before starting the discriminator its shape is: ", x.shape)
        x = self.main(x)
        x = self.classifier(x)
        return x


dis=Discriminator()

# Simulate input: shape [batch_size,n_views,img_c, img_h, img_w]

batch_size = 64
img_x, img_y, img_z = 32,32,32
dummy_input = torch.randn(batch_size,img_x,img_y, img_z)


output = dis(dummy_input)
print("output shape:", output.shape) #expected(64,1)
