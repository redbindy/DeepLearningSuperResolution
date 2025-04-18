import torch.nn as nn

class ESPCN(nn.Module):
    def __init__(self, scaleFactor):
        super(ESPCN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 5, padding = 2), 
            nn.Tanh(), 
            nn.Conv2d(64, 32, kernel_size = 3, padding = 1), 
            nn.Tanh(), 
            nn.Conv2d(32, scaleFactor ** 2, kernel_size = 3, padding = 1), 
            nn.PixelShuffle(scaleFactor)
        )

    def forward(self, x):
        return self.layers(x)