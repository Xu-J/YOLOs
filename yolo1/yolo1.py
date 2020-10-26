import torch.nn as nn
import numpy as np

class Yolo1net(nn.Module):
    def __init__(self):
        super(Yolo1net, self).__init__( )

        self.imgsize = 448
        self.relu_alpha = 0.1
        self.layer = nn.Sequential(
            nn.Conv2d(3,64,7,2,3 ),
            nn.LeakyReLU(self.relu_alpha),
            nn.MaxPool2d(2,2)  ,

            nn.Conv2d(64, 192, 3, 1, 1),
            nn.LeakyReLU(self.relu_alpha),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 128, 1, 1, 0),
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(self.relu_alpha),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 256, 1, 1, 0),
        nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(512, 256, 1, 1, 0),
        nn.LeakyReLU(self.relu_alpha),


        )