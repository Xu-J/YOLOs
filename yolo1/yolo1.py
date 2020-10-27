import torch.nn as nn
import numpy as np
import torch

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
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 1, 1, 0)  ,# (b_s, 256, 28, 28)	rd=3
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(256, 512, 3, 1, 1) , # (b_s, 512, 28, 28)	rd=3
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(512, 256, 1, 1, 0) , # (b_s, 256, 28, 28)	rd=4
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(256, 512, 3, 1, 1)  ,# (b_s, 512, 28, 28)	rd=4
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(512, 512, 1, 1, 0) , # (b_s, 512, 28, 28)
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(512, 1024, 3, 1, 1) , # (b_s, 1024, 28, 28)
            nn.LeakyReLU(self.relu_alpha),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(1024, 512, 1, 1, 0) , # (b_s, 512, 14, 14)	rd=1
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(512, 1024, 3, 1, 1) , # (b_s, 1024, 14, 14)	rd=1
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(1024, 512, 1, 1, 0) , # (b_s, 512, 14, 14)	rd=2
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(512, 1024, 3, 1, 1)  ,# (b_s, 1024, 14, 14)	rd=2
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(1024, 1024, 1, 1, 0) , # (b_s, 1024, 14, 14)
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(1024, 1024, 3, 2, 1),
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(1024, 1024, 3, 1, 1),  # (b_s, 1024, 7, 7)
            nn.LeakyReLU(self.relu_alpha),
            nn.Conv2d(1024, 1024, 3, 1, 1) , # (b_s, 1024, 7, 7)
            nn.LeakyReLU(self.relu_alpha),
        )

        self.layer2 = nn.Sequential(

            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(self.relu_alpha),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 30 * 7 * 7),

        )
    def forward(self,input):
        x = self.layer(input)
        x = x.view(x.size(0),-1)

        out = self.layer2(x)
        out = out.view(-1,7,7,30)
        return out



yolo = Yolo1net()
input = torch.rand(16,3,448,448)
out = yolo(input)
print(out.shape)