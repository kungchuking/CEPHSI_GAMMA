import torch
import torch.nn as nn

class denoise_block(nn.Module):
    def __init__(self):
        super(denoise_block, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.atrous_conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=1, dilation=1, padding=2, padding_mode="reflect")
        self.atrous_conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=1, dilation=2, padding=4, padding_mode="reflect")
        self.atrous_conv3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=1, dilation=4, padding=8, padding_mode="reflect")

        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.1))
        self.c = nn.Parameter(torch.tensor(0.1))

    def forward(self, frames):
        return torch.tanh(self.a * self.atrous_conv1(frames) + self.b * self.atrous_conv2(frames) + self.c * self.atrous_conv3(frames))

class video_denoise(nn.Module):
    def __init__(self, n_subframe=16):
        super(video_denoise, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_subframe = n_subframe

        self.denoise_block_1 = denoise_block()
        self.denoise_block_2 = denoise_block()

    def forward(self, x):
        # -- Step 1 Denoising
        y = torch.zeros_like(x).to(self.device)
        for i in range(x.size()[1]):
            triplet = torch.zeros((x.size()[0], 3, x.size()[2], x.size()[3])).to(self.device)
            if i == 0:
                triplet[:, 0, :, :] = x[:, i+1, :, :]
                triplet[:, 1, :, :] = x[:, i, :, :]
                triplet[:, 2, :, :] = x[:, i+1, :, :]
            elif i == x.size()[1] - 1:
                triplet[:, 0, :, :] = x[:, i-1, :, :]
                triplet[:, 1, :, :] = x[:, i, :, :]
                triplet[:, 2, :, :] = x[:, i-1, :, :]
            else:
                triplet[:, 0, :, :] = x[:, i-1, :, :]
                triplet[:, 1, :, :] = x[:, i, :, :]
                triplet[:, 2, :, :] = x[:, i+1, :, :]
            y[:, i, :, :] = self.denoise_block_1(triplet).squeeze(1)

        # -- Step 2 Denoising
        z = torch.zeros_like(y).to(self.device)
        for i in range(y.size()[1]):
            triplet = torch.zeros((y.size()[0], 3, y.size()[2], y.size()[3])).to(self.device)
            if i == 0:
                triplet[:, 0, :, :] = y[:, i+1, :, :]
                triplet[:, 1, :, :] = y[:, i, :, :]
                triplet[:, 2, :, :] = y[:, i+1, :, :]
            elif i == y.size()[1] - 1:
                triplet[:, 0, :, :] = y[:, i-1, :, :]
                triplet[:, 1, :, :] = y[:, i, :, :]
                triplet[:, 2, :, :] = y[:, i-1, :, :]
            else:
                triplet[:, 0, :, :] = y[:, i-1, :, :]
                triplet[:, 1, :, :] = y[:, i, :, :]
                triplet[:, 2, :, :] = y[:, i+1, :, :]
            z[:, i, :, :] = self.denoise_block_2(triplet).squeeze(1)

        return z

class video_denoise_3d(nn.Module):
    def __init__(self, n_subframe=16):
        super(video_denoise_3d, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_subframe = n_subframe

        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=(2, 2, 2), padding_mode="reflect"),
            nn.InstanceNorm3d(num_features=1, affine=True),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv3d(x)
