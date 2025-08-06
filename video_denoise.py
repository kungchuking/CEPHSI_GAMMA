import torch
import torch.nn as nn

class video_denoise(nn.Module):
    def __init__(self, frame_n, num_iter=16):
        super(video_denoise, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_iter = num_iter

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=frame_n, out_channels=frame_n, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
                nn.GroupNorm(num_groups=1, num_channels=frame_n),
                nn.Tanh()
            ) for _ in range(num_iter)
        ])

    def forward(self, frames, idx=None):
        x = frames
        ret = []

        if idx == None:
            for i in range(self.num_iter):
                x = self.layers[i](x)
                ret += [x]
        else:
            return self.layers[idx](x)

        return ret
