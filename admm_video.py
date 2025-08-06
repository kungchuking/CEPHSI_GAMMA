import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class ste_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()
    @staticmethod
    def backward(ctx, grad):
        return F.hardtanh(grad)

class STE(nn.Module):
    def __init__(self):
        super(STE, self).__init__()
    def forward(self, x):
        return ste_fn.apply(x)

class admm_video(nn.Module):
    def __init__(self,
        sigma_range=[0, 1e-9],
        frame_n=8,
        patch_size=[720, 1280],
        num_iters=16):

        super(admm_video, self).__init__()

        self.sigma_range = sigma_range
        self.frame_n = frame_n
        self.patch_size = patch_size
        
        # -- Learnable Parameters
        self.ce_weight = nn.Parameter(torch.Tensor(frame_n, *patch_size))
        self.rho = nn.Parameter(torch.Tensor(1))

        # -- Initialization
        nn.init.uniform(self.ce_weight, a=-1, b=1)
        nn.init.uniform(self.rho, a=0., b=0.1)

        self.ste = STE()
        self.sigmoid = nn.Sigmoid()

        self.num_iters = num_iters
        self.layers = []
        for i in range(num_iters):
            self.layers += [
                    nn.Sequential(
                        nn.Conv2d(in_channels=frame_n, out_channels=frame_n, kernel_size=3, stride=1, padding=5, padding_mode="reflect"),
                        nn.Sigmoid()
                    )
                ]

    def forward(self, frames):
        device = frames.device
        ce_code = self.ste(self.ce_weight)

        # -- define operator A and its adjoint based on ce_code
        Afun = lambda x, mask=ce_code: torch.stack([           # -- input x: (L, M, N)
            torch.sum(mask * x, axis=0),                       # -- channel 0
            torch.sum((1. / self.frame_n - mask) * x, axis=0)  # -- channel 1
        ], axis=0)                                             # -- shape: (2, M, N)
    
        Atfun = lambda y, mask=ce_code: (                      # -- input y: (2, M, N)
            mask * y[0][None, :, :] +                          # -- channel 0 backprojection
            (1. / self.frame_n - mask) * y[1][None, :, :]      # -- channel 1 backprojection
        )                                                      # -- output: (L, M, N)

        # -- ensure that the size of frames is correct.
        frames = frames[..., :self.patch_size[0], :self.patch_size[1]]

        ce_blur_img = Afun(frames)

        noise_level = np.random.uniform(*self.sigma_range)
        ce_blur_img_noisy = ce_blur_img + torch.tensor(noise_level, device=device) * torch.randn(ce_blur_img.shape, device=device)

        return self.admm(Afun=Afun, Atfun=Atfun, b=ce_blur_img_noisy)

    def cg_solve(self, A_new, b_new, x0=None, max_iter=200, tol=1e-12):
        # -- mind that A_new is positive definite here.
        # -- hence, b_new.size() == x.size()
        x = x0 if x0 is not None else torch.zeros_like(b_new)

        print ("[INFO] x.device: ", x.device)
        print ("[INFO] b_new.device: ", b_new.device)
        out = A_new(x)
        print ("[INFO] out.device: ", out.device)

        r = b_new - A_new(x)
        z = r
        p = z

        print ("[INFO] r.device: ", r.device)
        print ("[INFO] z.device: ", z.device)
        print ("[INFO] p.device: ", p.device)


        print ("[INFO] x.size(): ", x.size())

        delta_new = torch.sum(r * z)
        for i in range(max_iter):
            print ("[INFO] i: ", i)
            Ap = A_new(p)
            alpha = delta_new / torch.sum(p * Ap)
            x += alpha * p
            r -= alpha * Ap
            if torch.linalg.norm(r) < tol:
                break
            z = r
            delta_old = delta_new
            delta_new = torch.sum(r * z)
            beta = delta_new / delta_old
            p *= beta
            p += z
        return x

    def admm(self, Afun, Atfun, b):
        device = b.device
        
        print ("[INFO] b.device: ", b.device)

        x = torch.repeat_interleave(b, self.frame_n // 2, axis=0)
        z = torch.repeat_interleave(b, self.frame_n // 2, axis=0)
        u = torch.repeat_interleave(b, self.frame_n // 2, axis=0)
    
        for layer in self.layers:
            v = z - u
    
            # -- b ~ (2, M, N)
            # -- b_new ~ (L, M, N)
            b_new = Atfun(b) + self.rho * v
            print ("[INFO] b_new.device: ", b_new.device)
            print ("[INFO] self.rho.device: ", self.rho.device)
    
            A_new = lambda w : Atfun(Afun(w)) + self.rho * w
    
            x = self.cg_solve(A_new=A_new, b_new=b_new)
    
            v = x + u
            v_denoised = layer(v)
    
            u = u + x - z
    
        return x

