from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn


class StackedNN:
    """Stacked NN layers for an efficient batched forward pass."""

    def __init__(self, ann: Iterable[nn.Module], max_size: int, strides=None):
        # ann should be iterable List[torch.nn.Module]
        self.sweights = []  # stacked weights, list of len L (layers) with elements of dims: (Niter, in_dim, out_dim)
        self.sbiases = []
        self.non_lin = []
        self.layer_types = []  # Track the layer type ('conv' or 'mlp')
        self.ensemble_size = 1
        self.strides = strides
        self.max_size = max_size

        for f in ann:
            if isinstance(f, nn.Linear):
                self.sweights.append(f.weight.t().detach().clone()[None, ...])
                self.sbiases.append(f.bias.detach().clone()[None, None, ...])
                self.layer_types.append('mlp')
            elif isinstance(f, nn.Conv2d):
                self.sweights.append(f.weight.detach().clone()[None, ...])
                if f.bias is not None:
                    self.sbiases.append(f.bias.detach().clone()[None, ...])
                self.layer_types.append('conv')
            elif isinstance(f, nn.Flatten):
                continue
            else:
                self.non_lin.append(f)
        if len(self.non_lin) < len(self.sweights):
            self.non_lin += [nn.Identity()] * (len(self.sweights) - len(self.non_lin))

    def __call__(self, x):
        for layer_idx, (w, b, nl, layer_type) in enumerate(zip(self.sweights, self.sbiases, self.non_lin, self.layer_types)):

            if layer_type == 'conv':
                assert self.strides is not None, "Strides must be provided for convolutional layers."
                N, out_channels, in_channels, kernel_h, kernel_w = w.shape
                if N != self.ensemble_size:
                    self.ensemble_size = N

                # Broadcast on first layer
                if layer_idx == 0:
                    x = x.repeat(1, N, 1, 1)

                x = F.conv2d(
                    x,
                    w.view(N * out_channels, in_channels, kernel_h, kernel_w),
                    b.view(-1),
                    stride=self.strides[layer_idx],
                    groups=N
                )
                x = nl(x)

            elif layer_type == 'mlp':
                batch_size = x.size(0)
                if len(x.shape) > 3:  # Flatten if needed
                    N = self.ensemble_size
                    x = x.view(batch_size, N, -1).transpose(0, 1)

                matmul_result = torch.matmul(x, w)  # [Batch, M, Features] x [M, Features, Output_Size]
                x = nl(matmul_result + b)
        return x

    def push(self, ann):
        idx = 0
        for f in ann:
            if isinstance(f, nn.Linear):
                w = self.sweights[idx]
                b = self.sbiases[idx]
                if w.shape[0] > self.max_size:
                    w = w[1:]
                    b = b[1:]
                w_new = f.weight.t().detach().clone()[None, ...]
                b_new = f.bias.detach().clone()[None, None, ...]
                self.sweights[idx] = torch.cat((w, w_new), 0) if w.shape[0] > 0 else w_new
                self.sbiases[idx] = torch.cat((b, b_new), 0) if b.shape[0] > 0 else b_new
                idx += 1
            elif isinstance(f, nn.Conv2d):
                w = self.sweights[idx]
                b = self.sbiases[idx]
                if w.shape[0] > self.max_size:
                    w = w[1:]
                    b = b[1:]
                w_new = f.weight.detach().clone()[None, ...]
                self.sweights[idx] = torch.cat((w, w_new), 0) if w.shape[0] > 0 else w_new
                if f.bias is not None:
                    b_new = f.bias.detach().clone()[None, ...]
                    self.sbiases[idx] = torch.cat((b, b_new), 0) if b.shape[0] > 0 else b_new
                idx += 1

    def decay(self, c):
        for w, b in zip(self.sweights, self.sbiases):
            w.data *= c
            b.data *= c