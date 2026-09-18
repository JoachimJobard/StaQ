"""Continuous-action nets for ContraQ. The actor follows cleanrl's SAC implementation."""
import torch
from torch import nn

from src.utils.rl_tools import zero_linear

LOG_STD_MIN = -5
LOG_STD_MAX = 2


def _mlp(in_size, cfg, out_size, nl, zero_last=False):
    layers = []
    for _ in range(cfg.depth):
        layers += [nn.Linear(in_size, cfg.width), nl]
        in_size = cfg.width
    last = nn.Linear(cfg.width, out_size)
    layers.append(zero_linear(last) if zero_last else last)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Tanh-squashed Gaussian. Only an amortised sampler for exp(eta * S); it holds no value."""

    def __init__(self, cfg, state_dim, action_dim, action_low=-1.0, action_high=1.0, device=None):
        super().__init__()
        nl = cfg.nl if cfg.nl is not None else nn.ReLU(inplace=True)
        self.trunk = _mlp(state_dim, cfg, cfg.width, nl)  # trunk ends on a Linear+width
        self.mean_fc = nn.Linear(cfg.width, action_dim)
        self.logstd_fc = nn.Linear(cfg.width, action_dim)
        # Buffers, not constants: tanh gives [-1, 1] and envs are not all [-1, 1]
        # (Humanoid is [-0.4, 0.4]); getting this wrong silently rescales every action.
        self.register_buffer("action_scale", torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32))
        self.to(device)

    def forward(self, x):
        h = self.trunk(x)
        log_std = torch.tanh(self.logstd_fc(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return self.mean_fc(h), log_std

    def get_action(self, x):
        """Returns (action, log_prob, mean_action). rsample, so gradients flow through action."""
        mean, log_std = self(x)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        # Change-of-variables for the tanh squash. Without it log_prob is the density of the
        # PRE-squash variable, and every entropy term in critic and actor is wrong.
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean) * self.action_scale + self.action_bias


class Critic(nn.Module):
    """One ensemble member: the critic Q_k(s,a) and the student S(s,a) holding the running sum."""

    def __init__(self, cfg_q, cfg_student, state_dim, action_dim, kl_weight, entropy_weight, device=None):
        super().__init__()
        nl_q = cfg_q.nl if cfg_q.nl is not None else nn.ReLU(inplace=True)
        nl_s = cfg_student.nl if cfg_student.nl is not None else nn.ReLU(inplace=True)
        # zero_linear on both heads: Q_0 = 0 and S_{-1} = 0, matching StaQNet's convention.
        self.q = _mlp(state_dim + action_dim, cfg_q, 1, nl_q, zero_last=True)
        self.student = _mlp(state_dim + action_dim, cfg_student, 1, nl_s, zero_last=True)
        self.set_entropy_weight(kl_weight, entropy_weight)
        self.to(device)

    def set_entropy_weight(self, kl_weight, entropy_weight):
        self._kl_weight = kl_weight
        self.eta = 1.0 / (kl_weight + entropy_weight)
        self.decay = kl_weight / (kl_weight + entropy_weight)

    def forward(self, s, a):
        return self.q(torch.cat([s, a], dim=-1))

    def forward_student(self, s, a):
        return self.student(torch.cat([s, a], dim=-1))
