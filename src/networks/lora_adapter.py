import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, linear, r_max, alpha, warm_start, std=.2):
        super().__init__()
        self.linear = linear
        self.r_max = r_max
        self.alpha = alpha
        self.warm_start = warm_start
        self.std = std
        self.r = r_max  # Initialize r to r_max

        # Initialize LoRA parameters
        self.lora_A = nn.Parameter(torch.empty((linear.in_features, r_max)))
        self.lora_A = torch.nn.init.normal_(self.lora_A, mean=0, std=self.std)
        self.lora_B = nn.Parameter(torch.zeros((r_max, linear.out_features)))

    def forward(self, x):
        # Compute the LoRA adjustment
        if self.r == 0:
            return self.linear(x)
        else:
            return self.linear(x) + (x @ self.lora_A[:, :self.r]) @ self.lora_B[:self.r, :] * self.scale

    def reset_lora_parameters(self):
        # Reinitialize LoRA parameters
        self.lora_A = torch.nn.init.normal_(self.lora_A, mean=0, std=self.std)
        self.lora_B = torch.nn.init.zeros_(self.lora_B)

    def merged(self):
        # Merge LoRA parameters into the linear layer's weights
        out = nn.Linear(self.linear.in_features, self.linear.out_features, device=self.linear.weight.device)
        with torch.no_grad():
            out.weight.copy_(self.linear.weight + self.delta().t())
            if self.linear.bias is not None:
                out.bias.copy_(self.linear.bias)
        return out

    def set_rank(self, r):
        if r > self.r_max:
            raise ValueError(f"Rank {r} exceeds maximum rank {self.r_max}.")
        if r < 0:
            raise ValueError(f"Rank {r} must be non-negative.")
        self.r = r

    @torch.no_grad()
    def rebaseline(self):
        if self.warm_start == "merged":
            self.linear.weight+= self.delta().t()
        elif self.warm_start == "keep_base":
            pass  # Do nothing, keep the base weights 
        else:
            raise ValueError(f"Unknown rebaseline mode: {self.warm_start}")
        self.snapshot_base()
        self.reset_lora_parameters()
        # Only the adapters are stale. The base's Adam moments stay meaningful under
        # both modes: "merged" moves it by a legitimate update, and "keep_base" leaves
        # it untouched here -- though note set_phase(full=True) then unfreezes it, so
        # the *next* full iteration retrains it either way. "keep_base" therefore means
        # "do not fold the adapter in", NOT "the base is preserved".
        return [self.lora_A, self.lora_B]

    @property
    def scale(self):
        return self.alpha / self.r

    # ------------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def snapshot_base(self):
        """Record the base weights so the next full update can be measured against them."""
        self._w_snapshot = self.linear.weight.detach().clone() # the way snapshots are made right now is not saving any memory, in fact we add more memory with the storage of adaptaters. But it's ok since this is still a prototype. #TODO: more memory efficient snapshotting of the base weights

    @torch.no_grad()
    def base_update_svals(self):
        """Singular values of the *unconstrained* update the base made this iteration.

        Answers "what rank would a full retrain have used?", which is the reference
        the LoRA rank r should be compared against.
        """
        if getattr(self, "_w_snapshot", None) is None:
            return None
        return torch.linalg.svdvals(self.linear.weight - self._w_snapshot)

    @torch.no_grad()
    def adapter_svals(self):
        """Singular values of the current adapter offset, in weight space."""
        return torch.linalg.svdvals(self.delta())

    @torch.no_grad()
    def adapter_rel_norm(self):
        """||scale*BA||_F / ||W||_F -- how far the group has drifted from its base."""
        return (self.delta().norm() / self.linear.weight.norm()).item()

    def delta(self):
        if self.r == 0:
            return torch.zeros((self.linear.in_features, self.linear.out_features), device=self.linear.weight.device)
        a = self.lora_A[:, :self.r]  # Might cause problems if r regrows with Adam optimizer
        b = self.lora_B[:self.r, :]
        delta = (a @ b * self.scale)
        return delta

    