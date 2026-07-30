from collections import namedtuple

import numpy as np
import torch


class ReplayMemory:
    def __init__(self, max_size, state_shape, device, obs_type=torch.float32):
        self.device = device
        self.max_size = max_size
        self.size = 0
        self.write_idx = 0
        self.ReplayMemorySamples = namedtuple('ReplayMemorySamples',
                                              ['obs', 'act', 'rwd', 'terminated', 'nobs'])

        # Handle both CNN and MLP input shapes
        if isinstance(state_shape, int):
            state_shape = (state_shape,)

        self.repmem = self.ReplayMemorySamples(
            obs=torch.zeros(max_size, *state_shape, device=self.device, dtype=obs_type),
            act=torch.zeros(max_size, 1, device=self.device).long(),
            rwd=torch.zeros(max_size, 1, device=self.device),
            terminated=torch.zeros(max_size, 1, device=self.device),
            nobs=torch.zeros(max_size, *state_shape, device=self.device, dtype=obs_type),
        )

    def add_trans(self, trans):
        add_len = len(trans['rwd'])
        overflow = add_len + self.write_idx > self.max_size
        len_first_copy = self.max_size - self.write_idx
        data = self.repmem._asdict()
        for k in data:
            if k == 'act':
                v = trans[k].long()
            else:
                v = trans[k]
            if overflow:
                data[k][self.write_idx:] = v[:len_first_copy]
                data[k][0:add_len - len_first_copy] = v[len_first_copy:]
            else:
                data[k][self.write_idx:self.write_idx + add_len] = v
        self.write_idx = (self.write_idx + add_len) % self.max_size
        self.size = min(self.size + add_len, self.max_size)

    def sample_with_idxs(self, batch_size, device=None):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        if device is None:
            return self.ReplayMemorySamples(
                obs=self.repmem.obs[idxs].to(dtype=torch.float32),
                act=self.repmem.act[idxs],
                rwd=self.repmem.rwd[idxs],
                terminated=self.repmem.terminated[idxs],
                nobs=self.repmem.nobs[idxs].to(dtype=torch.float32),
            ), idxs
        else:
            return self.ReplayMemorySamples(
                obs=self.repmem.obs[idxs].to(device=device, dtype=torch.float32),
                act=self.repmem.act[idxs].to(device=device),
                rwd=self.repmem.rwd[idxs].to(device=device),
                terminated=self.repmem.terminated[idxs].to(device=device),
                nobs=self.repmem.nobs[idxs].to(device=device, dtype=torch.float32),
            ), idxs

    def sample(self, batch_size, device=None):
        return self.sample_with_idxs(batch_size, device)[0]
    
    def _get_idexes(self, start_idx, end_idx):  # start and end indexes are relative to how old is the data
        assert start_idx < end_idx
        datastarts = self.write_idx if self.size == self.max_size else 0
        lidx = (datastarts + start_idx) % self.max_size
        ridx = (datastarts + end_idx) % self.max_size
        return lidx, ridx

    def get(self, key, start_idx=None, end_idx=None):  # start and end indexes are relative to how old is the data
        if start_idx is None and end_idx is None:
            return getattr(self.repmem, key)[:self.size]
        else:
            lidx, ridx = self._get_idexes(start_idx, end_idx)
            if lidx < ridx:
                return getattr(self.repmem, key)[lidx:ridx]
            else:
                return torch.cat([getattr(self.repmem, key)[lidx:],
                                  getattr(self.repmem, key)[:ridx]], dim=0)
