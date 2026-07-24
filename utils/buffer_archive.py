import os
import torch

from rl_tools import ReplayMemory


class BufferArchive:
    """Accumulates each buffer for each iteration"""
    def __init__(self, save_dir:str):
        self.index = 0
        self.save_dir = save_dir
        self.manifest = []
        os.makedirs(save_dir, exist_ok=True)
    
    def _snapshot(self, repmem:ReplayMemory, half:bool=False):
        """Gets the current replay memory and extract the (s,a,r,s) named tuples. Store them into the buffer list. they are stored sequentially. A future update could only care about the last ones.

        Args:
            repmem (ReplayMemory): the replay memory used at the end of training of the current Q function
            half (bool): store floating fields (obs/nobs/rwd/terminated) as fp16 to halve disk
        """
        def _store(t):
            t = t.detach().cpu().clone()
            if half and t.is_floating_point():
                t = t.half()
            return t
        dict_buffer = {"iteration": self.index, 
                       "size": repmem.size, 
                       "max_size": repmem.max_size,
                       "write_idx": repmem.write_idx,
                       "buffer": {k: _store(repmem.get(k)) for k in repmem.repmem._fields}}
        return dict_buffer
    
    def push(self, repmem:ReplayMemory, half:bool=False):
        snap = self._snapshot(repmem, half)
        fname = f"buffer_q{self.index:04d}.pt"
        torch.save(snap, os.path.join(self.save_dir, fname))
        self.manifest.append({"iteration": self.index, "filename": fname, "size": snap["size"], "max_size": snap["max_size"]})
        self.index += 1
        del snap
    
    def save_manifest(self, config, total_trans, env_name, n_act, s_dim):
        torch.save({"format": "staq_buffer", "manifest": self.manifest, "config": config, "total_trans": total_trans, "env_name": env_name, "n_act": n_act, "s_dim": s_dim}, os.path.join(self.save_dir, "manifest.pt"))
