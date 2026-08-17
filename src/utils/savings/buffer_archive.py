import os

import torch

from src.utils.replay_memory import ReplayMemory


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

def load_buffer_archive(buffer_dir:str, buffer_number:int|None=None):
    """Load a buffer archive from disk. The manifest is loaded and the buffers are loaded sequentially.

    Args:
        buffer_dir (str): the directory where the buffer archive is stored
        buffer_number (int | None): the number of the buffer to load, if None loads all buffers
    Returns:
        list[dict]: a list of buffers, each buffer is a dict with keys "iteration", "size", "max_size", "write_idx", "buffer"
    """
    # weights_only=False is explicit, not incidental: torch 2.6 flipped the
    # default to True, and these archives store numpy scalars in the manifest
    # and the buffer config, which the restricted unpickler rejects. Matches
    # load_inference_checkpoint. The files are our own training output.
    manifest = torch.load(os.path.join(buffer_dir, "manifest.pt"), weights_only=False)
    buffers = []
    if buffer_number is None:
        for entry in manifest["manifest"]:
            fname = entry["filename"]
            buffer = torch.load(os.path.join(buffer_dir, fname), weights_only=False)
            buffers.append(buffer)
    else:
        entry = manifest["manifest"][buffer_number]
        fname = entry["filename"]
        buffer = torch.load(os.path.join(buffer_dir, fname), weights_only=False)
        buffers.append(buffer)
    return buffers