"""Optional full history of every StaQ iterate (not just the last ``max_expand``).

Why this exists
---------------
The live Q-function keeps only the last ``max_expand`` frozen iterates: every
``StackedNN.push`` drops the oldest one (``dwex_nn.py``). That bounded stack *is*
the StaQ sum, so its size is an algorithmic choice -- you cannot enlarge it just
to keep history without changing the learned policy. This module logs every
iterate *separately*, for analysis only, without touching training.

Design
------
- Snapshots are captured from the *trainable* net (``train_feat`` / ``train_q``),
  i.e. before the head decay is applied -> clean successive iterates, no decay
  artifact (better for the rank/PCA analysis than the stored, decayed stack).
- Capture reuses ``StackedNN`` (max_size=1) so the weight layout is guaranteed
  identical to what ``froz_feat`` stores (mlp transposed, conv as-is, BatchRenorm
  folded). Each snapshot is moved to CPU immediately to keep GPU memory flat.
- The saved archive is read back into an object exposing ``.froz_feat`` and
  ``.sig_q`` with ``.sweights / .sbiases / .layer_types``, so the existing
  ``analyse_qfunc`` runs on it unchanged.
"""

import os

import torch

from src.networks.stacked_nn import StackedNN


def _snapshot(module_iterable, strides=None):
    """One iterate, in StackedNN convention, on CPU.

    Returns (sweights, sbiases, layer_types); each sweight has a leading dim of 1
    (StackedNN's stacking axis), so consecutive snapshots concatenate into (N, ...).
    """
    snn = StackedNN(module_iterable, max_size=1, strides=strides)
    sweights = [w.detach().cpu().clone() for w in snn.sweights]
    sbiases = [b.detach().cpu().clone() for b in snn.sbiases]
    return sweights, sbiases, list(snn.layer_types)


class QArchive:
    """Accumulates every iterate of a single qfunc (features + head), per layer."""

    def __init__(self):
        self._feat_w = self._feat_b = self._feat_t = None
        self._head_w = self._head_b = self._head_t = None

    def push(self, qfunc):
        """Append the current trainable iterate. Call once per StaQ update."""
        fw, fb, ft = _snapshot(qfunc.train_feat, strides=qfunc.strides)
        hw, hb, ht = _snapshot([qfunc.train_q])
        if self._feat_w is None:
            self._feat_w = [[w] for w in fw]
            self._feat_b = [[b] for b in fb]
            self._feat_t = ft
            self._head_w = [[w] for w in hw]
            self._head_b = [[b] for b in hb]
            self._head_t = ht
        else:
            assert(self._feat_b is not None and self._head_b is not None and self._head_w is not None and self._head_b is not None), "QArchive.push called after .to_dict()"
            for dst, w in zip(self._feat_w, fw):
                dst.append(w)
            for dst, b in zip(self._feat_b, fb):
                dst.append(b)
            for dst, w in zip(self._head_w, hw):
                dst.append(w)
            for dst, b in zip(self._head_b, hb):
                dst.append(b)

    @staticmethod
    def _stack(list_of_lists, dtype):
        return [torch.cat(snaps, dim=0).to(dtype) for snaps in list_of_lists]

    def to_dict(self, half=False):
        dtype = torch.float16 if half else torch.float32
        return {
            "froz_feat_sweights": self._stack(self._feat_w, dtype),
            "froz_feat_sbiases": self._stack(self._feat_b, dtype),
            "froz_feat_layer_types": self._feat_t,
            "sig_q_sweights": self._stack(self._head_w, dtype),
            "sig_q_sbiases": self._stack(self._head_b, dtype),
            "sig_q_layer_types": self._head_t,
        }


def save_all_q_checkpoint(path, archives, config, total_trans, env_name, n_act, s_dim, half=False):
    """Serialize the full-history archives (one QArchive per qfunc)."""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    torch.save(
        {
            "q_archives": [a.to_dict(half=half) for a in archives],
            "config": config,
            "total_trans": total_trans,
            "env_name": env_name,
            "n_act": n_act,
            "s_dim": s_dim,
            "dtype": "float16" if half else "float32",
            "format": "staq_all_qfuncs_v1",
        },
        path,
    )



# --- read side: make an archive look like a qfunc for analyse_qfunc ---------


class _ArchivedStack:
    """Minimal stand-in for a StackedNN: only what analyse_qfunc reads."""

    def __init__(self, sweights, sbiases, layer_types):
        self.sweights = [w.float() for w in sweights]
        self.sbiases = [b.float() for b in sbiases]
        self.layer_types = list(layer_types)


class ArchivedQFunc:
    """Exposes .froz_feat / .sig_q so analyse_qfunc(archived_qfunc) works as-is."""

    def __init__(self, d):
        self.froz_feat = _ArchivedStack(
            d["froz_feat_sweights"], d["froz_feat_sbiases"], d["froz_feat_layer_types"]
        )
        self.sig_q = _ArchivedStack(
            d["sig_q_sweights"], d["sig_q_sbiases"], d["sig_q_layer_types"]
        )


def load_all_q_checkpoint(path, device="cpu"):
    """Load a full-history checkpoint; ``checkpoint['qfuncs']`` are ArchivedQFuncs."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    checkpoint["qfuncs"] = [ArchivedQFunc(d) for d in checkpoint["q_archives"]]
    return checkpoint
