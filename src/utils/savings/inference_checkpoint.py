"""Read side for the legacy ``staq_inference_qfuncs_v2`` checkpoint format.

Why this exists
---------------
Checkpoints written before the big refactor pickled *live objects* -- a
``dwex_nn.StaQInference`` holding two ``dwex_nn.StackedNN`` instances -- rather
than a plain tensor dict. Pickle stores class *paths*, not class *code*, so the
file literally references ``dwex_nn.StackedNN``. That module no longer exists
(it became ``src/networks/stacked_nn.py``, and ``StaQInference`` was dropped
entirely), so a bare ``torch.load`` raises ``ModuleNotFoundError``.

This module registers a stand-in ``dwex_nn`` in ``sys.modules`` so the old
pickles resolve again. ``StackedNN``'s attribute layout is unchanged across the
refactor, so the saved state drops straight into the current class.

Not to be confused with ``q_archive.py``, which reads ``staq_all_qfuncs_v1`` --
a different, tensor-dict format written by ``save_all_q_checkpoint``.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.networks.stacked_nn import StackedNN

FORMAT = "staq_inference_qfuncs_v2"
_LEGACY_MODULE = "dwex_nn"


class StaQInference:
    """Callable stand-in for the pre-refactor ``dwex_nn.StaQInference``.

    NOTE: instances are never created by calling this class. Pickle allocates
    them with ``object.__new__`` and then injects the saved fields straight into
    ``__dict__``, so ``__init__`` is never invoked -- which is precisely why
    there isn't one. Any default must live at *class* level (below).

    The fields injected by the unpickler are exactly: ``froz_feat``, ``sig_q``,
    ``eta``, ``w_correction``, ``output_size``, ``device``, ``max_expand``.
    ``normalize()`` tidies them up afterwards.
    """

    # Annotations only (no value assigned): supplied by the pickled state.
    froz_feat: StackedNN
    sig_q: StackedNN

    # Class-level defaults, since __init__ never runs.
    eta: float = 1.0
    w_correction: float = 1.0
    output_size: int = 0
    max_expand: int = 0

    # --- forward -----------------------------------------------------------

    def stacked_features(self, x: torch.Tensor) -> torch.Tensor:
        """Per-iterate features, shape ``(n_iterates, batch, hidden_width)``.

        ``StackedNN`` broadcasts a plain ``(batch, s_dim)`` input across the
        iterate axis on its first matmul, so no manual expansion is needed.
        Kept separate from ``__call__`` because the compression code needs this
        intermediate tensor and must not re-derive the forward pass.
        """
        return self.froz_feat(x)

    def logits_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Head + reduction + global scaling, shape ``(batch, output_size)``."""
        return self.sig_q(feats).sum(0) * self.eta * self.w_correction

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """StaQ logits for a ``(batch, s_dim)`` observation batch.

        Mirrors ``StaQNet.get_logits`` minus its training-time ``froz_feat is
        None`` guard.
        """
        return self.logits_from_features(self.stacked_features(x))

    # --- housekeeping ------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Where the weights actually are.

        Deliberately a property: the pickled ``device`` attribute records the
        machine the *run* happened on and is NOT updated by ``map_location``, so
        it can claim ``cuda`` while every tensor sits on CPU. ``normalize()``
        deletes the stale value; this reads the truth off the tensors.
        """
        return self.froz_feat.sweights[0].device

    @property
    def n_iterates(self) -> int:
        """Frozen iterates in the stack. This is ``max_expand + 1``:
        ``StackedNN.push`` only drops the oldest once the size *exceeds*
        ``max_size``."""
        return self.froz_feat.sweights[0].shape[0]

    def to(self, device: torch.device | str) -> StaQInference:
        """Move both stacks in place and return self.

        ``StackedNN`` is a plain class, not an ``nn.Module``, and its weights are
        ordinary tensors rather than ``nn.Parameter``s -- so it has no ``.to()``
        of its own and this has to be done by hand.
        """
        for stack in (self.froz_feat, self.sig_q):
            stack.sweights = [w.to(device) for w in stack.sweights]
            stack.sbiases = [b.to(device) for b in stack.sbiases]
        return self

    def normalize(self) -> StaQInference:
        """Repair the three warts in the pickled state. Called by the loader."""
        # 1. eta / w_correction / output_size come back as 0-d numpy scalars.
        #    Arithmetic is fine (torch treats numpy scalars as weak, so float32
        #    stays float32), but np.int64 leaking into shapes, range() and
        #    f-strings is friction.
        self.eta = float(self.eta)
        self.w_correction = float(self.w_correction)
        self.output_size = int(self.output_size)
        self.max_expand = int(self.max_expand)

        # 2. Non-linearities are stored as ReLU(inplace=True). Fine for a plain
        #    forward, but it breaks register_full_backward_hook on any preceding
        #    layer ("Output 0 of BackwardHookFunctionBackward is a view and is
        #    being modified inplace") -- which is exactly what
        #    LayerStatsCollector does.
        for stack in (self.froz_feat, self.sig_q):
            stack.non_lin = [
                nn.ReLU(inplace=False) if isinstance(f, nn.ReLU) else f
                for f in stack.non_lin
            ]

        # 3. Drop the stale pickled `device` so the property above is used.
        self.__dict__.pop("device", None)
        return self


@dataclass(frozen=True)
class CheckpointMeta:
    """Everything in the checkpoint that is not a Q-function."""

    config: dict[str, Any]
    env_name: str
    n_act: int
    s_dim: int
    total_trans: int
    dtype: str


def _install_legacy_shim() -> None:
    """Register a fake ``dwex_nn`` module so the old pickles resolve.

    Pickle never verifies that a module is real -- it does ``__import__`` then
    ``getattr``. Supplying an object with the right attribute names is enough.

    Runs at import time and is idempotent: the entry must exist in
    ``sys.modules`` before *any* ``torch.load`` in the process. Keeping it here
    also guarantees a single ``StackedNN`` class object; a duplicate shim
    elsewhere would create a second class and silently break ``isinstance``.
    """
    if _LEGACY_MODULE in sys.modules:
        return
    # `Any` because a bare ModuleType has no declared attributes; setting them
    # is the whole point here.
    shim: Any = types.ModuleType(_LEGACY_MODULE)
    shim.StackedNN = StackedNN
    shim.StaQInference = StaQInference
    sys.modules[_LEGACY_MODULE] = shim


_install_legacy_shim()


def load_inference_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
) -> tuple[list[StaQInference], CheckpointMeta]:
    """Load a ``staq_inference_qfuncs_v2`` checkpoint.

    Args:
        path: the ``.pt`` file written by the pre-refactor training script.
        device: where to put the weights. Loading always goes through CPU first
            and then moves explicitly, so a GPU-trained checkpoint opens on a
            CPU-only machine.

    Returns:
        ``(qfuncs, meta)`` -- one ``StaQInference`` per ensemble member, each
        directly callable on a ``(batch, s_dim)`` observation tensor.
    """
    # weights_only=True cannot work here: the payload is genuine pickled
    # objects, not a state dict.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    fmt = checkpoint.get("format")
    if fmt != FORMAT:
        hint = (
            " Use q_archive.load_all_q_checkpoint for this one."
            if fmt == "staq_all_qfuncs_v1"
            else ""
        )
        raise ValueError(f"expected format {FORMAT!r}, got {fmt!r}.{hint}")

    qfuncs = [q.normalize().to(device) for q in checkpoint["qfuncs"]]
    meta = CheckpointMeta(
        config=checkpoint["config"],
        env_name=str(checkpoint["env_name"]),
        n_act=int(checkpoint["n_act"]),
        s_dim=int(checkpoint["s_dim"]),
        total_trans=int(checkpoint["total_trans"]),
        dtype=str(checkpoint["dtype"]),
    )
    return qfuncs, meta
