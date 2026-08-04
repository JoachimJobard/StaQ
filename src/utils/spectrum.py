import torch


def spectrum_stats(svals: torch.Tensor, energy_levels=(0.9, 0.99)) -> dict:
    """Summarise a singular-value spectrum with rank-like scalars.

    Args:
        svals (torch.Tensor): singular values, descending (as returned by ``svdvals``).
        energy_levels (tuple[float, ...]): cumulative-energy thresholds; for each one
            we report the smallest ``k`` whose top-``k`` subspace captures that share
            of ``sum(s^2)``.

    Returns:
        dict: ``eff_rank`` (entropy-based effective rank, Roy & Vetterli), ``s_max``,
        ``rank{level}`` per requested energy level, and ``nnz`` (numerically nonzero
        singular values).
    """
    s = svals.detach().float()
    total = s.sum()
    if total <= 0:
        return {"eff_rank": 0.0, "s_max": 0.0, "nnz": 0.0,
                **{f"rank{int(l * 100)}": 0.0 for l in energy_levels}}

    # Entropy-based effective rank: exp(-sum p log p) with p the normalised spectrum.
    # Equals k for a flat rank-k spectrum and 1 for a rank-1 one, and unlike a
    # hard threshold it degrades smoothly as small components appear.
    p = s / total
    eff_rank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()

    energy = torch.cumsum(s.pow(2), 0) / s.pow(2).sum()
    out = {"eff_rank": eff_rank,
           "s_max": s[0].item(),
           "nnz": float((s > s[0] * 1e-6).sum().item())}
    for level in energy_levels:
        out[f"rank{int(level * 100)}"] = float((energy < level).sum().item() + 1)
    return out
