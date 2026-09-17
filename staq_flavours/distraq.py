from itertools import permutations
from time import time
from typing import cast

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.optim.adam import Adam

from src.networks.distraq_net import DistraQNet
from src.staq import StaQTrainer
from src.utils.rl_tools import centered_l1, centered_mse, kl_loss, reverse_kl_loss
from staq_flavours.distraq_config import DistraQConfig

# Distillation objectives, keyed by cfg.student.loss. Resolved once in _init_qfuncs,
# so a typo fails at startup rather than at the first distillation.
DISTIL_LOSSES = {
    "kl": kl_loss,
    "centered_mse": centered_mse,
    "centered-mse": centered_mse,
    "centered_l1": centered_l1,
    "reverse_kl": reverse_kl_loss,
}


def _policy_kl(ref_logits: torch.Tensor, est_logits: torch.Tensor) -> torch.Tensor:
    """Per-state KL(softmax(ref) || softmax(est))."""
    ref_logp, est_logp = ref_logits.log_softmax(-1), est_logits.log_softmax(-1)
    return (ref_logp.exp() * (ref_logp - est_logp)).sum(-1)


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = logits.log_softmax(-1)
    return -(logp.exp() * logp).sum(-1)


def _compare(estimate: torch.Tensor, reference: torch.Tensor) -> dict[str, torch.Tensor]:
    """Agreement between two (states, n_act) logit tensors in the same (eta-scaled) units.

    Everything is shift-invariant per state, like the softmax policy itself:
      rel_err       ||centered(est - ref)|| / ||centered(ref)||, averaged over states
      scale_ratio   mean ||centered(est)|| / mean ||centered(ref)||; < 1 means est is flatter
      greedy_agree  fraction of states with the same argmax
      kl, kl_p90    KL(pi_ref || pi_est): mean and upper tail, where the damage concentrates
      entropy_gap   H(pi_est) - H(pi_ref); > 0 means est is more hesitant
    """
    est_c = estimate - estimate.mean(-1, keepdim=True)
    ref_c = reference - reference.mean(-1, keepdim=True)
    ref_norm = ref_c.norm(dim=-1)
    kl = _policy_kl(reference, estimate)
    return {
        "rel_err": ((est_c - ref_c).norm(dim=-1) / ref_norm.clamp_min(1e-9)).mean(),
        "scale_ratio": est_c.norm(dim=-1).mean() / ref_norm.mean().clamp_min(1e-9),
        "greedy_agree": (estimate.argmax(-1) == reference.argmax(-1)).float().mean(),
        "kl": kl.mean(),
        "kl_p90": kl.quantile(0.9),
        "entropy_gap": (_entropy(estimate) - _entropy(reference)).mean(),
    }


class DistraQTrainer(StaQTrainer):
    cfg: DistraQConfig
    AGREEMENT_STATES = 1024
    AGREEMENT_PAIRS = (
        ("student", "exact"),   # E_k: total error of what the agent actually acts with
        ("target", "exact"),    # lambda * E_{k-1}: error inherited from previous rounds
        ("student", "target"),  # eps_k: this round's distillation error alone
    )

    def _make_qfuncs(self) -> list:
        return [DistraQNet(self.s_dim,
                           self.n_act,
                           nb_hidden=self.cfg.nb_hidden,
                           hidden_width=self.cfg.hidden_width,
                           memory_size=self.cfg.memory_size,
                           kl_weight = self.cfg.kl_weight,
                           entropy_weight = self.cfg.init_ew,
                           use_w_correction=self.cfg.w_correction,
                           cfg_student=self.cfg.student,
                           device=self.device,
                           network_type=self.cfg.network_type,
                           cnn_config=self.cnn_config,) for _ in range(self.cfg.n_ensemble)]

    def _init_qfuncs(self):
        super()._init_qfuncs()
        try:
            self.distil_loss = DISTIL_LOSSES[self.cfg.student.loss]
        except KeyError:
            raise ValueError(f"unknown student.loss {self.cfg.student.loss!r}; "
                             f"expected one of {sorted(DISTIL_LOSSES)}") from None
        self.student_optimizer = Adam([p for q in self.qfuncs for p in q.student.parameters()], lr=self.cfg.student.lr)

    def _distil(self):
        n = min(self.cfg.student.distil_states, self.repmem.size)
        obs = self.repmem.sample(n, device=self.device).obs
        probe = obs[:self.AGREEMENT_STATES]  # obs is already a random sample
        stats = {"kl_loss_before": [], "kl_loss_after": [], "kl_new_vs_old_policy": []}
        snapshots = []  # per member, logits on the probe, for the agreement stats

        for q in self.qfuncs:
            with torch.no_grad():
                student_before = q.student(obs)
                # kl_loss softmaxes both arguments, so target stays as raw logits.
                target = q.decay * student_before + q(obs)
                stats["kl_loss_before"].append(kl_loss(student_before * q.eta, target * q.eta).item())

            q.student.train(True)
            for _ in range(self.cfg.student.distil_steps):
                idx = torch.randint(0, len(obs), (self.cfg.student.batch_size,), device=self.device)
                loss = self.distil_loss(q.student(obs[idx]) * q.eta, target[idx] * q.eta)
                self.student_optimizer.zero_grad()
                loss.backward()
                self.student_optimizer.step()
            q.student.train(False)

            with torch.no_grad():
                student_after = q.student(obs)
                stats["kl_loss_after"].append(kl_loss(student_after * q.eta, target * q.eta).item())
                # How far this round moved the policy, KL(pi_new || pi_old). This replaces
                # the base log/kl_pk_pkpone, which is identically 0 for DistraQ: it is
                # computed before _distil, while the student is still unchanged.
                stats["kl_new_vs_old_policy"].append(_policy_kl(student_after * q.eta, student_before * q.eta).mean().item())
                if self.cfg.student.keep_archive:
                    scale = q.eta * q.w_correction  # the same scale get_logits applies
                    snapshots.append({
                        "student": q.get_logits(probe),
                        "target": target[:len(probe)] * scale,
                        "exact": q.get_staq_logits(probe),
                    })

        for name, values in stats.items():
            self._log(f"distil/{name}", float(np.mean(values)))
        if snapshots:
            self._log_agreement(snapshots)

    def _log_agreement(self, members: list[dict[str, torch.Tensor]]):
        """Student vs the exact StaQ sum, both built from the very same Q_k.

        Caveat: `exact` is the archive, truncated at memory_size, so it is off by lambda**M
        of the mass -- ~0.5% on Breakout but ~2.5% on Asterix at M=300. Raise memory_size
        for diagnostic runs rather than chasing an effect of that size.
        """
        # Ensemble level, averaging logits as get_logits_ensemble_torch does: what acts.
        ensemble = {k: torch.stack([m[k] for m in members]).mean(0) for k in members[0]}
        for est, ref in self.AGREEMENT_PAIRS:
            for name, value in _compare(ensemble[est], ensemble[ref]).items():
                self._log(f"agreement/{est}_vs_{ref}/{name}", value.item())

        # Same headline pair averaged over members. The gap to the ensemble-level numbers
        # is what averaging logits across members contributes on its own.
        per_member = [_compare(m["student"], m["exact"]) for m in members]
        for name in per_member[0]:
            self._log(f"agreement/student_vs_exact_members/{name}",
                      torch.stack([d[name] for d in per_member]).mean().item())

        # Do the members disagree more after distillation than their exact sums do? If the
        # student disagreement is well above the exact one, distillation manufactures it.
        if len(members) > 1:
            for k in ("student", "exact"):
                kls = [_policy_kl(members[i][k], members[j][k]).mean()
                       for i, j in permutations(range(len(members)), 2)]
                self._log(f"agreement/member_disagreement/{k}", torch.stack(kls).mean().item())

    def _update_staq_networks(self, old_logits_tilde, testb, old_dist):
        elapsed_time_training = super()._update_staq_networks(old_logits_tilde, testb, old_dist)
        start_time_distil = time()
        self._distil()
        elapsed_time_distil = time() - start_time_distil
        self._log('distil/time', elapsed_time_distil, self.total_trans)
        return elapsed_time_training + elapsed_time_distil


@hydra.main(version_base=None, config_path="../conf", config_name="distraq")
def main(cfg: DictConfig):
    run_cfg = cast(DistraQConfig, OmegaConf.to_object(cfg))
    if run_cfg.wandb:
        import wandb
        wandb.init(project=run_cfg.wandb_project, name=run_cfg.run_name, config=cast(dict, OmegaConf.to_container(cfg, resolve=True)),
                   sync_tensorboard=True, group=run_cfg.run_name.rsplit('_s', 1)[0])
    logging_path = HydraConfig.get().runtime.output_dir
    DistraQTrainer(run_cfg, logging_path).run()

if __name__ == "__main__":
    main()
