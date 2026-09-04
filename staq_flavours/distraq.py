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
from src.utils.rl_tools import centered_l1, centered_mse, kl_loss
from staq_flavours.distraq_config import DistraQConfig


class DistraQTrainer(StaQTrainer):
    cfg: DistraQConfig 
    def __init__(self, cfg: DistraQConfig, logging_path: str):
        super().__init__(cfg, logging_path)  # calls _init_qfuncs()

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
                           cnn_config=self.cnn_config) for _ in range(self.cfg.n_ensemble)]

    def _init_qfuncs(self):
        super()._init_qfuncs()
        self.student_optimizer = Adam([p for q in self.qfuncs for p in q.student.parameters()], lr=self.cfg.student.lr)

    def _update_schedules(self):
        super()._update_schedules()
        # self._distil(self.repmem.sample(self.cfg.student.distil_states, device=self.device).obs)

    def _distil(self):
        n = min(self.cfg.student.distil_states, self.repmem.size)
        obs = self.repmem.sample(n, device=self.device).obs
        before, after = [], []
        for q in self.qfuncs:
            with torch.no_grad():
                # kl_loss softmaxes both arguments, so target stays as raw logits.
                target = q.decay * q.student(obs) + q(obs)
                before.append(kl_loss(q.student(obs) * q.eta, target * q.eta).item())

            q.student.train(True)  # set student to training mode
            for step in range(self.cfg.student.distil_steps):   
                idx = torch.randint(0, len(obs), (self.cfg.student.batch_size,), device=self.device)
                student_logits_batch = q.student(obs[idx]) * q.eta    
                # Compute the KL divergence loss
                target_batch = target[idx] * q.eta
                if q.cfg_student.loss ==  "kl":
                    loss = kl_loss(student_logits_batch, target_batch)
                elif q.cfg_student.loss == "centered_l1":
                    loss = centered_l1(student_logits_batch, target_batch)
                elif q.cfg_student.loss in ["centered-mse","centered_mse"]:
                    loss = centered_mse(student_logits_batch, target_batch)
                else:
                    raise NotImplementedError
                loss.backward()
                self.student_optimizer.step()
                self.student_optimizer.zero_grad()
            with torch.no_grad():
                after.append(kl_loss(q.student(obs)*q.eta, target*q.eta).item())
        self._log('distil/kl_loss_before', float(np.mean(before)), self.total_trans)
        self._log('distil/kl_loss_after', float(np.mean(after)), self.total_trans)

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
