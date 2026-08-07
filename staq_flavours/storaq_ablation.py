
from typing import cast

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim.adam import Adam

from src.networks.lora_adapter import LoRALinear
from src.networks.staq_net import StaQNet
from staq_flavours.storaq import StoRAQTrainer
from staq_flavours.storaq_ablation_config import ShadowStoRAQConfig


class ShadowStoRAQTrainer(StoRAQTrainer):
    """Ablation Study to compare the behaviour between restarting a full train round from the LoRA adaptater vs a shadow network trained in parallel.

    Args:
        StoRAQTrainer (_type_): _description_
    """

    def _init_qfuncs(self):
        super()._init_qfuncs()
        self.shadow_qfuncs = [StaQNet(self.s_dim,
                                      self.n_act,
                                      nb_hidden=self.cfg.nb_hidden,
                                      hidden_width=self.cfg.hidden_width,
                                      memory_size=self.cfg.memory_size,
                                      use_w_correction=self.cfg.w_correction,
                                      kl_weight = self.cfg.kl_weight,
                                      entropy_weight = self.cfg.init_ew,
                                      device=self.device,
                                      network_type=self.cfg.network_type,
                                      cnn_config=self.cnn_config) for _ in range(self.cfg.n_ensemble)]
        self.shadow_optim = Adam([p for q in self.shadow_qfuncs for p in q.parameters() ], lr=self.cfg.lr, weight_decay=self.cfg.l2_weight)


    def _extra_grad_step(self, db, targ, nologits, idxs):
        self.shadow_optim.zero_grad()
        targ = targ.detach()
        curr_qalls = [qfunc(db.obs) for qfunc in self.shadow_qfuncs]
  

        if self.cfg.mode == 'mean':
            lossq = torch.stack([(qall.gather(dim=1, index=db.act) - targ).pow(2).mean()
                    for qall in curr_qalls]).mean()
        else:
            lossq = torch.stack([(qall.gather(dim=1, index=db.act) - targ).pow(2).mean()
                            for qall in curr_qalls]).mean()
        lossq.backward()
        self.shadow_optim.step()
        self._shadow_loss = lossq.item()

    def _log_iteration(self):
        super()._log_iteration()
        # Same targets, same batches as loss/bellerror -- so the gap between the
        # two curves is the capacity cost of the rank-r constraint, isolated.
        if getattr(self, '_shadow_loss', None) is not None:
            self._log('loss/shadow_bellerror', self._shadow_loss, self._phase_iter)

    def _apply_phase(self, full, r):
        if full:
            self._inject_shadow()
        for q in self.qfuncs:
            q.train_feat.requires_grad_(False)
            for lay in q.lora_layers():
                lay.lora_A.requires_grad_(True)
                lay.lora_B.requires_grad_(True)
                if full:
                    lay.set_rank(r)

    @torch.no_grad()
    def _inject_shadow(self):
        for qfunc, shadow_qfunc in zip(self.qfuncs, self.shadow_qfuncs):
            for layer, shadow_layer in zip(qfunc.train_feat, shadow_qfunc.train_feat):
                base = layer.linear if isinstance(layer, LoRALinear) else layer
                if isinstance(base, nn.Linear):
                    # shadow is a plain StaQNet: its layers are nn.Linear, not LoRALinear
                    base.weight.copy_(shadow_layer.weight)
                    if base.bias is not None:
                        base.bias.copy_(shadow_layer.bias)
            for layer in qfunc.lora_layers():
                layer.reset_lora_parameters()
        for q in self.qfuncs:
            for layer in q.lora_layers():
                self.q_optim.state.pop(layer.lora_A, None)
                self.q_optim.state.pop(layer.lora_B, None)


@hydra.main(version_base=None, config_path="../conf", config_name="storaq_ablation")
def main(cfg: DictConfig):
    run_cfg = cast(ShadowStoRAQConfig, OmegaConf.to_object(cfg))

    if run_cfg.wandb:
        import wandb

        wandb.init(
            project=run_cfg.wandb_project,
            config=cast(dict, OmegaConf.to_container(cfg, resolve=True)),
            sync_tensorboard=True,
            name=run_cfg.run_name,
            group=run_cfg.run_name.rsplit('_s', 1)[0],
            monitor_gym=True,
        )
    logging_path = HydraConfig.get().runtime.output_dir
    ShadowStoRAQTrainer(run_cfg, logging_path).run()


if __name__ == "__main__":
    main()
