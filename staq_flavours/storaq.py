
from typing import cast

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.networks.lora_net import SToRAQNet
from src.staq import StaQTrainer
from src.utils.spectrum import spectrum_stats
from staq_flavours.storaq_config import StoRAQConfig


class StoRAQTrainer(StaQTrainer):
    """
    A trainer for the StoRAQ flavour of StaQ. This trainer is used to train a
    StaQ model with the StoRAQ flavour, which uses a stochastic rank-adaptive
    quantization method to compress the model.
    """

    cfg: StoRAQConfig  # narrows the parent's RunConfig annotation

    def __init__(self, cfg: StoRAQConfig, logging_path: str):
        super().__init__(cfg, logging_path)  # calls _init_qfuncs()
        self.rank_schedule = instantiate(cfg.lora.schedule)

    def _make_qfuncs(self) -> list:
        return [SToRAQNet(self.s_dim,
                               self.n_act,
                               max_rank=self.cfg.lora.max_rank,
                               lora_layers_index=self.cfg.lora.lora_layers_index,
                               warm_start=self.cfg.lora.warm_start,
                               alpha=self.cfg.lora.alpha,
                               freeze_non_lora=self.cfg.lora.freeze_non_lora,
                               nb_hidden=self.cfg.nb_hidden,
                               hidden_width=self.cfg.hidden_width,
                               memory_size=self.cfg.memory_size,
                               use_w_correction=self.cfg.w_correction,
                               kl_weight = self.cfg.kl_weight,
                               entropy_weight = self.cfg.init_ew,
                               device=self.device,
                               network_type=self.cfg.network_type,
                               cnn_config=self.cnn_config) for _ in range(self.cfg.n_ensemble)]

    def _update_schedules(self):
        super()._update_schedules()
        full, r = self.rank_schedule.phase(self.iter)
        # Remembered for _log_iteration: total_trans has advanced by then, so
        # self.iter no longer names the iteration this phase belongs to.
        self._phase_full, self._phase_r, self._phase_iter = full, r, self.iter
        if full:
            stale = [p for q in self.qfuncs for p in q.rebaseline()]
            for p in stale:
                self.q_optim.state.pop(p, None)
        for q in self.qfuncs:
            q.set_phase(full, r)

    def _log_iteration(self):
        step = self._phase_iter
        self._log('lora/is_full_retrain', float(self._phase_full), step)
        self._log('lora/rank', float(self._phase_r), step)

        n_layers = len(self.qfuncs[0].lora_layers())
        for idx in range(n_layers):
            layers = [q.lora_layers()[idx] for q in self.qfuncs]
            tag = f'lora/L{idx}'

            # How far this group has drifted from its base.
            self._log(f'{tag}/adapter_rel_norm',
                      float(np.mean([l.adapter_rel_norm() for l in layers])), step)

            # Rank actually used by the adapter. Below the allowed r means the
            # bottleneck is not binding; pinned at r means it probably is.
            adapter = [spectrum_stats(l.adapter_svals()) for l in layers]
            for key in ('eff_rank', 'rank90', 'rank99'):
                self._log(f'{tag}/adapter_{key}',
                          float(np.mean([s[key] for s in adapter])), step)

            # On a full retrain: the spectrum of the *unconstrained* update.
            # This is the reference -- it says what rank the update wanted.
            if self._phase_full:
                svals = [l.base_update_svals() for l in layers]
                svals = [s for s in svals if s is not None]
                if svals:
                    base = [spectrum_stats(s) for s in svals]
                    for key in ('eff_rank', 'rank90', 'rank99', 's_max'):
                        self._log(f'{tag}/full_update_{key}',
                                  float(np.mean([s[key] for s in base])), step)
                    self.logger.add_histogram(f'{tag}/full_update_svals',
                                              svals[0].cpu(), step)


@hydra.main(version_base=None, config_path="../conf", config_name="storaq")
def main(cfg: DictConfig):
    run_cfg = cast(StoRAQConfig, OmegaConf.to_object(cfg))

    if run_cfg.wandb:
        import wandb

        wandb.init(
            project=run_cfg.wandb_project,
            # to_container needs the DictConfig, not the dataclass instance
            config=cast(dict, OmegaConf.to_container(cfg, resolve=True)),
            sync_tensorboard=True,
            name=run_cfg.run_name,
            group=run_cfg.run_name.rsplit('_s', 1)[0],  # seeds of one condition together
            monitor_gym=True,
        )
    logging_path = HydraConfig.get().runtime.output_dir
    StoRAQTrainer(run_cfg, logging_path).run()


if __name__ == "__main__":
    main()
