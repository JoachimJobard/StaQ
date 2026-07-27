from os.path import join
from time import time
from typing import cast

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from networks.staq_net import StaQNet
from src.config import AppConfig, RunConfig
from src.utils.rl_tools import (
    _finish_action,
    linear_schedule,
    make_envs,
    make_network_type,
    stable_kl_div,
    update_target,
)
from utils.replay_memory import ReplayMemory
from utils.sampler import Sampler


class StaQTrainer:
    def __init__(self, cfg: RunConfig, logging_path: str):
        self.cfg = cfg
        assert cfg.target_type in ['hard', 'soft']
        assert cfg.mode in ['mean', 'min']
        torch.set_num_threads(2)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.env, self.env_eval, self.obs_type = make_envs(cfg.env_name)
        self.n_act = self.env.action_space.n
        self.s_dim = self.env.observation_space.shape if cfg.network_type == "cnn" else self.env.observation_space.shape[0]

        self.sampler = Sampler(self.env)
        self.sampler_eval = Sampler(self.env_eval)

        print('logging path', logging_path)
        self.logger = SummaryWriter(logging_path)

        self.cnn_config = make_network_type(cfg.network_type, cfg.env_name)
        self.qfuncs = [StaQNet(self.s_dim, 
                               self.n_act, 
                               nb_hidden=cfg.nb_hidden, 
                               hidden_width=cfg.hidden_width, 
                               memory_size=cfg.memory_size,
                               use_w_correction=cfg.w_correction,
                               kl_weight = cfg.kl_weight,
                               entropy_weight = cfg.init_ew,
                               device=self.device,
                               network_type=cfg.network_type,
                               cnn_config=self.cnn_config) for _ in range(cfg.n_ensemble)]
        self.q_optim = torch.optim.Adam([p for qfunc in self.qfuncs for p in qfunc.parameters()], lr=cfg.lr, weight_decay=cfg.l2_weight)
        self.qtars = update_target(self.qfuncs, update_type='hard') # initialize as the first q funcs

        self.repmem = ReplayMemory(cfg.rep_mem_size, self.s_dim, self.device, obs_type=self.obs_type)

        self.total_trans = 0
        self.total_time_elapsed = 0.0
        self.latest_train_return = None
        self.latest_eval_return = None
        self.progress_bar = tqdm(total=cfg.timesteps, unit='steps')    

    def run(self):
        while self.total_trans < self.cfg.timesteps:
            total_start_time = time()
            self._update_schedules()
            self._collect_rollouts()
            old_logits_tilde, testb, old_dist = self._policy_snapshot()
            self.training_start_time = time()
            nologits, precompute_time_elapsed = self._precompute_logits()
            self._train(nologits)
            training_time_elapsed = self._update_staq_networks(old_logits_tilde, testb, old_dist)
            if self.total_trans % self.cfg.eval_interval == 0:
                self.evaluate_policy()
            self.total_time_elapsed += time() - total_start_time

            self._log("timings/precompute", precompute_time_elapsed, self.total_trans)
            self._log("timings/training", training_time_elapsed, self.total_trans)
            self._log("timings/total", self.total_time_elapsed, self.total_trans)
            self._log('loss/max_abs_reward', self.sampler.max_abs_reward, self.total_trans)

            self.progress_bar.set_postfix({
                'train/return': self.latest_train_return,
                'eval/return': self.latest_eval_return,
            }, refresh=False)
            self.progress_bar.update(self.cfg.trans_per_iter)

        self.progress_bar.close()


    def evaluate_policy(self):
        [qfunc.train(False) for qfunc in self.qfuncs]
        eval_start_time = time()
        eval_rollouts = [self.sampler_eval.rollouts(self.numpy_argmax_policy, 1, np.inf, returns_only=True) for _ in range(self.cfg.n_eval_episodes)]
        self.latest_eval_return = np.mean([r[1][0].value for r in eval_rollouts])
        self._log('eval/return', self.latest_eval_return, self.total_trans)
        self._log('eval/mean_ep_length', np.mean([r[1][0].step for r in eval_rollouts]), self.total_trans)
        self._log('timings/eval', time() - eval_start_time, self.total_trans)


    def _train(self, nologits):
        max_q_ratio = 0
        grad_steps = 0
        [qfunc.train(True) for qfunc in self.qfuncs]
        max_ent = self.cfg.init_ew * np.log(self.n_act)
        max_abs_q = (self.cfg.rwd_scale * self.sampler.max_abs_reward + self.cfg.gamma * max_ent) / (1.0 - self.cfg.gamma)

        while grad_steps < int(self.cfg.udr * self.cfg.trans_per_iter):
            if self.cfg.target_type == 'hard' and grad_steps % self.cfg.hard_target_steps == 0:
                self.qtars = update_target(self.qfuncs, update_type='hard')
            elif self.cfg.target_type == 'soft':
                self.qtars = update_target(self.qfuncs, self.qtars, self.cfg.soft_target_polyak, update_type='soft')

            self.q_optim.zero_grad()
            db, idxs = self.repmem.sample_with_idxs(self.cfg.batch_size, device=self.device)
            curr_qalls = [qfunc(db.obs) for qfunc in self.qfuncs]
            with torch.no_grad():
                next_qalls = [qtar(db.nobs) for qtar in self.qtars]

            # For logging
            current_max_q_ratio = max([q.abs().max().item() for q in curr_qalls] + [q.abs().max().item() for q in next_qalls]) / max_abs_q
            max_q_ratio = max(max_q_ratio, current_max_q_ratio)

            with torch.no_grad():
                # sample next action
                nol = nologits[idxs]
                nopol = torch.distributions.Categorical(logits=nol)
                ## ----- Getting V(no) ------
                na = nopol.sample().view(-1, 1)
                qnos = [qall.gather(dim=1, index=na) for qall in next_qalls]

                ent_term = self.eweight * nopol.entropy()[:, None]

            scaled_rwd = self.cfg.rwd_scale * db.rwd
            if self.cfg.mode == 'mean':
                targ = sum([qno.detach() for qno in qnos]) / self.cfg.n_ensemble
                targ = scaled_rwd + self.cfg.gamma * (1 - db.terminated) * (targ + ent_term)
                lossq = torch.stack([(qall.gather(dim=1, index=db.act) - targ).pow(2).mean()
                        for qall in curr_qalls]).mean()
            else:
                targ = torch.hstack([qno.detach() for qno in qnos]).min(1, True)[0]
                targ = scaled_rwd + self.cfg.gamma * (1 - db.terminated) * (targ + ent_term)
                lossq = torch.stack([(qall.gather(dim=1, index=db.act) - targ).pow(2).mean()
                                for qall in curr_qalls]).mean()

            loss_logqdist = (((self.eweight * nol - sum(next_qalls) / self.cfg.n_ensemble) ** 2) * (1 - db.terminated)).mean()
            lossq.backward()

            if grad_steps % 100 == 0:
                self._log('loss/bellerror', lossq.item(), int(self.cfg.udr * (self.total_trans - self.cfg.trans_per_iter)) + grad_steps)
                self._log('loss/logqdist', loss_logqdist.item(), int(self.cfg.udr * (self.total_trans - self.cfg.trans_per_iter)) + grad_steps)
                nopol_probs = cast(torch.Tensor, nopol.probs)
                self._log('loss/nb_prob_act_u0.01', (nopol_probs < 0.01).sum() / self.cfg.batch_size, int(self.cfg.udr * (self.total_trans - self.cfg.trans_per_iter)) + grad_steps)
                self._log('loss/max_q_ratio', max_q_ratio, int(self.cfg.udr * (self.total_trans - self.cfg.trans_per_iter)) + grad_steps)

                max_q_ratio = 0

            self.q_optim.step()
            grad_steps += 1

    def _update_staq_networks(self, old_logits_tilde, testb, old_dist):
        [qfunc.train(False) for qfunc in self.qfuncs]
        with torch.no_grad():
            if self.cfg.w_correction:
                old_logits_tilde += (sum([qfunc(testb.obs) for qfunc in self.qfuncs]) / self.cfg.n_ensemble) * self.qfuncs[0].eta * self.qfuncs[0].decay ** (self.cfg.memory_size - 1) / (
                            1 - self.qfuncs[0].decay ** self.cfg.memory_size)
            [qfunc.update_sigq() for qfunc in self.qfuncs]
            sumqnext = self.get_logits_ensemble_torch(self.qfuncs, testb.obs)

            training_time_elapsed = time() - self.training_start_time

            # logging
            # doesn't take into account greedy
            self._log('log/kl_pk_pkpone', stable_kl_div(old_dist.probs, torch.softmax(sumqnext, dim=1)).mean(), self.total_trans // self.cfg.trans_per_iter)
            self._log('log/kl_pk_pktilde', stable_kl_div(old_dist.probs, torch.softmax(old_logits_tilde, dim=1)).mean(), self.total_trans // self.cfg.trans_per_iter)
        return training_time_elapsed


            
    def _precompute_logits(self):
        precompute_start_time = time()
        with torch.no_grad():
            nologits = torch.zeros(self.repmem.size, self.n_act, device=self.device)
            sid = 0
            for k in range(min(200, self.repmem.size), self.repmem.size + 1, 200):
                nologits[sid:k] = self.get_logits_ensemble_torch(self.qfuncs, self.repmem.repmem.nobs[sid:k].to(torch.float32))
                sid = k
            if self.repmem.size > sid:
                nologits[sid:self.repmem.size] = self.get_logits_ensemble_torch(self.qfuncs, self.repmem.repmem.nobs[sid:self.repmem.size].to(torch.float32))
        precompute_time_elapsed = time() - precompute_start_time
        return nologits, precompute_time_elapsed


    def _policy_snapshot(self):
        testb = self.repmem.sample(200, device=self.device)  # for logging
        with torch.no_grad():
            old_dist = torch.distributions.Categorical(logits=self.get_logits_ensemble_torch(self.qfuncs, testb.obs))
            old_logits_tilde = self.get_logits_ensemble_torch(self.qfuncs, testb.obs, no_old=True)
            pol_entropy = old_dist.entropy().mean()
            self._log('log/entropy', pol_entropy, self.total_trans // self.cfg.trans_per_iter)
        return old_logits_tilde, testb, old_dist
            
    def _update_schedules(self):
        self.eweight = self.entropy_weight_function(self.total_trans)
        [qfunc.set_entropy_weight(self.eweight) for qfunc in self.qfuncs]
        self._log('pars/entrop_weight', self.eweight, self.total_trans)
        [qfunc.train(False) for qfunc in self.qfuncs]
        self.eps = self.epsilon(self.total_trans)
        self._log('pars/epsilon', self.eps, self.total_trans)
        
    def _collect_rollouts(self):
        rollout_start_time = time()
        new_trans, returns, entropies = self.sampler.rollouts(self.numpy_egreedy_softpolicy, min_trans=self.cfg.trans_per_iter, max_trans=self.cfg.trans_per_iter)
        # logging
        for ret, entr in zip(returns, entropies):
            self.latest_train_return = ret.value
            self._log('train/return', ret.value, ret.global_step)
            self._log('train/return_n_entropy', ret.value + self.eweight * entr, ret.global_step)

        self.repmem.add_trans(new_trans)
        self.total_trans += self.cfg.trans_per_iter

        rollout_time_elapsed = time() - rollout_start_time
        self._log('timings/rollout', rollout_time_elapsed, self.total_trans)

    def entropy_weight_function(self, t:int) -> float:
        return linear_schedule(t, self.cfg.init_ew, self.cfg.final_ew, self.cfg.end_decay) / np.log(self.n_act)

    def epsilon(self, t:int) -> float:
        return linear_schedule(t, self.cfg.init_eps, self.cfg.final_eps, self.cfg.end_decay)

    def _ensemble_dist(self, obs)-> torch.distributions.Categorical:
        obs = torch.tensor(obs[None, :].astype(np.float32), device=self.device)
        with torch.no_grad():
            return torch.distributions.Categorical(logits=self.get_logits_ensemble_torch(self.qfuncs, obs))

    def _log(self, tag, value, step=None):
        self.logger.add_scalar(tag, value, self.total_trans if step is None else step)

    def numpy_softmax(self, obs):
        dist = self._ensemble_dist(obs)
        return _finish_action(dist.sample().squeeze()), _finish_action(dist.entropy().squeeze())

    def numpy_argmax_policy(self, obs):
        dist = self._ensemble_dist(obs)
        probs = cast(torch.Tensor, dist.probs)
        return _finish_action(probs.argmax(1).squeeze()), _finish_action(dist.entropy().squeeze())

    def numpy_egreedy_softpolicy(self, obs):
        with torch.no_grad():
            act, entrop = self.numpy_softmax(obs)
        if np.random.rand() < self.eps:
            act = np.random.randint(self.qfuncs[0].output_size)
        return act, entrop

    def get_logits_ensemble_torch(self,qfuncs, obs, no_old=False):
        return sum([qfunc.get_logits(obs, no_old) for qfunc in qfuncs]) / len(qfuncs)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    run_cfg = cast(AppConfig, OmegaConf.to_object(cfg))

    if run_cfg.wandb:
        import wandb

        wandb.init(
            project=run_cfg.wandb_project,
            # to_container needs the DictConfig, not the dataclass instance
            config=cast(dict, OmegaConf.to_container(cfg, resolve=True)),
            sync_tensorboard=True,
            name=join(run_cfg.env_name, str(run_cfg.seed)),
            monitor_gym=True,
        )
    logging_path = HydraConfig.get().runtime.output_dir
    StaQTrainer(run_cfg, logging_path).run()


if __name__ == "__main__":
    main()