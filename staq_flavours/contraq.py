"""ContraQ: continuous-action DistraQ (Design A).

Per iteration k, in this order -- the order is forced, each step needs the previous one:

    collect          with pi_phi
    _train           Q_k, soft policy evaluation of the frozen pi_{k-1}
    _learn_student   S_k <- lambda * S_{k-1} + Q_k        (scalar regression, no normaliser)
    _learn_actor     pi_phi <- argmin KL(pi || exp(eta * S_k))

The actor is an amortised sampler for exp(eta*S), not a second policy: the memory lives in
S, outside the parametric family, so the actor's projection error is refreshed every
iteration instead of compounding. Setting anchor='policy' replaces S by the previous actor
in the target and recovers MDPO -- the ablation that tests whether the sum earns its place.

Does not subclass StaQTrainer: 14 of its 23 methods assume enumerable actions (the
Categorical policies, _precompute_logits, ew/log(n_act)), and its __init__ asserts a
Discrete action space. Shared infrastructure (Sampler, ReplayMemory, rl_tools) is reused.
"""
from copy import deepcopy
from time import time
from typing import cast

import hydra
import numpy as np
import torch
from gymnasium.spaces import Box
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src.networks.contraq_net import Actor, Critic
from src.utils.replay_memory import ReplayMemory
from src.utils.rl_tools import linear_schedule, make_continuous_envs, update_target
from src.utils.sampler import Sampler
from staq_flavours.contraq_config import ContraQConfig


class ContraQTrainer:
    def __init__(self, cfg: ContraQConfig, logging_path: str):
        self.cfg = cfg
        assert cfg.target_type in ['hard', 'soft']
        assert cfg.mode in ['mean', 'min']
        assert cfg.anchor in ['sum', 'policy']
        torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.env, self.env_eval, self.obs_type = make_continuous_envs(cfg.env_name)
        act_space = self.env.action_space
        assert isinstance(act_space, Box) and act_space.shape is not None
        assert self.env.observation_space.shape is not None
        self.s_dim = self.env.observation_space.shape[0]
        self.a_dim = act_space.shape[0]
        self.sampler = Sampler(self.env)
        self.sampler_eval = Sampler(self.env_eval)

        print('logging path', logging_path)
        self.logger = SummaryWriter(logging_path)

        self.actor = Actor(cfg.actor, self.s_dim, self.a_dim,
                           float(act_space.low[0]), float(act_space.high[0]),
                           device=self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=cfg.actor.lr)
        self.critics = [Critic(cfg.critic, cfg.student, self.s_dim, self.a_dim,
                               cfg.kl_weight, cfg.init_ew, device=self.device)
                        for _ in range(cfg.n_ensemble)]
        self.q_optim = Adam([p for c in self.critics for p in c.q.parameters()], lr=cfg.lr)
        self.student_optim = Adam([p for c in self.critics for p in c.student.parameters()],
                                  lr=cfg.student.lr)
        self.qtars = update_target(self.critics, update_type='hard')

        self.repmem = ReplayMemory(cfg.rep_mem_size, self.s_dim, self.device,
                                   obs_type=self.obs_type, action_dim=self.a_dim,
                                   action_dtype=torch.float32)
        self.total_trans = 0
        self.total_time_elapsed = 0.0
        self.latest_train_return = None
        self.latest_eval_return = None
        self.progress_bar = tqdm(total=cfg.timesteps, unit='steps')
        self._update_schedules()   # eweight/eta must exist before any component is used

    # ---------------------------------------------------------------- main loop
    def run(self):
        while self.total_trans < self.cfg.timesteps:
            t0 = time()
            self._update_schedules()
            self._collect_rollouts()
            self._train()
            self._learn_student()
            self._learn_actor()
            if self.total_trans % self.cfg.eval_interval == 0:
                self.evaluate_policy()
            self.total_time_elapsed += time() - t0
            self._log('timings/total', self.total_time_elapsed)
            self.progress_bar.set_postfix({'train/return': self.latest_train_return,
                                           'eval/return': self.latest_eval_return}, refresh=False)
            self.progress_bar.update(self.cfg.trans_per_iter)
        self.progress_bar.close()
        self.logger.close()

    def _update_schedules(self):
        # No /log(n_act) here: it has no continuous analogue (differential entropy is
        # unbounded below). SAC's convention would be a target entropy of -a_dim.
        self.eweight = linear_schedule(self.total_trans, self.cfg.init_ew, self.cfg.final_ew,
                                       self.cfg.end_decay)
        for c in self.critics:
            c.set_entropy_weight(self.cfg.kl_weight, self.eweight)
        self.eta = self.critics[0].eta
        self._log('pars/entrop_weight', self.eweight)
        self._log('pars/eta', self.eta)

    def _collect_rollouts(self):
        t0 = time()
        new_trans, returns, _ = self.sampler.rollouts(
            self.numpy_policy, min_trans=self.cfg.trans_per_iter, max_trans=self.cfg.trans_per_iter)
        for ret in returns:
            self.latest_train_return = ret.value
            self._log('train/return', ret.value, ret.global_step)
        self.repmem.add_trans(new_trans)
        self.total_trans += self.cfg.trans_per_iter
        self._log('timings/rollout', time() - t0)

    # ---------------------------------------------------------------- critic
    def _train(self):
        t0 = time()
        self.actor.train(False)  # frozen during evaluation: Q_k is the soft Q of pi_{k-1}
        grad_steps = 0
        while grad_steps < int(self.cfg.udr * self.cfg.trans_per_iter):
            if self.cfg.target_type == 'hard' and grad_steps % self.cfg.hard_target_steps == 0:
                self.qtars = update_target(self.critics, update_type='hard')
            elif self.cfg.target_type == 'soft':
                self.qtars = update_target(self.critics, self.qtars, self.cfg.soft_target_polyak,
                                           update_type='soft')
            db = self.repmem.sample(self.cfg.batch_size, device=self.device)
            with torch.no_grad():
                nact, nlogp, _ = self.actor.get_action(db.nobs)
                qnext = torch.cat([tar(db.nobs, nact) for tar in self.qtars], dim=-1)
                qnext = qnext.mean(-1, keepdim=True) if self.cfg.mode == 'mean' else qnext.min(-1, keepdim=True).values
                # -eweight * log pi is the entropy bonus; the critic's coefficient is ew,
                # NOT the actor's 1/eta = kl_weight + ew. They coincide only at kl_weight=0.
                targ = self.cfg.rwd_scale * db.rwd + self.cfg.gamma * (1 - db.terminated) * (
                    qnext - self.eweight * nlogp)
            lossq = torch.stack([(c(db.obs, db.act) - targ).pow(2).mean() for c in self.critics]).mean()
            self.q_optim.zero_grad()
            lossq.backward()
            self.q_optim.step()
            if grad_steps % 100 == 0:
                self._log('loss/bellerror', lossq.item())
            grad_steps += 1
        self._log('timings/training', time() - t0)

    # ---------------------------------------------------------------- student
    def _distil_actions(self, obs):
        """Buffer actions for coverage, actor samples where the actor will actually query S."""
        n_actor = int(len(obs) * self.cfg.student.actor_action_frac)
        acts = self.repmem.sample(len(obs), device=self.device).act
        if n_actor:
            with torch.no_grad():
                acts[:n_actor] = self.actor.get_action(obs[:n_actor])[0]
        return acts

    def _learn_student(self):
        t0 = time()
        n = min(self.cfg.student.distil_states, self.repmem.size)
        batch = self.repmem.sample(n, device=self.device)   # ONE sample: obs and act must correspond
        obs, act = batch.obs, self._distil_actions(batch.obs)
        residuals = []
        for c in self.critics:
            with torch.no_grad():
                target = c.decay * c.forward_student(obs, act) + c(obs, act)
            for _ in range(self.cfg.student.steps):
                idx = torch.randint(0, len(obs), (self.cfg.student.batch_size,), device=self.device)
                pred = c.forward_student(obs[idx], act[idx]) * c.eta
                loss = self._student_loss(pred, target[idx] * c.eta, obs[idx])
                self.student_optim.zero_grad()
                loss.backward()
                self.student_optim.step()
            with torch.no_grad():
                residuals.append(((c.forward_student(obs, act) - target) * c.eta).abs().mean().item())
        self._log('distil/abs_residual', float(np.mean(residuals)))
        self._log('timings/distil', time() - t0)

    def _student_loss(self, pred, target, obs):
        """MSE on eta-scaled sums. S only matters up to a per-state constant, so with
        `centered` the shared offset is removed -- the continuous analogue of centered_mse,
        except the mean is over the sampled actions in the batch rather than over all actions."""
        d = pred - target
        if self.cfg.student.centered:
            d = d - d.mean()
        return d.pow(2).mean()

    # ---------------------------------------------------------------- actor
    def _energy(self, s, a):
        """min over members: this is the point of use, where pessimism stops the actor
        exploiting the students' errors off the data distribution."""
        e = torch.cat([c.forward_student(s, a) for c in self.critics], dim=-1)
        return e.mean(-1, keepdim=True) if self.cfg.mode == 'mean' else e.min(-1, keepdim=True).values

    def _learn_actor(self):
        t0 = time()
        self.actor.train(True)
        anchor = deepcopy(self.actor).eval() if self.cfg.anchor == 'policy' else None
        for p in [p for c in self.critics for p in list(c.q.parameters()) + list(c.student.parameters())]:
            p.requires_grad_(False)   # freeze params, do NOT detach the energy tensor
        losses, entropies = [], []
        for _ in range(self.cfg.actor.steps):
            s = self.repmem.sample(self.cfg.actor.batch_size, device=self.device).obs
            if self.cfg.actor.action_samples > 1:
                s = s.repeat_interleave(self.cfg.actor.action_samples, dim=0)
            a, logp, _ = self.actor.get_action(s)
            if anchor is None:                      # Design A: energy is the distilled sum
                loss = (logp - self.eta * self._energy(s, a)).mean()
            else:                                   # Design B / MDPO: anchor on the previous actor
                lam = self.critics[0].decay
                with torch.no_grad():
                    anchor_logp = self._anchor_logp(anchor, s, a)
                q = torch.cat([c(s, a) for c in self.critics], dim=-1)
                q = q.mean(-1, keepdim=True) if self.cfg.mode == 'mean' else q.min(-1, keepdim=True).values
                loss = (logp - lam * anchor_logp - self.eta * q).mean()
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            losses.append(loss.item())
            entropies.append(-logp.mean().item())
        for p in [p for c in self.critics for p in list(c.q.parameters()) + list(c.student.parameters())]:
            p.requires_grad_(True)
        self.actor.train(False)
        if losses:  # actor.steps may be 0 in an ablation
            self._log('actor/loss', float(np.mean(losses)))
            self._log('actor/entropy', float(np.mean(entropies)))
        self._log('timings/actor', time() - t0)

    @staticmethod
    def _anchor_logp(anchor, s, a):
        """log pi_anchor(a|s) for actions sampled from the CURRENT actor (inverse tanh)."""
        mean, log_std = anchor(s)
        y = ((a - anchor.action_bias) / anchor.action_scale).clamp(-0.999999, 0.999999)
        x = torch.atanh(y)
        normal = torch.distributions.Normal(mean, log_std.exp())
        lp = normal.log_prob(x) - torch.log(anchor.action_scale * (1 - y.pow(2)) + 1e-6)
        return lp.sum(-1, keepdim=True)

    # ---------------------------------------------------------------- policies / eval
    def numpy_policy(self, obs):
        with torch.no_grad():
            s = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None, :]
            a, logp, _ = self.actor.get_action(s)
        return a[0].cpu().numpy(), -logp.item()

    def numpy_det_policy(self, obs):
        with torch.no_grad():
            s = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None, :]
            _, logp, mean = self.actor.get_action(s)
        return mean[0].cpu().numpy(), -logp.item()

    def evaluate_policy(self):
        t0 = time()
        rollouts = [self.sampler_eval.rollouts(self.numpy_det_policy, 1, np.inf, returns_only=True)
                    for _ in range(self.cfg.n_eval_episodes)]
        self.latest_eval_return = float(np.mean([r[1][0].value for r in rollouts]))
        self._log('eval/return', self.latest_eval_return)
        self._log('eval/mean_ep_length', float(np.mean([r[1][0].step for r in rollouts])))
        self._log('timings/eval', time() - t0)

    def _log(self, tag, value, step=None):
        self.logger.add_scalar(tag, value, self.total_trans if step is None else step)


@hydra.main(version_base=None, config_path="../conf", config_name="contraq")
def main(cfg: DictConfig):
    run_cfg = cast(ContraQConfig, OmegaConf.to_object(cfg))
    if run_cfg.wandb:
        import wandb
        wandb.init(project=run_cfg.wandb_project, name=run_cfg.run_name,
                   config=cast(dict, OmegaConf.to_container(cfg, resolve=True)),
                   sync_tensorboard=True, group=run_cfg.run_name.rsplit('_s', 1)[0])
    ContraQTrainer(run_cfg, HydraConfig.get().runtime.output_dir).run()


if __name__ == "__main__":
    main()
