from collections import namedtuple

import numpy as np
import torch


class Sampler:
    # Sample transitions from gymnasium type environments
    def __init__(self, env, device='cpu'):
        self.curr_rollout = []
        self.policy = None
        self.env = env
        self.Return = namedtuple('Return', 'global_step value step')
        self.max_abs_reward = 0
        self.curr_return = 0
        self.curr_entropy = 0.
        self.total_step = 0
        self.Trans = namedtuple('Trans', 'obs act rwd terminated truncated nobs nact')  # order must match yield below
        self.device = device

    def _rollout(self):
        # Generates SARSA type transitions until episode's end
        assert self.policy is not None, "Policy must be set before calling _rollout."
        observation, _ = self.env.reset()
        action, entropy = self.policy(observation)
        done = False
        step = 0
        while not done:
            step += 1
            next_observation, rwd, terminated, truncated, _ = self.env.step(action)
            self.max_abs_reward = max(self.max_abs_reward, np.abs(rwd))
            next_action, next_entropy = self.policy(next_observation)
            yield self.Trans(observation, action, rwd, terminated, truncated, next_observation, next_action), entropy, step
            observation = next_observation
            action = next_action
            entropy = next_entropy
            done = terminated or truncated

    def rollouts(self, policy, min_trans, max_trans, returns_only=False):
        # Keep generating full trajectories until min_trans transitions are collected.
        # Specifying max_trans < inf can stop data collection before trajectory's end.
        # If min_trans = max_trans, will collect exactly min_trans transitions.
        assert (min_trans <= max_trans)
        returns = []
        entropies = []
        self.policy = policy
        all_trans = []
        # Generating transitions and computing returns
        gathered_steps = 0
        while gathered_steps < min_trans:
            for trans, entr, step in self.curr_rollout:
                if not returns_only:
                    all_trans.append(trans)
                self.curr_return += trans.rwd
                self.curr_entropy += entr
                self.total_step += 1
                gathered_steps += 1
                if trans.truncated or trans.terminated:
                    returns.append(self.Return(self.total_step, self.curr_return, step))
                    entropies.append(self.curr_entropy)
                    self.curr_return = 0
                    self.curr_entropy = 0
                if gathered_steps >= max_trans:
                    break
            if not gathered_steps >= max_trans:
                self.curr_rollout = self._rollout()

        # Saving into a dictionary of 2D torch.FloatTensor
        if returns_only:
            return None, returns, entropies
        else:
            paths = {}
            for key in set(self.Trans._fields):
                paths[key] = torch.tensor(np.asarray([getattr(t, key) for t in all_trans]), device=self.device, dtype=torch.float)
                if paths[key].ndim == 1:
                    paths[key] = paths[key][:, None]
            return paths, returns, entropies