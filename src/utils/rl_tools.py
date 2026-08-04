from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers.time_limit import TimeLimit


def _finish_action(tensor: torch.Tensor, device='cpu'):
    """Converts a tensor to a numpy array and moves it to the specified device."""
    return tensor.detach().cpu().numpy() if device == 'cpu' else tensor.detach().to(device).numpy()

def linear_schedule(t, start, end, duration):
    frac = min(t/duration, 1.0)
    return start + frac * (end - start)

def update_target(sources, targets: list|None=None, tau: float|None=None, update_type=None):
    if update_type == 'hard':
        targets = [deepcopy(source) for source in sources]
    elif update_type == 'soft':
        assert targets is not None, "Targets must be provided for soft updates."
        assert tau is not None, "Tau must be provided for soft updates."
        for source, target in zip(sources, targets):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    else:
        raise ValueError(f'Update type must be "hard" or "soft", not "{update_type}"')

    [target.train(False) for target in targets]
    return targets

def make_envs(env_name)->tuple[gym.Env, gym.Env, torch.dtype]:
    obs_type = torch.float32
    if env_name.startswith('MinAtar'):
        env = ChannelsFirst(gym.make(env_name))
        env_eval = ChannelsFirst(gym.make(env_name))
        env = TimeLimit(env, max_episode_steps=5000)
        env_eval = TimeLimit(env_eval, max_episode_steps=5000)
    elif env_name in ['Hopper-v4', 'Ant-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Humanoid-v4']:
        env = gym.make(env_name, render_mode='rgb_array')
        env_eval = gym.make(env_name, render_mode='rgb_array')
        env = ContToDiscreteActWrap(env)
        env_eval = ContToDiscreteActWrap(env_eval)
    else:
        env = gym.make(env_name)
        env_eval = gym.make(env_name)

    return env, env_eval, obs_type

def make_network_type(network_type:str, env_name:str):
    #old hardcoded network type selection, could be improved to be more flexible
    if network_type == 'mlp':
        cnn_config = None
    elif network_type == 'cnn':
        if env_name.startswith('MinAtar'):
            cnn_config = {'channels': [16],
                          'kernel_size': [3],
                          'stride': [1]}
        else:
            raise NotImplementedError(f"cnn network type not implemented for env {env_name}")
    else:
        raise NotImplementedError(f"network type {network_type} not implemented")
    return cnn_config

def zero_linear(f):
    f.weight.data.zero_()
    if f.bias is not None:
        f.bias.data.zero_()
    return f

def stable_kl_div(old_probs, new_probs, epsilon=1e-12, lib=torch):
    kl = new_probs * (lib.log(new_probs + epsilon) - lib.log(old_probs + epsilon))
    return kl


class ChannelsFirst(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = env.observation_space.shape
        assert shape is not None and len(shape) == 3, "ChannelsFirst wrapper expects an observation space with 3 dimensions (H, W, C)."
        self.observation_space = gym.spaces.Box(0, 255, (shape[2], shape[0], shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return observation.swapaxes(0, 2)

class ContToDiscreteActWrap(gym.ActionWrapper):
    def __init__(self, env:gym.Env):
        super().__init__(env)
        self.neutral_action = (env.action_space.low + env.action_space.high) / 2 # type: ignore
        self.dim_base = len(self.neutral_action)
        self.action_space = gym.spaces.Discrete(2 * self.dim_base + 1)

    def action(self, a):
        act = self.neutral_action.copy()
        if a:
            ind = (a - 1) % self.dim_base
            act[ind] = self.env.action_space.low[ind] if a <= self.dim_base else self.env.action_space.high[ind] # type: ignore
        return act