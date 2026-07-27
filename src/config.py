import time
from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class RunConfig:
    """Configuration for StaQ."""
    # ---------- Logging & hardware ----------
    # NB: the log directory is supplied by the caller (Hydra's runtime.output_dir),
    # not configured here.
    device:str='cuda'

    # ---------- Environment ----------
    env_name: str = 'CartPole-v1'
    seed: int = 42
    rwd_scale:float=10.

    # --------- Network Architecture ----------
    network_type:str='mlp'
    lr:float=1e-4
    batch_size:int=256
    soft_target_polyak:float=0.005
    hidden_width:int=256
    nb_hidden:int=2

    # --------- StaQ Memory ----------
    memory_size:int=300

    # --------- Training ----------
    timesteps: int = 5_000_000
    trans_per_iter: int = 5_000
    n_ensemble:int = 2
    mode:str='mean'
    kl_weight:float=20.
    init_ew:float=2.0
    final_ew:float=0.4
    target_type:str='hard'
    hard_target_steps:int=200
    init_eps:float=0.05
    final_eps:float=0.05
    end_decay:int=500_000
    l2_weight:float=0.0
    udr:float=1.
    w_correction:bool=True
    gamma:float=0.99

    # ---------- Replay Memory ----------
    rep_mem_size:int=50_000

    # ---------- Evaluation ----------
    n_eval_episodes:int=10
    eval_interval:int=100_000

@dataclass
class AppConfig(RunConfig):
    wandb: bool = True
    wandb_project: str = "StaQ"

cs = ConfigStore.instance()
cs.store(name="base_config", node=AppConfig)
