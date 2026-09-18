from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from src.config import AppConfig


@dataclass
class NetConfig:
    depth: int = 2
    width: int = 256
    lr: float = 3e-4
    nl: str | None = None  # None -> ReLU


@dataclass
class ActorConfig(NetConfig):
    steps: int = 200          # projection steps per iteration; warm-started, so few are needed
    batch_size: int = 256
    action_samples: int = 1   # actions drawn per state; >1 cuts the reverse-KL gradient variance


@dataclass
class StudentConfig(NetConfig):
    steps: int = 1000
    batch_size: int = 256
    distil_states: int = 10_000
    # Fraction of distillation actions resampled from the current actor. The actor queries
    # S at a ~ pi, so training only on buffer actions leaves S unconstrained exactly where
    # it will be maximised -- the off-support exploitation failure mode.
    actor_action_frac: float = 0.5
    centered: bool = True     # S matters only up to a per-state constant; centre over sampled actions


@dataclass
class ContraQConfig(AppConfig):
    critic: NetConfig = field(default_factory=NetConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    # 'sum' = Design A (anchor is the distilled sum). 'policy' = Design B (anchor is the
    # previous actor) == MDPO, the ablation that shows whether the sum earns its place.
    anchor: str = "sum"
    # kl_weight is in the name: kl_weight=0 collapses to SAC for BOTH anchors, so without
    # it the SAC ablation would share a run name (and a wandb group) with anchor=sum.
    run_name: str = ("${env_name}_contraq_${anchor}_kl${kl_weight}"
                     "_w${student.width}_ssteps${student.steps}_s${seed}")


cs = ConfigStore.instance()
cs.store(name="contraq_config", node=ContraQConfig)
