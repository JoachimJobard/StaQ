from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import II

from src.config import AppConfig


@dataclass
class RankScheduleConfig:
    _target_: str = "src.utils.rank_schedule.constant_rank.ConstantRankSchedule"
    # Follows lora.max_rank so a sweep only has to set one key. Override
    # explicitly if a strategy should request less than the allocated capacity.
    max_rank: int = II("lora.max_rank")
    low_bound_lora: int = 150
    high_bound_lora: int = 500
    low_freq: int = 5
    high_freq: int = 10

@dataclass
class LoRAConfig:
    max_rank: int = 128
    # Follows max_rank so the adapter scale (alpha / r) stays 1 across a rank
    # sweep. A fixed alpha would make the adapter's effective step size vary
    # with r, confounding the very axis being swept.
    alpha: float = II("lora.max_rank")
    lora_layers_index: list[int] = field(default_factory=lambda: [1])
    schedule: RankScheduleConfig = field(default_factory=RankScheduleConfig)
    warm_start: str = "merged"  # Options: "merged", "keep_base"
    freeze_non_lora: bool = True  # Whether to freeze non-LoRA parameters during training

@dataclass
class StoRAQConfig(AppConfig):
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    # Identifies the experimental condition. Used for the wandb run name and for
    # the hydra output/sweep subdirectory, so a multirun is navigable by condition.
    run_name: str = ("${env_name}_r${lora.max_rank}_ws${lora.warm_start}"
                     "_frz${lora.freeze_non_lora}_s${seed}")

cs = ConfigStore.instance()
cs.store(name="storaq_config", node=StoRAQConfig)
