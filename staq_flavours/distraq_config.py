from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from src.config import AppConfig


@dataclass
class StudentConfig:
    depth: int = 2
    distil_steps: int = 200 # 5_000 is the original budget
    lr: float = 3e-4
    width: int = 256
    distil_states: int = 10_000
    batch_size: int = 256
    channels: list[int] | None = None # For CNN and minatar, none = reuse teacher config
    kernel_size:list[int] | None = None
    strides:list[int] | None = None
    loss:str = "kl" #options are kl, mse, centered_mse

@dataclass
class DistraQConfig(AppConfig):
    student: StudentConfig = field(default_factory=StudentConfig)
    run_name: str = ("${env_name}_d${student.depth}_w${student.width}"
                     "_dsteps${student.distil_steps}_s${seed}_loss{student.loss}")

cs = ConfigStore.instance()
cs.store(name="distraq_config", node=DistraQConfig)
