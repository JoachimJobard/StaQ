from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from staq_flavours.storaq_config import StoRAQConfig


@dataclass
class ShadowStoRAQConfig(StoRAQConfig):
    # freeze_non_lora and warm_start are absent from the name on purpose: in this
    # ablation the whole trunk is owned by the shadow, so neither knob applies.
    run_name: str = "${env_name}_shadow_r${lora.max_rank}_s${seed}"


cs = ConfigStore.instance()
cs.store(name="storaq_ablation_config", node=ShadowStoRAQConfig)
