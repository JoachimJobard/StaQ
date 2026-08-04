from torch import nn

from src.networks.lora_adapter import LoRALinear
from src.networks.staq_net import StaQNet
from src.utils.rl_tools import zero_linear


class SToRAQNet(StaQNet):
    def __init__(self, input_size, output_size, nb_hidden, hidden_width, memory_size, kl_weight, entropy_weight, max_rank, lora_layers_index:list[int],alpha, freeze_non_lora, use_w_correction, warm_start:str="merged", device=None, nl=None, network_type="mlp", cnn_config: dict | None= None):
        self.max_rank = max_rank
        self.lora_layers_index = lora_layers_index
        self.alpha = alpha
        self.warm_start = warm_start
        self.freeze_non_lora = freeze_non_lora
        super().__init__(input_size, output_size, nb_hidden, hidden_width, memory_size, kl_weight, entropy_weight,
                         use_w_correction, device=device, nl=nl, network_type=network_type, cnn_config=cnn_config)
        

    def _get_networks(self):
        if self.network_type == "mlp":
            insize = self.input_size
            ops = []
            for i in range(self.nb_hidden):
                if i in self.lora_layers_index:
                    linear = nn.Linear(insize, self.hidden_width).to(self.device)
                    lora_linear = LoRALinear(linear, self.max_rank,warm_start=self.warm_start, alpha=self.alpha).to(self.device)
                    ops.append(lora_linear)
                    ops.append(self.nl)
                else:
                    ops.append(nn.Linear(insize, self.hidden_width))
                    ops.append(self.nl)
                insize = self.hidden_width
            output_layer = zero_linear(nn.Linear(self.hidden_width, self.output_size).to(self.device))
            return nn.Sequential(*ops).to(self.device), output_layer
        else:
            raise NotImplementedError("CNN network type is not implemented for SToRAQNet.")

    def _feat_for_archive(self): # Merge the LoRA parameters for saving
        return nn.Sequential(*[layer.merged() if isinstance(layer, LoRALinear) else layer for layer in self.train_feat]).to(self.device)

    def lora_layers(self) -> list[LoRALinear]:
        return [layer for layer in self.train_feat if isinstance(layer, LoRALinear)]

    def reset_lora_parameters(self):
        for layer in self.train_feat:
            if isinstance(layer, LoRALinear):
                layer.reset_lora_parameters()

    def rebaseline(self):
        return [p for layer in self.train_feat if isinstance(layer, LoRALinear) for p in layer.rebaseline()]

    def set_phase(self, full: bool, r: int):
        for layer in self.train_feat:
            if isinstance(layer, LoRALinear):
                # depending the policy, update the full layer or only the LoRA parameters
                layer.linear.requires_grad_(full)
                layer.lora_A.requires_grad = not full
                layer.lora_B.requires_grad = not full
                if full:
                    # Only re-rank at a group boundary, where B == 0 so the change
                    # in both the term count and the alpha/r scale is a no-op.
                    layer.set_rank(r)
            elif self.freeze_non_lora:
                layer.requires_grad_(full)
