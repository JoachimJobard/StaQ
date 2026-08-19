from torch import nn

from src.networks.staq_net import StaQNet
from src.utils.rl_tools import zero_linear


class DistraQNet(StaQNet):
    def __init__(self, input_size, output_size, nb_hidden, hidden_width, memory_size, kl_weight, entropy_weight, use_w_correction, cfg_student, device=None, nl=None, network_type="mlp", cnn_config: dict | None= None):

        super().__init__(input_size, output_size, nb_hidden, hidden_width, memory_size, kl_weight, entropy_weight, use_w_correction, device=device, nl=nl, network_type=network_type, cnn_config=cnn_config)
        self.student = self._init_student(cfg_student)
        self.cfg_student = cfg_student

    def _init_student(self, cfg):
        layers = []
        layers.append(nn.Linear(self.input_size, cfg.width))
        layers.append(nn.ReLU())
        
        for _ in range(cfg.depth - 1):
            layers.append(nn.Linear(cfg.width, cfg.width))
            layers.append(nn.ReLU())
        
        layers.append(zero_linear(nn.Linear(cfg.width, self.output_size)))
        
        return nn.Sequential(*layers).to(self.device)
   
    def get_logits(self, x, no_old=False):
        """no_old is purely a placeholder from inheritance here"""
        return self.student(x) * self.eta * self.w_correction 

    def update_sigq(self, decay=True):
        # Update the frozen features and frozen Q function
        self.train_feat.train(False)



