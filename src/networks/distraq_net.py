import torch
from torch import nn

from src.networks.staq_net import StaQNet
from src.utils.rl_tools import zero_linear


class DistraQNet(StaQNet):
    def __init__(self, input_size, output_size, nb_hidden, hidden_width, memory_size, kl_weight, entropy_weight, use_w_correction, cfg_student, device=None, nl=None, network_type="mlp", cnn_config: dict|None=None):

        super().__init__(input_size, output_size, nb_hidden, hidden_width, memory_size, kl_weight, entropy_weight, use_w_correction, device=device, nl=nl, network_type=network_type, cnn_config=cnn_config)
        self.student = self._init_student(cfg_student)
        self.cfg_student = cfg_student

    def _init_student(self, cfg):
        # mlp and cnn differ only in what precedes the hidden Linears and in the
        # width they start from.
        if self.network_type == "mlp":
            layers, insize = [], self.input_size
        else:
            assert self.network_config is not None, "CNN configuration must be provided for CNN network type."
            in_channels, H, W = self.input_size
            layers = []

            # Build CNN from config; cfg fields default to the teacher's trunk.
            current_channels = in_channels
            for out_channels, ker_size, stride in zip(cfg.channels or self.network_config['channels'],
                                                      cfg.kernel_size or self.network_config['kernel_size'],
                                                      cfg.strides or self.network_config['strides']):
                layers.extend([
                    nn.Conv2d(current_channels, out_channels,
                              kernel_size=ker_size, stride=stride),
                    self.nl
                ])
                current_channels = out_channels

            # Flatten and get output size
            layers.append(nn.Flatten())
            cnn = nn.Sequential(*layers).to(self.device)

            with torch.no_grad():
                sample_input = torch.zeros(1, in_channels, H, W).to(self.device)
                insize = cnn(sample_input).shape[1]

        for _ in range(cfg.depth):
            layers.append(nn.Linear(insize, cfg.width))
            layers.append(self.nl)
            insize = cfg.width

        layers.append(zero_linear(nn.Linear(cfg.width, self.output_size).to(self.device)))

        return nn.Sequential(*layers).to(self.device)
   
    def get_logits(self, x, no_old=False):
        """no_old is purely a placeholder from inheritance here"""
        return self.student(x) * self.eta * self.w_correction 

    def update_sigq(self, decay=True):
        # Update the frozen features and frozen Q function
        self.train_feat.train(False)



