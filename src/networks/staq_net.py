import torch
from torch import nn

from networks.stacked_nn import StackedNN
from utils.rl_tools import zero_linear


class StaQNet:
    def __init__(self, input_size, output_size, nb_hidden, hidden_width, memory_size, kl_weight, entropy_weight, use_w_correction,
                 device=None, nl=None, network_type="mlp", cnn_config: dict | None= None):
        super().__init__()

        assert nb_hidden > 0, "Number of hidden layers must be greater than 0."
        self.nb_hidden = nb_hidden  # total number of hidden layers (remains fixed)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_width = hidden_width
        self.memory_size = memory_size
        self._kl_weight = kl_weight
        self._entropy_weight = entropy_weight
        self.eta = 1. / (kl_weight + entropy_weight)
        self.decay = kl_weight / (kl_weight + entropy_weight)
        self.use_w_correction = use_w_correction
        self.strides = None
        if device is None:
            device = torch.device('cpu')
        if nl is None:
            nl = nn.ReLU(inplace=True)
        if self.use_w_correction:
            self.w_correction = 1. / (1. - self.decay ** self.memory_size)
        else:
            self.w_correction = 1.

        print(f'step size eta {self.eta} and decay {self.decay}')
        self.device = device
        self.nl = nl

        self.network_type = network_type
        self.network_config = cnn_config

        # Frozen net
        self.froz_feat = None
        # self.froz_q = None
        self.sig_q = StackedNN([zero_linear(nn.Linear(hidden_width, output_size).to(self.device))], self.memory_size)  # frozen Q

        # Trainable mlp
        self.train_feat, self.train_q = self._get_networks()

    def __call__(self, x):
        return self.train_q(self.train_feat(x))  # returns new Q

    def get_logits(self, x, no_old=False):  # get logits, that depend only on old features
        # no_old for debug only
        if self.froz_feat is None:  # no old features yet
            return torch.zeros(len(x), self.output_size, device=self.device)
        else:
            if no_old and self.sig_q.sweights[0].shape[0] == self.memory_size:  # debug only, for computing KL(pik, tilde{pik}) in the paper
                return self.sig_q(self.froz_feat(x))[1:].sum(0) * self.eta * self.w_correction
            return self.sig_q(self.froz_feat(x)).sum(0) * self.eta * self.w_correction  # it was easier to put eta here

    def _get_networks(self):
        # Since StaQ builds from previous updates, this is called only once at initialisation. The networks are then updated with update_sigq() and _update_staq_networks()
        if self.network_type == "mlp":
            insize = self.input_size
            ops = []
            for _ in range(self.nb_hidden):
                ops.append(nn.Linear(insize, self.hidden_width))
                ops.append(self.nl)
                insize = self.hidden_width
            output_layer = zero_linear(nn.Linear(self.hidden_width, self.output_size).to(self.device))

            return nn.Sequential(*ops).to(self.device), output_layer
        else:
            assert self.network_config is not None, "CNN configuration must be provided for CNN network type."
            insize = self.input_size
            in_channels = insize[0]
            H, W = insize[1], insize[2]
            ops = []

            # Build CNN from config
            current_channels = in_channels
            self.strides = self.network_config['strides']
            for out_channels, ker_size, stride in zip(self.network_config['channels'], self.network_config['kernel_size'], self.network_config['strides']):
                ops.extend([
                    nn.Conv2d(current_channels, out_channels,
                              kernel_size=ker_size, stride=stride),
                    self.nl
                ])
                current_channels = out_channels

            # Flatten and get output size
            ops.append(nn.Flatten())
            cnn = nn.Sequential(*ops).to(self.device)

            with torch.no_grad():
                sample_input = torch.zeros(1, in_channels, H, W).to(self.device)
                n_flatten = cnn(sample_input).shape[1]

            # Add final linear layer
            ops.append(nn.Linear(n_flatten, self.hidden_width))
            ops.append(self.nl)

            output_layer = zero_linear(nn.Linear(self.hidden_width, self.output_size).to(self.device))

            return nn.Sequential(*ops).to(self.device), output_layer

    def parameters(self):
        if self.train_feat is not None:
            return [*self.train_feat.parameters(), *self.train_q.parameters()]
        return []

    def train(self, train_mode):
        if self.train_feat is not None:
            self.train_feat.train(train_mode)

    def set_entropy_weight(self, weight):
        self._entropy_weight = weight
        self.eta = 1. / (self._kl_weight + self._entropy_weight)
        self.decay = self._kl_weight / (self._kl_weight + self._entropy_weight)
        if self.use_w_correction:
            self.w_correction = 1. / (1. - self.decay ** self.memory_size)
        else:
            self.w_correction = 1.

    def update_sigq(self, decay=True):
        # merge frozen and trainable MLPs and delete oldest function if necessary
        self.train_feat.train(False)
        if self.froz_feat is None:  # building first frozen feat network
            self.froz_feat = StackedNN(self.train_feat, self.memory_size, strides=self.strides)
            self.sig_q = StackedNN([self.train_q], self.memory_size)
        else:
            self.froz_feat.push(self.train_feat)
            if decay:
                self.sig_q.decay(self.decay)
            self.sig_q.push([self.train_q])