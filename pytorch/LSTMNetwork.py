import math

import torch


class LSTMNetwork(torch.nn.Module):
    def __init__(self):
        super(LSTMNetwork, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=10, proj_size=1)

    def forward(self, x, hidden = None):
        shape = list(x.shape)
        num_parameters = math.prod(shape)

        if hidden == None:
            h = torch.rand(1, num_parameters, 1, dtype=torch.float32)
            c = torch.rand(1, num_parameters, 10, dtype=torch.float32)
            hidden = (h, c)

        x = x.reshape((1, num_parameters, 1))
        # x, hidden = self.lstm(x, hidden)
        x = x.reshape(shape)
        return x, hidden
