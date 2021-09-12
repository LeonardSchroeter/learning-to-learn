import torch


class QuadraticFunction(torch.nn.Module):
    def __init__(self, dimension, **kwargs):
        super(QuadraticFunction, self).__init__(**kwargs)
        self.dimension = dimension
        self.W = torch.rand(dimension, dimension)
        self.y = torch.rand(dimension)

    def forward(self, theta):
        return torch.linalg.norm(torch.matmul(self.W, theta) - self.y)
