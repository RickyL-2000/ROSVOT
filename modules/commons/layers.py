import torch
from torch import nn
from torch.autograd import Function

class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1, eps=1e-5):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args

    def forward(self, x):
        return x.permute(self.args)


def Linear(in_features, out_features, bias=True, init_type='xavier'):
    m = nn.Linear(in_features, out_features, bias)
    if init_type == 'xavier':
        nn.init.xavier_uniform_(m.weight)
    elif init_type == 'kaiming':
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx=None, init_type='normal'):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    if init_type == 'normal':
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    elif init_type == 'kaiming':
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, input, coeff=1.):
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

