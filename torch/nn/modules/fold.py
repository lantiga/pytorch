import torch
from torch.autograd import Variable

from .module import Module
from .utils import _single, _pair, _triple
from .._functions.thnn.fold import Col2Im


class Fold(Module):
    """
    TODO: docstring
    """

    def __init__(self, output_size, kernel_size, dilation, padding, stride):
        super(Fold, self).__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        return Col2Im.apply(input, self.output_size, self.kernel_size, self.dilation,
                        self.padding, self.stride)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'output_size=' + str(self.output_size) \
            + ', kernel_size=' + str(self.kernel_size) \
            + ', dilation=' + str(self.dilation) \
            + ', padding=' + str(self.padding) \
            + ', stride=' + str(self.stride) + ')'

