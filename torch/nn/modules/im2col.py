import torch
from torch.autograd import Variable

from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F


class Im2Col(Module):
    """PLACEHOLDER TEMPLATE COMMENT - [TODO](JEB) needs updating
    """

    def __init__(self, kH, kW, dH, dW, padH, padW, sH, sW):
        super(Im2Col, self).__init__()
        self.kH = kH
        self.kW = kW
        self.dH = dH
        self.dW = dW
        self.padH = padH
        self.padW = padW
        self.sH = sH
        self.sW = sW

    def forward(self, input):
        return F.im2col(input, self.kH, self.kW, self.dH, self.dW,
                        self.padH, self.padW, self.sH, self.sW)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'kH=' + str(self.kH) \
            + ', kW=' + str(self.kH) \
            + ', dH=' + str(self.dH) \
            + ', dW=' + str(self.dW) \
            + ', padH=' + str(self.padH) \
            + ', padW=' + str(self.padW) \
            + ', sH=' + str(self.sH) \
            + ', sW=' + str(self.sW) + ')'

