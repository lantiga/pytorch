import torch
from torch.autograd import Variable

from .module import Module
from .utils import _single, _pair, _triple
from .._functions.thnn.unfold import Im2Col


class Unfold(Module):
    """
    Rearranges image blocks into columns.

    Converts each sliding :math:`kernel_size` block of the input into a column of
    the output.  The output has :math:`\prod_i k_i, k_i\in kernel_size` rows and
    contains as many columns as there are :math:`kernel_size` neighborhoods of
    the input according to the stride and dilation values.

    | If :attr:`padding` is non-zero, then the input is implicitly
    zero-padded on both sides by :attr:`padding` number of points
    | :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what
    dilation does.

    Args:
        kernel_size (int or tuple): the size of the neighborhoods to convert
        stride (int or tuple): the stride of the sliding neighborhood. Default: 1
        padding (int or tuple, optional): implicit zero padding to be added on
                                          both sides of input. Default: 0
        dilation (int or tuple, optional): a parameter that controls the
                                           stride of elements within the
                                           neighborhood

    Shape:
        - Input: :math:`(C, L_{in})`
        - Output: :math:`(C * \prod(kernel_size), L_{out},)` where
          :math:`L_{out} = floor((L_{in} + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    Examples::

        >>> # kernel_size (2, 2), dilation (1, 1), padding (0, 0), stride (1, 1)
        >>> unfold = nn.Unfold((2, 2), (1, 1), (0, 0), (1, 1))
        >>> input = autograd.Variable(torch.randn(1, 256, 256))
        >>> output = unfold(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, kernel_size, dilation, padding, stride):
        super(Unfold, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        return Im2Col.apply(input, self.kernel_size, self.dilation,
                        self.padding, self.stride)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', dilation=' + str(self.dilation) \
            + ', padding=' + str(self.padding) \
            + ', stride=' + str(self.stride) + ')'

