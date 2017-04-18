from torch.autograd.function import Function, once_differentiable
from torch._thnn import type2backend

from . import _all_functions

class Col2Im(Function):

    @staticmethod
    def forward(ctx, input, output_size, kernel_size, dilation, padding, stride):

        ctx.output_size = output_size
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.padding = padding
        ctx.stride = stride

        ctx.save_for_backward(input)
        input = input.contiguous()

        ctx._backend = type2backend[type(input)]

        output = input.new()

        ctx._backend.Col2Im_updateOutput(ctx._backend.library_state,
                                    input, output,
                                    output_size[0], output_size[1],
                                    kernel_size[0], kernel_size[1],
                                    dilation[0], dilation[1],
                                    padding[0], padding[1],
                                    stride[0], stride[1])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        input = input.contiguous()

        grad_input = grad_output.new()

        ctx._backend.Col2Im_updateGradInput(ctx._backend.library_state,
                                       input, grad_output,
                                       grad_input,
                                       ctx.output_size[0], ctx.output_size[1],
                                       ctx.kernel_size[0], ctx.kernel_size[1],
                                       ctx.dilation[0], ctx.dilation[1],
                                       ctx.padding[0], ctx.padding[1],
                                       ctx.stride[0], ctx.stride[1])
        return grad_input, None, None, None, None, None

# TODO: remove this file?
# _all_functions.append(Col2Im)
