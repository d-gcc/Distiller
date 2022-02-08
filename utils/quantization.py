import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def weight_quantization(b, level=1):

    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            #input.div_(alpha)                        # weights are first divided by alpha
            if level > 0:
                mean = input.mean()
                std = input.std()
                alpha = mean.add(std.mul(level))
                input_c = input.clamp(min=-alpha, max=alpha)       # then clipped to [-1,1]
            else:
                input_c = input
            sign = input_c.sign()
            input_abs = input_c.abs()
            input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            #input_q = input_q.mul(alpha)               # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()             # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs()>1.).float()
            sign = input.sign()
            #grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            return grad_input #, grad_alpha

    return _pq().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        #assert (w_bit <=5 and w_bit > 0) or w_bit == 32
        self.w_bit = w_bit-1
        self.level = level
        self.weight_q = weight_quantization(b=self.w_bit, level=self.level)
        self.register_parameter('wgt_alpha', Parameter(torch.tensor(1.0)))

    def forward(self, weight):
        if self.w_bit == 32:
            weight_q = weight
        else:
            #mean = weight.data.mean()
            #std = weight.data.std()
            #weight = weight.add(-mean).div(std)      # weights normalization
            weight_q = self.weight_q(weight)
        return weight_q


def act_quantization(b, level=1):

    def uniform_quant(x, b=3):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            if level > 0:
                mean = input.mean()
                std = input.std()
                alpha = mean.add(std.mul(level))
                input_c = input.clamp(max=alpha)
            else:
                input_c = input
            input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            #input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input > 1.).float()
            #grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
            grad_input = grad_input*(1-i)
            return grad_input #, grad_alpha

    return _uq().apply


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, bit=None, config=None):
        super(Conv1dSamePadding, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)
    
        self.bit = bit
        self.additive = config.additive
        self.device = config.device
        self.level = config.std_dev

        self.weight_quant = weight_quantization(self.bit-1, level=self.level)
        self.weight = nn.Parameter(torch.randn(self.weight.size()))
        
        if self.bit != 32:
            self.weight = nn.Parameter(self.weight_quant(self.weight))
            self.weight_quant = weight_quantization(self.bit-1, level=self.level)
            self.act_alq = act_quantization(self.bit, level=self.level)
            self.act_alpha = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if self.bit == 32:
            return conv1d_same_padding(x, self.weight, self.bias, self.stride,
                                       self.dilation, self.groups, self.device)
        else:
            weight_q = self.weight_quant(self.weight)
            x = self.act_alq(x)
            return conv1d_same_padding(x, weight_q, self.bias, self.stride,
                                       self.dilation, self.groups, self.device)

def conv1d_same_padding(input, weight, bias, stride, dilation, groups, device):
    input = input.to(device)
    weight = weight.to(device)
    #print(weight.reshape(-1))
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)

    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int,config = None) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,config=config),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)