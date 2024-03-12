import torch
import math

import torch.nn as nn
from torch import Tensor
from torch.nn import Module, init
from torch.nn.parameter import Parameter
from torch.nn import functional as F

import collections
from itertools import repeat

class PHMLinear(nn.Module):

  def __init__(self, n, in_features, out_features, bias=True):
    super(PHMLinear, self).__init__()
    self.n = n
    self.in_features = in_features
    self.out_features = out_features

    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter('bias', None)

    self.A = Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))

    self.S = Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, self.out_features//n, self.in_features//n))))

    self.weight = torch.zeros((self.out_features, self.in_features))

    if self.bias is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      if fan_in != 0:
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

  def kronecker_product1(self, a, b):
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features))
    for i in range(self.n):
        H = H + torch.kron(self.A[i], self.F[i])
    return H

  def forward(self, input: Tensor) -> Tensor:
    self.weight = torch.sum(self.kronecker_product1(self.A, self.S), dim=0)
    input = input.type(dtype=self.weight.type())
    return F.linear(input, weight=self.weight, bias=self.bias)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}, n={}'.format(
      self.in_features, self.out_features, self.bias is not None, self.n)

  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.A, a=math.sqrt(5))
    init.kaiming_uniform_(self.S, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      if fan_in != 0:
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)


class PHMConv1d(Module):
  def __init__(self, n, in_features, out_features, kernel_size, stride=1, padding=0, dilation=1, bias=True, boundary=False):
    super(PHMConv1d, self).__init__()
    self.n = n
    self.in_features = in_features
    self.out_features = out_features
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation

    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter('bias', None)

    if boundary:
      self.bnd = nn.Parameter(torch.Tensor(out_features, 1, 1))
    else:
      self.register_parameter('bnd', None)

    self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
    self.F = nn.Parameter(torch.nn.init.xavier_uniform_(
        torch.zeros((n, self.out_features//n, self.in_features//n, kernel_size))))
    self.weight = torch.zeros((self.out_features, self.in_features))

    if self.bias is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      if fan_in != 0:
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    if self.bnd is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      if fan_in != 0:
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bnd, -bound, bound)

  def kronecker_product1(self, A, F):
    siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(F.shape[-3:-1]))
    siz2 = torch.Size(torch.tensor(F.shape[-1:]))
    res = A.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1) * F.unsqueeze(-3).unsqueeze(-5)
    siz0 = res.shape[:1]
    out = res.reshape(siz0 + siz1 + siz2)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size))
    for i in range(self.n):
        kron_prod = torch.kron(self.A[i], self.F[i]).view(self.out_features, self.in_features, self.kernel_size, self.kernel_size)
        H = H + kron_prod
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.A, self.F), dim=0)
    # self.weight = self.kronecker_product2()
    input = input.type(dtype=self.weight.type())

    if self.bnd is not None:
      self.weight = torch.cat((self.weight, self.bnd), 1)

    return F.conv1d(input, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}, n={}'.format(
      self.in_features, self.out_features, self.bias is not None, self.n)

  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.A, a=math.sqrt(5))
    init.kaiming_uniform_(self.F, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      if fan_in != 0:
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)


class PHMConv2d(Module):

  def __init__(self, n, in_features, out_features, kernel_size, stride=1, padding=0, dilation=1, bias=True):
    super(PHMConv2d, self).__init__()
    self.n = n
    self.in_features = in_features
    self.out_features = out_features
    if not isinstance(kernel_size, collections.abc.Iterable):
      self.kernel_size = tuple(repeat(kernel_size, 2))
    else:
      self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation

    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter('bias', None)

    self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
    self.F = nn.Parameter(torch.nn.init.xavier_uniform_(
        torch.zeros((n, self.out_features//n, self.in_features//n, self.kernel_size[0], self.kernel_size[1]))))
    self.weight = torch.zeros((self.out_features, self.in_features))

    if self.bias is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      if fan_in != 0:
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

  def kronecker_product1(self, A, F):
    siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(F.shape[-4:-2]))
    siz2 = torch.Size(torch.tensor(F.shape[-2:]))
    res = A.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1).unsqueeze(-1) * F.unsqueeze(-4).unsqueeze(-6)
    siz0 = res.shape[:1]
    out = res.reshape(siz0 + siz1 + siz2)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size))
    for i in range(self.n):
        kron_prod = torch.kron(self.A[i], self.F[i]).view(self.out_features, self.in_features, self.kernel_size, self.kernel_size)
        H = H + kron_prod
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.A, self.F), dim=0)
    # self.weight = self.kronecker_product2()
    input = input.type(dtype=self.weight.type())
    return F.conv2d(input, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}, n={}'.format(
      self.in_features, self.out_features, self.bias is not None, self.n)

  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.A, a=math.sqrt(5))
    init.kaiming_uniform_(self.F, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      if fan_in != 0:
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)


class PHMConv3d(Module):

  def __init__(self, n, in_features, out_features, kernel_size, stride=1, padding=0, dilation=1, bias=True):
    super(PHMConv3d, self).__init__()
    self.n = n
    self.in_features = in_features
    self.out_features = out_features
    if not isinstance(kernel_size, collections.abc.Iterable):
      self.kernel_size = tuple(repeat(kernel_size, 3))
    else:
      self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation

    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter('bias', None)

    self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
    self.F = nn.Parameter(torch.nn.init.xavier_uniform_(
        torch.zeros((n, self.out_features//n, self.in_features//n, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))))
    self.weight = torch.zeros((self.out_features, self.in_features))

    if self.bias is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      if fan_in != 0:
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

  def kronecker_product1(self, A, F):
    siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(F.shape[-5:-3]))
    siz2 = torch.Size(torch.tensor(F.shape[-3:]))
    res = A.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * F.unsqueeze(-5).unsqueeze(-7)
    siz0 = res.shape[:1]
    out = res.reshape(siz0 + siz1 + siz2)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size, self.kernel_size))
    for i in range(self.n):
        kron_prod = torch.kron(self.A[i], self.F[i]).view(self.out_features, self.in_features, self.kernel_size, self.kernel_size, self.kernel_size)
        H = H + kron_prod
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.A, self.F), dim=0)
    # self.weight = self.kronecker_product2()
    input = input.type(dtype=self.weight.type())
    return F.conv3d(input, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}, n={}'.format(
      self.in_features, self.out_features, self.bias is not None, self.n)

  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.A, a=math.sqrt(5))
    init.kaiming_uniform_(self.F, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      if fan_in != 0:
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)
