import torch
from torch.amp import custom_bwd, custom_fwd
from torch import nn
import torch.autograd as autograd

class ProLU(torch.autograd.Function):
    STE: torch.autograd.Function
    ReLU: torch.autograd.Function

    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, m, b):
        gate = (m + b > 0) & (m > 0)
        ctx.save_for_backward(m, gate)
        return torch.where(gate, m, 0)

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "This method should be overridden by a subclass of ProLU to provide a backward implementation."
        )


class ProLU_ReLU(ProLU):
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        m, gate = ctx.saved_tensors
        gated_grad = torch.where(gate, grad_output, 0)
        grad_m, grad_b = gated_grad.clone(), gated_grad.clone()
        return grad_m, grad_b, None


class ProLU_STE(ProLU):
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        m, gate = ctx.saved_tensors
        gated_grad = torch.where(gate, grad_output, 0)
        grad_b = gated_grad * m
        grad_m = gated_grad + grad_b.clone()
        return grad_m, grad_b, None


ProLU.STE = ProLU_STE
ProLU.ReLU = ProLU_ReLU


def prolu_ste(m: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ProLU_STE.apply(m, b)


def prolu_relu(m: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ProLU_ReLU.apply(m, b)

# [https://arxiv.org/pdf/2110.11940]
def and_ail(x, y):
    mask = (x < 0) & (y < 0)
    out = torch.where(mask, x + y, torch.minimum(x, y))
    return out

def or_ail(x, y):
    mask = (x > 0) & (y > 0)
    out = torch.where(mask, x + y, torch.maximum(x, y))
    return out

def xor_ail(x, y):
    out = torch.sgn(x*y) * torch.minimum(torch.abs(x), torch.abs(y))
    return out


class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth) -> torch.Tensor:
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class JumpReLU(nn.Module):
    def __init__(self, feature_size, bandwidth, device='cpu'):
        super(JumpReLU, self).__init__()
        self.log_threshold = nn.Parameter(torch.zeros(feature_size, device=device))
        self.bandwidth = bandwidth

    def forward(self, x) -> torch.Tensor:
        return JumpReLUFunction.apply(x, self.log_threshold, self.bandwidth)


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth
