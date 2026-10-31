"""Optimized Triton SwiGLU kernel for Qwen2.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def qwen2_mlp_act_fwd(
    gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(gate)
    out = gate * sig * up

    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def qwen2_mlp_act_bwd(
    grad_ptr, gate_ptr, up_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(gate)
    gs = gate * sig
    dup = grad * gs
    dgate = grad * up * (sig + gs * (1.0 - sig))

    tl.store(gate_ptr + offsets, dgate.to(gate_ptr.dtype.element_ty), mask=mask)
    tl.store(up_ptr + offsets, dup.to(up_ptr.dtype.element_ty), mask=mask)


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        if not gate.is_contiguous():
            gate = gate.contiguous()
        if not up.is_contiguous() or up.dtype != gate.dtype:
            up = up.to(gate.dtype).contiguous()

        out = torch.empty_like(gate)
        n = gate.numel()
        BLOCK = 8192

        qwen2_mlp_act_fwd[(triton.cdiv(n, BLOCK),)](
            gate, up, out, n, BLOCK=BLOCK, num_warps=8
        )
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        if not grad.is_contiguous() or grad.dtype != gate.dtype:
            grad = grad.to(gate.dtype).contiguous()

        n = gate.numel()
        BLOCK = 8192

        qwen2_mlp_act_bwd[(triton.cdiv(n, BLOCK),)](
            grad, gate, up, n, BLOCK=BLOCK, num_warps=8
        )
        return gate, up


def swiglu(gate, up):
    return SwiGLU.apply(gate, up)


def patch_qwen2_mlp():
    from transformers.models.qwen2 import modeling_qwen2

    orig_mlp = modeling_qwen2.Qwen2MLP.forward

    def mlp_forward(self, x):
        if not x.is_cuda:
            return orig_mlp(self, x)
        return self.down_proj(SwiGLU.apply(self.gate_proj(x), self.up_proj(x)))

    modeling_qwen2.Qwen2MLP.forward = mlp_forward
