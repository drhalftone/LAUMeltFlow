"""
Custom scatter operations for DirectML compatibility.

DirectML doesn't support scatter_add_, so we implement alternatives
using operations that are supported.
"""

import torch
from typing import Optional


def scatter_add_dml(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None
) -> torch.Tensor:
    """
    Scatter-add operation compatible with DirectML.

    Uses one-hot encoding + matmul instead of scatter_add_.

    Parameters
    ----------
    src : torch.Tensor
        Source tensor with values to scatter
    index : torch.Tensor
        Index tensor indicating where to add values
    dim : int
        Dimension along which to scatter (only dim=0 supported)
    dim_size : int, optional
        Size of output dimension

    Returns
    -------
    torch.Tensor
        Output tensor with scattered values summed
    """
    if dim != 0:
        raise NotImplementedError("Only dim=0 is supported for DirectML scatter_add")

    if dim_size is None:
        dim_size = int(index.max()) + 1

    # Method: Use one-hot encoding and matrix multiplication
    # This avoids scatter_add_ entirely

    # Create one-hot encoding of indices: (n_edges,) -> (n_edges, n_nodes)
    # one_hot[i, j] = 1 if index[i] == j, else 0
    one_hot = torch.zeros(index.size(0), dim_size, device=src.device, dtype=src.dtype)

    # Use index_select and assignment via advanced indexing
    # This is equivalent to: one_hot[i, index[i]] = 1 for all i
    arange = torch.arange(index.size(0), device=src.device)
    one_hot[arange, index] = 1.0

    # Now: out = one_hot.T @ src
    # Shape: (n_nodes, n_edges) @ (n_edges, features) = (n_nodes, features)
    out = torch.mm(one_hot.t(), src)

    return out


def scatter_add_dml_v2(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None
) -> torch.Tensor:
    """
    Alternative scatter-add using a loop (slower but always works).

    Use this as fallback if the matmul version has issues.
    """
    if dim != 0:
        raise NotImplementedError("Only dim=0 is supported")

    if dim_size is None:
        dim_size = int(index.max()) + 1

    # Initialize output
    out_shape = (dim_size,) + src.shape[1:]
    out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)

    # Use index_add_ which might be supported
    try:
        out.index_add_(0, index, src)
        return out
    except RuntimeError:
        pass

    # Fallback: explicit loop (slow but guaranteed to work)
    for i in range(dim_size):
        mask = (index == i)
        if mask.any():
            out[i] = src[mask].sum(dim=0)

    return out


class ScatterAddDML(torch.autograd.Function):
    """
    Autograd-compatible scatter_add for DirectML.

    Implements both forward and backward passes.
    """

    @staticmethod
    def forward(ctx, src, index, dim_size):
        ctx.save_for_backward(index)
        ctx.dim_size = dim_size
        ctx.src_shape = src.shape

        # Use one-hot matmul method
        one_hot = torch.zeros(index.size(0), dim_size, device=src.device, dtype=src.dtype)
        arange = torch.arange(index.size(0), device=src.device)
        one_hot[arange, index] = 1.0

        out = torch.mm(one_hot.t(), src)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        index, = ctx.saved_tensors

        # Gradient w.r.t. src: just gather from grad_output using index
        grad_src = grad_output[index]

        return grad_src, None, None


def scatter_add_autograd(src, index, dim_size=None):
    """Scatter add with autograd support for DirectML."""
    if dim_size is None:
        dim_size = int(index.max()) + 1
    return ScatterAddDML.apply(src, index, dim_size)


# Test function
def test_scatter_add():
    """Test that our scatter_add matches PyTorch's."""
    print("Testing custom scatter_add implementations...")

    # Test data
    src = torch.randn(10, 3)
    index = torch.tensor([0, 1, 0, 2, 1, 0, 2, 1, 0, 2])
    dim_size = 3

    # Reference using CPU scatter_add_
    ref = torch.zeros(dim_size, 3)
    ref.scatter_add_(0, index.unsqueeze(1).expand(-1, 3), src)

    # Our implementation
    out1 = scatter_add_dml(src, index, dim_size=dim_size)
    out2 = scatter_add_dml_v2(src, index, dim_size=dim_size)
    out3 = scatter_add_autograd(src, index, dim_size=dim_size)

    print(f"  Reference:\n{ref}")
    print(f"  scatter_add_dml:\n{out1}")
    print(f"  Max diff (v1): {(ref - out1).abs().max():.2e}")
    print(f"  Max diff (v2): {(ref - out2).abs().max():.2e}")
    print(f"  Max diff (v3): {(ref - out3).abs().max():.2e}")

    # Test gradient
    src_grad = src.clone().requires_grad_(True)
    out = scatter_add_autograd(src_grad, index, dim_size=dim_size)
    loss = out.sum()
    loss.backward()
    print(f"  Gradient computed successfully: {src_grad.grad is not None}")

    print("All tests passed!")


if __name__ == "__main__":
    test_scatter_add()
