import torch
from typing import List, Optional, Union, Callable

def apply_qk_clip_per_head(
    query_weights: torch.Tensor,
    key_weights: torch.Tensor,
    max_logits_per_head: Union[List[float], torch.Tensor],
    tau: float = 100.0
) -> None:
    """
    Apply per-head QK-Clip following Algorithm 1, lines 11-16 (IN-PLACE).
    
    Args:
        query_weights: [d_model, d_model] Query projection weights (modified in-place)
        key_weights: [d_model, d_model] Key projection weights (modified in-place)
        max_logits_per_head: List or tensor of max logits per head
        tau: Threshold for clipping
    """
    if isinstance(max_logits_per_head, list):
        max_logits_per_head = torch.tensor(
            max_logits_per_head,
            device=query_weights.device,
            dtype=query_weights.dtype
        )
    apply_qk_clip_vectorized(query_weights, key_weights, max_logits_per_head, tau)

@torch.no_grad()
def apply_qk_clip_vectorized(
    query_weights: torch.Tensor,
    key_weights: torch.Tensor,
    max_logits_per_head: torch.Tensor,
    tau: float = 100.0
) -> None:
    """
    Vectorized per-head QK-Clip for efficient processing (IN-PLACE).
    
    Applies clipping to all heads simultaneously using in-place torch operations.
    Modifies query_weights and key_weights directly to avoid allocations.
    
    Args:
        query_weights: [d_model, d_model] Query projection weights (modified in-place)
        key_weights: [d_model, d_model] Key projection weights (modified in-place)
        max_logits_per_head: Tensor of max logits per head [num_heads]
        tau: Threshold for clipping (default: 100.0)
    """
    d_model = query_weights.shape[0]
    num_heads = len(max_logits_per_head)
    d_k = d_model // num_heads
    
    # Ensure tensor type
    if not isinstance(max_logits_per_head, torch.Tensor):
        max_logits_per_head = torch.tensor(
            max_logits_per_head, 
            device=query_weights.device,
            dtype=query_weights.dtype
        )
    
    # Compute scaling factors: gamma = tau / max_logit where max_logit > tau
    needs_clip = max_logits_per_head > tau
    
    # If no clipping needed, return early
    if not needs_clip.any():
        return
    
    gamma = torch.where(
        needs_clip,
        tau / max_logits_per_head.clamp(min=1e-8),
        torch.ones_like(max_logits_per_head)
    )
    sqrt_gamma = torch.sqrt(gamma)
    
    # Reshape weights to [d_model, num_heads, d_k] for per-head scaling
    # Views share underlying storage, so in-place ops modify original tensor
    q_reshaped = query_weights.view(d_model, num_heads, d_k)
    k_reshaped = key_weights.view(d_model, num_heads, d_k)
    
    # Apply per-head scaling IN-PLACE: broadcast sqrt_gamma [num_heads] over [d_model, num_heads, d_k]
    q_reshaped.mul_(sqrt_gamma.view(1, num_heads, 1))
    k_reshaped.mul_(sqrt_gamma.view(1, num_heads, 1))

def _newton_schulz_impl(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Optimized Newton-Schulz iteration for matrix orthogonalization.
    
    Uses bfloat16 for faster GPU computation and optimized matrix operations.
    Based on the quintic iteration from the Muon paper with coefficients
    chosen to maximize convergence rate.
    
    Args:
        G: Input matrix tensor
        steps: Number of iteration steps (default: 5)
        eps: Numerical stability epsilon (default: 1e-7)
        
    Returns:
        Orthogonalized matrix approximating G @ (G^T @ G)^(-1/2)
    """
    # Optimized coefficients from Muon paper (quintic iteration)
    a, b, c = 3.4445, -4.7750, 2.0315
    
    # Use bfloat16 for faster computation on modern GPUs
    original_dtype = G.dtype
    if G.device.type == 'cuda' and torch.cuda.is_bf16_supported():
        X = G.bfloat16()
    else:
        X = G.float()
    
    # Normalize spectral norm to at most 1
    X = X / (X.norm() + eps)
    
    # Handle rectangular matrices: always work with "wide" matrices
    transposed = False
    if X.size(0) > X.size(1):
        X = X.mT 
        transposed = True
    
    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        # Fuse: B @ X = (b*A + c*A@A) @ X = b*(A@X) + c*(A@A@X)
        AX = A @ X
        X = a * X + b * AX + c * (A @ AX)  
    
    # Transpose back if needed
    if transposed:
        X = X.mT
    
    return X.to(original_dtype)


# Try to use torch.compile if available (PyTorch 2.0+)
try:
    newton_schulz_fast = torch.compile(_newton_schulz_impl)
except (AttributeError, RuntimeError):
    newton_schulz_fast = _newton_schulz_impl

def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration for matrix orthogonalization.
    
    This is a wrapper that uses the optimized JIT-compiled version when available.
    
    Args:
        G: Input matrix tensor
        steps: Number of iteration steps
        eps: Small epsilon for numerical stability
        
    Returns:
        Orthogonalized matrix
    """
    return newton_schulz_fast(G, steps, eps)


