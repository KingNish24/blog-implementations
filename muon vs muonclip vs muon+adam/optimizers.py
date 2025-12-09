import torch
import torch.nn as nn
from typing import Optional, Callable
import math
from .utlis import apply_qk_clip_per_head, newton_schulz

class MuonPlusAdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr <= 0:
            raise ValueError("lr must be positive")

        params = list(params)

        muon_params  = [p for p in params if p.ndim == 2]
        adamw_params = [p for p in params if p.ndim != 2]

        param_groups = [
            {"params": muon_params,  "type": "muon",  "lr": lr},
            {"params": adamw_params, "type": "adamw", "lr": lr},
        ]

        super().__init__(param_groups, defaults={"lr": lr})

        # create inner optimizers targeting SAME PARAM TENSORS
        self._muon  = torch.optim.Muon(muon_params, lr=lr)
        self._adamw = torch.optim.AdamW(adamw_params, lr=lr)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):

        loss = closure() if closure else None

        self._adamw.step()
        self._muon.step()

        return loss

    def zero_grad(self, set_to_none=True):
        self._adamw.zero_grad(set_to_none)
        self._muon.zero_grad(set_to_none)

class Muon(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr <= 0:
            raise ValueError("lr must be positive")

        params = list(params)

        muon_params  = [p for p in params if p.ndim == 2]

        param_groups = [
            {"params": muon_params,  "type": "muon",  "lr": lr},
        ]

        super().__init__(param_groups, defaults={"lr": lr})

        # create inner optimizers targeting SAME PARAM TENSORS
        self._muon  = torch.optim.Muon(muon_params, lr=lr)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):

        loss = closure() if closure else None

        self._muon.step()

        return loss

    def zero_grad(self, set_to_none=True):
        self._muon.zero_grad(set_to_none)

# Credit: MuonClip implementation adapted from 

class MuonClip(torch.optim.Optimizer):  
    """
    MuonClip Optimizer - Combines Muon optimizer with QK-Clip for stable LLM training.
    
    This optimizer applies:
    1. Muon updates with Newton-Schulz orthogonalization for 2D+ parameters
    2. Standard momentum for 1D parameters
    3. QK-Clip to prevent attention logit explosion
    
    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-3)
        momentum: Momentum coefficient μ (default: 0.95)
        weight_decay: Weight decay coefficient λ (default: 0.01)
        tau: QK-Clip threshold τ (default: 100.0)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        eps: Numerical stability epsilon (default: 1e-7)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        tau: float = 100.0,
        ns_steps: int = 5,
        eps: float = 1e-7
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if tau <= 0.0:
            raise ValueError(f"Invalid tau value: {tau}")
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            tau=tau,
            ns_steps=ns_steps,
            eps=eps
        )
        super().__init__(params, defaults)
        
        # For QK-Clip functionality
        self.model = None
        self.attention_layers = []
    
    def set_model(self, model: nn.Module):
        """
        Set model reference for QK-Clip functionality.
        
        Args:
            model: PyTorch model containing attention layers
        """
        self.model = model
        if hasattr(model, 'get_attention_layers'):
            self.attention_layers = model.get_attention_layers()
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize momentum buffer
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = state['momentum_buffer']
                
                # Apply momentum: Mt = μMt−1 + Gt
                buf.mul_(momentum).add_(grad)
                
                if p.ndim >= 2:  # 2D+ parameters - use Muon
                    # Apply Newton-Schulz orthogonalization
                    if p.ndim > 2:
                        original_shape = buf.shape
                        buf_2d = buf.view(buf.shape[0], -1)
                        orthogonal_update = newton_schulz(buf_2d, ns_steps, eps)
                        orthogonal_update = orthogonal_update.view(original_shape)
                    else:
                        orthogonal_update = newton_schulz(buf, ns_steps, eps)
                    
                    # RMS matching factor: √(max(n,m) × 0.2)
                    n, m = p.shape[0], p.shape[1] if p.ndim > 1 else 1
                    rms_factor = math.sqrt(max(n, m) * 0.2)
                    orthogonal_update = orthogonal_update * rms_factor
                    
                    # Update: Wt = Wt−1 − η(Ot + λWt−1)
                    p.add_(orthogonal_update + weight_decay * p, alpha=-lr)
                else:
                    # 1D parameters - standard momentum
                    p.add_(buf + weight_decay * p, alpha=-lr)
        
        # Apply QK-Clip
        self._apply_qk_clip()
        
        return loss
    
    def _apply_qk_clip(self):
        """Apply QK-Clip to attention layers to prevent logit explosion."""
        if not self.attention_layers:
            return
        
        tau = self.param_groups[0]['tau']
        
        for layer_name, attention_layer in self.attention_layers:
            if not hasattr(attention_layer, 'max_logits'):
                continue
            
            max_logits = attention_layer.max_logits
            if not max_logits:
                continue
            
            # Handle both scalar and per-head max logits
            if isinstance(max_logits, (int, float)):
                max_logits = [max_logits]
            
            if hasattr(attention_layer, 'query') and hasattr(attention_layer, 'key'):
                # In-place modification - no copy needed
                apply_qk_clip_per_head(
                    attention_layer.query.weight.data,
                    attention_layer.key.weight.data,
                    max_logits,
                    tau
                )
                
