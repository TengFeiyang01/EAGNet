from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace


class JointNLLLoss(nn.Module):
    """
    QCNext Joint Multi-Agent NLL Loss
    Based on the paper formulation: joint trajectory distribution as mixture of Laplace distributions
    Uses scene-level winner-take-all strategy
    
    Joint distribution: ∑_{k=1}^K π_k ∏_{i=1}^{A'} ∏_{t=1}^{T'} f(p_i^{t,x}|μ_{i,k}^{t,x}, b_{i,k}^{t,x}) f(p_i^{t,y}|μ_{i,k}^{t,y}, b_{i,k}^{t,y})
    """

    def __init__(self, 
                 reduction: str = 'mean'):
        super(JointNLLLoss, self).__init__()
        self.reduction = reduction

    def forward(self, 
                pred_loc: torch.Tensor,
                pred_scale: torch.Tensor, 
                target: torch.Tensor, 
                agent_batch: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute joint multi-agent trajectory NLL loss with scene-level winner-take-all
        
        Args:
            pred_loc: [K, A, T, 2] - Location parameters (μ) for K joint scenes
            pred_scale: [K, A, T, 2] - Scale parameters (b) for K joint scenes  
            target: [A, T, 2] - Ground truth trajectories
            agent_batch: [A] - Batch assignment for each agent
            mask: [A, T] - Valid mask for agents and timesteps
            
        Returns:
            loss: Scalar loss value using winner-take-all strategy
        """
        K, A, T, _ = pred_loc.shape
        batch_size = agent_batch.max().item() + 1
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(0).expand(K, -1, -1)  # [K, A, T]
        else:
            mask_expanded = torch.ones(K, A, T, dtype=torch.bool, device=pred_loc.device)

        # Expand target for all joint scenes
        target_expanded = target.unsqueeze(0).expand(K, -1, -1, -1)  # [K, A, T, 2]
        
        # Compute negative log-likelihood for each joint scene
        # Using Laplace distribution: f(x|μ,b) = (1/2b)exp(-|x-μ|/b)
        # NLL = log(2b) + |x-μ|/b
        
        # Ensure scale is positive
        pred_scale = pred_scale.clamp(min=1e-6)
        
        # Compute NLL for x and y dimensions separately
        nll_x = torch.log(2 * pred_scale[..., 0]) + torch.abs(target_expanded[..., 0] - pred_loc[..., 0]) / pred_scale[..., 0]
        nll_y = torch.log(2 * pred_scale[..., 1]) + torch.abs(target_expanded[..., 1] - pred_loc[..., 1]) / pred_scale[..., 1]
        
        # Combined NLL for both dimensions
        joint_nll = nll_x + nll_y  # [K, A, T]
        
        # Apply mask and compute scene-level NLL
        masked_nll = joint_nll * mask_expanded.float()  # [K, A, T]
        
        # Sum over agents and timesteps to get scene-level NLL for each joint scene
        scene_nlls = []
        for k in range(K):
            scene_nll = 0.0
            for b in range(batch_size):
                # Get agents in this batch/scene
                batch_mask = (agent_batch == b)
                if batch_mask.sum() == 0:
                    continue
                
                # Sum NLL for all agents and timesteps in this scene
                scene_agent_nll = masked_nll[k, batch_mask].sum()
                scene_nll += scene_agent_nll
            
            scene_nlls.append(scene_nll)
        
        scene_nlls = torch.stack(scene_nlls)  # [K]
        
        # Winner-take-all: select the best joint scene (minimum NLL)
        best_scene_idx = scene_nlls.argmin()
        winner_nll = scene_nlls[best_scene_idx]
        
        if self.reduction == 'mean':
            # Normalize by number of valid predictions
            valid_count = mask_expanded[0].sum().clamp(min=1)
            return winner_nll / valid_count
        elif self.reduction == 'sum':
            return winner_nll
        elif self.reduction == 'none':
            return winner_nll
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class JointMixtureNLLLoss(nn.Module):
    """
    QCNext Joint Mixture NLL Loss for Scene-level Classification
    Optimizes the mixing coefficients π_k using classification loss
    """

    def __init__(self, reduction: str = 'mean'):
        super(JointMixtureNLLLoss, self).__init__()
        self.reduction = reduction

    def forward(self, 
                pred_loc: torch.Tensor,
                pred_scale: torch.Tensor,
                target: torch.Tensor, 
                scene_pi: torch.Tensor,
                agent_batch: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute scene-level classification loss L_cls
        
        Args:
            pred_loc: [K, A, T, 2] - Location parameters for K joint scenes
            pred_scale: [K, A, T, 2] - Scale parameters for K joint scenes
            target: [A, T, 2] - Ground truth trajectories  
            scene_pi: [K] - Scene-level confidence scores
            agent_batch: [A] - Batch assignment for each agent
            mask: [A, T] - Valid mask
            
        Returns:
            loss: Classification loss for mixing coefficients
        """
        K = pred_loc.shape[0]
        
        # Compute displacement error for each joint scene to find the best one
        target_expanded = target.unsqueeze(0).expand(K, -1, -1, -1)  # [K, A, T, 2]
        displacement_errors = torch.norm(pred_loc - target_expanded, p=2, dim=-1)  # [K, A, T]
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(0).expand(K, -1, -1)  # [K, A, T]
            displacement_errors = displacement_errors * mask_expanded.float()
        
        # Sum displacement errors for each joint scene
        scene_errors = displacement_errors.sum(dim=(1, 2))  # [K]
        
        # Find the best joint scene (minimum displacement error)
        best_scene_idx = scene_errors.argmin()
        
        # Classification loss: optimize the confidence of the best scene
        scene_log_probs = F.log_softmax(scene_pi, dim=0)
        classification_loss = -scene_log_probs[best_scene_idx]
        
        return classification_loss


def compute_joint_displacement_error(pred: torch.Tensor, 
                                   target: torch.Tensor, 
                                   agent_batch: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute joint displacement error for selecting the best scene
    
    Args:
        pred: [K, A, T, 2] - Predicted positions for K joint scenes
        target: [A, T, 2] - Ground truth positions
        agent_batch: [A] - Batch assignment for each agent
        mask: [A, T] - Valid mask
        
    Returns:
        errors: [K] - Total displacement error for each joint scene
    """
    K, A, T, _ = pred.shape
    target_expanded = target.unsqueeze(0).expand(K, -1, -1, -1)  # [K, A, T, 2]
    
    # Compute L2 displacement errors
    displacement_errors = torch.norm(pred - target_expanded, p=2, dim=-1)  # [K, A, T]
    
    if mask is not None:
        mask_expanded = mask.unsqueeze(0).expand(K, -1, -1)  # [K, A, T]
        displacement_errors = displacement_errors * mask_expanded.float()
    
    # Sum over all agents and timesteps for each joint scene
    scene_errors = displacement_errors.sum(dim=(1, 2))  # [K]
    
    return scene_errors