import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedTrajectoryPrediction(nn.Module):
    def __init__(self, encoder, decoder):
        super(MaskedTrajectoryPrediction, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, trajectory, mask):
        # trajectory: [batch_size, seq_len, feature_dim]
        masked_trajectory = trajectory.clone()
        masked_trajectory[mask] = 0  # 或其他掩蔽策略

        encoded = self.encoder(masked_trajectory)
        reconstructed = self.decoder(encoded)

        loss = F.mse_loss(reconstructed[mask], trajectory[mask])
        return loss
