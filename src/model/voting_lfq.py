from math import log2
import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy_loss(
    logits,
    mask=None,
    temperature=0.01,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
    eps=1e-5,
):
    """
    Entropy loss of unnormalized logits
    logits: Affinities are over the last dimension, [batch, seq, codebook_size]
    mask: True for valid tokens, False for padding tokens, [batch, seq]
    """
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + eps, -1)

    if mask is not None:
        # Expand mask dimensions to match probs
        mask_expanded = mask
        for _ in range(probs.ndim - mask.ndim):
            mask_expanded = mask_expanded.unsqueeze(-1)

        # Calculate masked average probabilities
        masked_probs = probs * mask_expanded
        avg_probs = masked_probs.sum(tuple(range(mask.ndim))) / mask.sum()
    else:
        # Average over all dimensions except the last one
        avg_probs = probs.mean(tuple(range(probs.ndim - 1)))

    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    if mask is not None:
        # Calculate average of masked sample entropy
        masked_sample_entropy = sample_entropy * mask
        sample_entropy = masked_sample_entropy.sum() / mask.sum()
    else:
        sample_entropy = torch.mean(sample_entropy)

    loss = (sample_minimization_weight * sample_entropy) - (
        batch_maximization_weight * avg_entropy
    )

    return sample_entropy, avg_entropy, loss


class VotingLFQ(nn.Module):
    """Voting Lookup Free Quantization (Voting-LFQ)"""
    def __init__(
        self,
        *,
        dim = None,
        codebook_size = None,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
        projection_has_bias = True,
        codebook_scale = 1.0,
        num_voters: int = 5,
        num_clean_input: int = 3,
    ):
        super().__init__()

        # Check parameters
        if dim is None or codebook_size is None:
            raise ValueError('dim and codebook_size must be specified for LFQ')
        if num_voters < 1:
            raise ValueError('num_voters must be at least 1')
        if num_clean_input < 0:
            raise ValueError('num_clean_input must be non-negative')

        if not log2(codebook_size).is_integer():
            raise ValueError(f'LFQ codebook size must be a power of 2')

        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_dim = int(log2(self.codebook_size))
        self.has_projections = self.dim != self.codebook_dim
        self.num_voters = num_voters
        self.num_clean_input = min(num_voters, num_clean_input)
        self.codebook_scale = codebook_scale

        self.project_in = nn.ModuleList([
            nn.Linear(self.dim, self.codebook_dim, bias=projection_has_bias) if self.has_projections else nn.Identity()
            for _ in range(num_voters)
        ])
        self.project_out = nn.Linear(self.codebook_dim, self.dim, bias=projection_has_bias) if self.has_projections \
            else nn.Identity()

        # Entropy loss weights
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

        # Weights for converting binary bits to indices [1, 2, 4, 8, 16, ...]
        self.register_buffer('bit_weights', 2 ** torch.arange(self.codebook_dim), persistent=False)
        self.register_buffer('zero', torch.tensor(0.), persistent=False)

        all_codes = torch.arange(self.codebook_size)
        bits = self.indices_to_bits(all_codes)
        codebook = bits * 2.0 - 1.0  # Convert to {-1, 1}

        self.register_buffer('codebook', codebook, persistent=False)

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_bits(self, x):
        """
        Convert indices to big endian bits
        x: long tensor of indices
        returns: big endian bits
        """
        bit_weights = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
        # x is now big endian bits, last dimension is bits
        x = (x.unsqueeze(-1) & bit_weights) != 0
        return x

    def get_codebook_entry(self, x, bhwc, order):
        """Get codebook entries"""
        bit_weights = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)

        x = (x.unsqueeze(-1) & bit_weights) != 0
        x = x * 2.0 - 1.0  # Convert back to float

        # Reshape dimensions
        b, h, w, c = bhwc
        x = x.view(b, h, w, c)
        x = x.permute(0, 3, 1, 2)  # b h w c -> b c h w
        return x

    def bits_to_indices(self, bits):
        """
        Convert bits to indices
        bits: bool tensor of big endian bits, last dimension is bit dimension
        returns: indices, long integers from 0 to self.codebook_size
        """
        assert bits.shape[-1] == self.codebook_dim
        bit_weights = 2 ** torch.arange(
            0,
            self.codebook_dim,
            1,
            dtype=torch.long,
            device=bits.device,
        )
        return (bits * bit_weights).sum(-1)

    def decode(self, x):
        """
        Decoding function
        x: [batch_size, seq_len] long tensor containing codebook indices from 0 to self.codebook_size
        """
        x = self.indices_to_bits(x)
        # Convert to float type
        x = x.to(self.dtype)
        # Convert to -1 or 1
        x = x * 2 - 1
        return x

    def forward(
        self,
        x,
        x_noise = None,
        mask = None,
    ):
        """
        Forward pass
        x: input tensor of clean audio, with shape [batch_size, seq_len, feature_dim]
        x_noise: input tensor of noise audio, with the same shape
        """
        # Use all clean audio during inference
        if not self.training or x_noise is None:
            x_noise = x

        if mask is not None:
            mask = mask.bool()

        # Check input dimensions
        assert x.ndim == 3, f"Expected 3D input [batch, seq, dim], got {x.ndim}D input with shape {x.shape}"

        batch_size, seq_len, feature_dim = x.shape

        # Collect results from all voters
        voter_x_i = []
        voter_quantized = []
        voter_loss_breakdown = {}

        # Randomly select n_clean inputs as clean audio, others use noisy audio
        clean_indices = set(torch.randperm(self.num_voters, device=x.device)[:self.num_clean_input].cpu().tolist())
        codebook_value = x.new_tensor(self.codebook_scale)

        # Perform quantization for each voter
        for i, project_in_i in enumerate(self.project_in):
            # Choose clean or noise input based on index
            input_x = x if i in clean_indices else x_noise

            # 1. Project to latent space
            if self.has_projections:
                x_i = project_in_i(input_x)
            else:
                x_i = input_x

            # 2. Quantization step - use codebook_scale for quantized values
            quantized = torch.where(x_i > 0, codebook_value, -codebook_value)

            # Use straight-through gradients
            quantized = x_i + (quantized - x_i).detach()

            # 3. Append the quantized result for this voter to the list
            voter_x_i.append(x_i)
            voter_quantized.append(quantized)

            # Entropy auxiliary loss
            if self.training:
                # Use normalized codebook for distance calculation
                codebook = self.codebook.to(device=x_i.device, dtype=x_i.dtype)

                # Calculate logits: 2 * (x @ codebook.T)
                # x: [batch, seq, codebook_dim], codebook: [codebook_size, codebook_dim]
                x_flat = x_i.view(-1, self.codebook_dim)  # [batch*seq, codebook_dim]
                logits = 2 * torch.matmul(x_flat, codebook.t())  # [batch*seq, codebook_size]
                logits = logits.view(batch_size, seq_len, self.codebook_size)  # [batch, seq, codebook_size]

                per_sample_entropy, codebook_entropy, entropy_aux_loss = entropy_loss(
                    logits = logits,
                    mask = mask,
                    sample_minimization_weight = self.sample_minimization_weight,
                    batch_maximization_weight = self.batch_maximization_weight
                )
            else:
                per_sample_entropy = codebook_entropy = self.zero
                entropy_aux_loss = self.zero

            local_loss_breakdown = dict(
                per_sample_entropy = per_sample_entropy,
                codebook_entropy = codebook_entropy,
                entropy_aux_loss = entropy_aux_loss,
            )

            for key, value in local_loss_breakdown.items():
                if key not in voter_loss_breakdown:
                    voter_loss_breakdown[key] = []
                voter_loss_breakdown[key].append(value)

        # 4. Calculate final quantized voting results
        sum_quantized = torch.stack(voter_quantized, dim=0).sum(dim=0)  # [batch, seq, codebook_dim]
        final_quantized = sum_quantized / self.num_voters  # [batch, seq, codebook_dim]

        # Convert majority-voted signs to integer token IDs with bit weights [1, 2, 4, ...].
        indices = self.bits_to_indices(final_quantized > 0)  # [batch, seq]

        # During inference, take sign again to ensure quantized results are exactly 1 or -1
        if not self.training:
            final_quantized = torch.where(final_quantized > 0, codebook_value, -codebook_value)

        if self.has_projections:
            out = self.project_out(final_quantized)
        else:
            out = final_quantized

        # Calculate commitment loss
        if self.training:
            voter_loss_breakdown["commitment_loss"] = []
            for x_i in voter_x_i:
                commit_loss = F.mse_loss(x_i, final_quantized.detach(), reduction='none')
                if mask is not None:
                    commit_loss = commit_loss[mask]
                commit_loss = commit_loss.mean()
                voter_loss_breakdown['commitment_loss'].append(commit_loss)
        else:
            commit_loss = self.zero
            voter_loss_breakdown['commitment_loss'] = [commit_loss]

        # Calculate consensus loss for all voters
        consensus_loss = self.zero
        if self.training and self.num_voters > 1:
            # Collect projection outputs x_i from all voters
            all_voter_x_i = voter_x_i  # Directly use x_i from all voters

            # Calculate mean of all x_i as consensus target
            all_voter_x_i_stack = torch.stack(all_voter_x_i, dim=0)  # [num_voters, batch, seq, codebook_dim]
            consensus_target = all_voter_x_i_stack.mean(dim=0)  # [batch, seq, codebook_dim]

            # Calculate MSE loss between each voter's x_i and consensus target
            consensus_losses = []
            for voter_x_i in all_voter_x_i:
                if mask is not None:
                    mse_loss = F.mse_loss(voter_x_i, consensus_target.detach(), reduction='none')
                    mse_loss = mse_loss[mask].mean()  # Only calculate loss for valid positions
                else:
                    mse_loss = F.mse_loss(voter_x_i, consensus_target.detach())
                consensus_losses.append(mse_loss)

            # Average over voters so the configured loss weight is comparable for N=3/5.
            consensus_loss = torch.stack(consensus_losses).mean()

        # Calculate final losses
        loss_breakdown = {key: torch.stack(value).mean() if value else None
                          for key, value in voter_loss_breakdown.items()}
        loss_breakdown['consensus_loss'] = consensus_loss

        return out, indices, loss_breakdown

if __name__ == "__main__":
    quantizer = VotingLFQ(
        codebook_size = 2**13,              # codebook size, must be a power of 2
        dim = 1280,                         # this is the input feature dimension
        sample_minimization_weight = 1.0,
        batch_maximization_weight = 1.0,
    )

    # dummy input
    seq_feats = torch.randn(2, 100, 1280)  # [batch_size, seq_len, feature_dim]
    noise_std = 0.1
    seq_feats_noise = seq_feats + torch.randn_like(seq_feats) * noise_std  # [batch_size, seq_len, feature_dim]

    print(f"original signal std: {seq_feats.std():.4f}")
    print(f"noise std: {noise_std}")
    print(f"SNR: {seq_feats.std() / noise_std:.2f}")

    mask = torch.tensor([[True] * 23 + [False] * 77, [True] * 99 + [False] * 1])
    hidden_quantized, indices, loss_breakdown = quantizer(seq_feats, seq_feats_noise, mask)

    assert seq_feats.shape == hidden_quantized.shape
    print(f"input shape:")
    print(f"\tclean: {seq_feats.shape}")
    print(f"\tnoise: {seq_feats_noise.shape}")
    print(f"output shape: {hidden_quantized.shape}")
    print(f"indices shape: {indices.shape}")
    print(f"loss_breakdown:")
    for k, v in loss_breakdown.items():
        print(f"\t{k}:", v)
    print(f"Voting-LFQ sequence test passed!")
