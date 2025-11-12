import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class KCLLoss(nn.Module):
    def __init__(self, L, K, temperature=0.1, hard_neg=True):
        super().__init__()
        self.L = L  # Layer interval
        self.K = K  # Number of negative samples
        self.temperature = temperature  # Temperature parameter
        self.hard_neg = hard_neg

    def forward(self, I_embeddings):
        # I_embeddings is a list of embeddings for each layer, each element is [n_nodes, hidden_dim]
        if len(I_embeddings) == 0:
            return torch.tensor(0.0, device=I_embeddings[0].device)

        # Calculate sum and sum_neg for each layer
        sum_list = []
        sum_neg_list = []
        for emb in I_embeddings:
            # sum_i: [hidden_dim]
            sum_i = torch.sum(emb, dim=0)
            sum_list.append(sum_i)

            # Calculate sum_neg_i by discarding the top K embeddings with largest norms
            if emb.size(0) == 0:
                sum_neg_i = torch.zeros_like(sum_i)
            else:
                l2_norms = torch.norm(emb, p=2, dim=1)  # [n_nodes]
                sorted_indices = torch.argsort(l2_norms, descending=True)
                mask = torch.ones(emb.size(0), dtype=torch.bool, device=emb.device)
                # Discard the top K embeddings with the largest norms
                mask[sorted_indices[:self.K]] = False
                emb_neg = emb[mask]
                sum_neg_i = torch.sum(emb_neg, dim=0) if emb_neg.size(0) > 0 else torch.zeros_like(sum_i)
            sum_neg_list.append(sum_neg_i)
        
        # Combine all samples
        D = len(sum_list)
        samples = sum_list + sum_neg_list  # First D are sum_list, next D are sum_neg_list
        sample_tensors = torch.stack(samples)  # [2D, hidden_dim]
        
        # Generate positive pairs
        positive_pairs = []
        for i in range(D - self.L):
            j = i + self.L
            if j < D:
                positive_pairs.append((i, j))  # Indices in sum_list
        
        if len(positive_pairs) == 0:
            return torch.tensor(0.0, device=I_embeddings[0].device)
        
        total_loss = 0.0
        for (i, j) in positive_pairs:
            s_i = sum_list[i]
            s_j = sum_list[j]

            # Candidate negative indices: all samples except i and j
            # Indices: 0~D-1 are sum_list, D~2D-1 are sum_neg_list
            # So exclude i and j
            candidate_indices = list(range(2*D))
            try:
                candidate_indices.remove(i)
                candidate_indices.remove(j)
            except ValueError:
                pass

            if len(candidate_indices) == 0:
                continue  # No negative samples available, skip

            # Generate K negative indices
            candidate_indices_tensor = torch.tensor(candidate_indices, device=s_i.device)
            num_candidates = len(candidate_indices_tensor)

            if num_candidates == 0:
                continue

            if self.hard_neg:
                # Generate K negative indices
                candidate_indices_tensor = torch.tensor(candidate_indices, device=s_i.device)
                num_candidates = len(candidate_indices_tensor)

                if num_candidates == 0:
                    continue

                if num_candidates < self.K:
                    # Sample with replacement
                    repeats = (self.K + num_candidates - 1) // num_candidates
                    indices = candidate_indices_tensor.repeat(repeats)[:self.K]
                else:
                    indices = candidate_indices_tensor[torch.randperm(num_candidates, device=s_i.device)[:self.K]]

                selected_indices = indices.tolist()

                # Get negative tensors
                negatives = sample_tensors[selected_indices]  # [K, hidden_dim]

                # # Calculate positive similarity
                pos_sim_ij = F.cosine_similarity(s_i.unsqueeze(0), s_j.unsqueeze(0).detach())  # [1]

                # Calculate positive similarity using L2 norm
                # pos_sim_ij = -torch.norm(s_i.unsqueeze(0) - s_j.unsqueeze(0).detach(), p=2, dim=1)  # [1]

                # Calculate similarities for anchor s_i with positive s_j and negatives
                anchor_i = s_i.unsqueeze(0)  # [1, hidden_dim]
                sim_i_neg = F.cosine_similarity(anchor_i, negatives)  # [K]
                # sim_i_neg = -torch.norm(anchor_i - negatives, p=2, dim=1)  # [K]

                # Calculate denominator: exp(pos_sim_ij / temp) + sum(exp(sim_i_neg / temp))
                numerator = torch.exp(pos_sim_ij / self.temperature)
                denominator = numerator + torch.sum(torch.exp(sim_i_neg / self.temperature))
                loss_i = -torch.log(numerator / denominator)

                # Similarly handle anchor s_j with positive s_i
                pos_sim_ji = F.cosine_similarity(s_j.unsqueeze(0).detach(), s_i.unsqueeze(0))
                # pos_sim_ji = -torch.norm(s_j.unsqueeze(0).detach() - s_i.unsqueeze(0), p=2, dim=1)
                anchor_j = s_j.unsqueeze(0)
                sim_j_neg = F.cosine_similarity(anchor_j, negatives)
                # sim_j_neg = -torch.norm(anchor_j - negatives, p=2, dim=1)

                denominator_j = torch.exp(pos_sim_ji / self.temperature) + torch.sum(torch.exp(sim_j_neg / self.temperature))
                loss_j = -torch.log(torch.exp(pos_sim_ji / self.temperature) / denominator_j)

                total_loss += loss_i + loss_j
            else:
                # Calculate positive similarity
                pos_sim_ij = F.cosine_similarity(s_i.unsqueeze(0), s_j.unsqueeze(0).detach())  # [1]
                # pos_sim_ij = -torch.norm(s_i.unsqueeze(0) - s_j.unsqueeze(0).detach(), p=2, dim=1)  # [1]

                # Calculate denominator: exp(pos_sim_ij / temp)
                numerator = torch.exp(pos_sim_ij / self.temperature)
                denominator = numerator
                loss_i = -torch.log(numerator / denominator)

                # Similarly handle anchor s_j with positive s_i
                pos_sim_ji = F.cosine_similarity(s_j.unsqueeze(0).detach(), s_i.unsqueeze(0))
                # pos_sim_ji = -torch.norm(s_j.unsqueeze(0).detach() - s_i.unsqueeze(0), p=2, dim=1)
                denominator_j = torch.exp(pos_sim_ji / self.temperature)
                loss_j = -torch.log(torch.exp(pos_sim_ji / self.temperature) / denominator_j)

                total_loss += (loss_i + loss_j)

        # Average loss
        total_loss /= len(positive_pairs) * 2  # Each pair contributes two loss terms
        return total_loss