import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import pdb

class KCLWOPosLoss(nn.Module):
    def __init__(self, L, K, temperature=0.1, hard_neg=True):
        super().__init__()
        self.L = L  # Layer interval
        self.K = K  # Number of negative samples
        self.temperature = temperature  # Temperature parameter
        self.hard_neg = hard_neg

    def forward(self, I_embeddings, g0, g1, g2):
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
                if num_candidates < self.K:
                    # Sample with replacement
                    repeats = (self.K + num_candidates - 1) // num_candidates
                    indices = candidate_indices_tensor.repeat(repeats)[:self.K]
                else:
                    indices = candidate_indices_tensor[torch.randperm(num_candidates, device=s_i.device)[:self.K]]

                selected_indices = indices.tolist()

                # Get negative tensors
                negatives = sample_tensors[selected_indices]  # [K, hidden_dim]

                # Calculate positive similarity
                pos_sim_ij = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0).detach())

                # Calculate similarities for anchor s_i with positive s_j and negatives
                anchor_i = s_i.unsqueeze(0)  # [1, hidden_dim]
                sim_i_neg = F.cosine_similarity(anchor_i, negatives)  # [K]

                # Calculate denominator: exp(pos_sim_ij / temp) + sum(exp(sim_i_neg / temp))
                numerator = torch.exp(pos_sim_ij / self.temperature)
                denominator = numerator + torch.sum(torch.exp(sim_i_neg / self.temperature))
                loss_i = -torch.log(numerator / denominator)

                # Similarly handle anchor s_j with positive s_i
                pos_sim_ji = F.cosine_similarity(g2.unsqueeze(0), g1.unsqueeze(0).detach())

                anchor_j = s_j.unsqueeze(0)
                sim_j_neg = F.cosine_similarity(anchor_j, negatives)

                denominator_j = torch.exp(pos_sim_ji / self.temperature) + torch.sum(torch.exp(sim_j_neg / self.temperature))
                loss_j = -torch.log(torch.exp(pos_sim_ji / self.temperature) / denominator_j)

                total_loss += loss_i + loss_j
            else:
                # Calculate positive similarity
                # pos_sim_ij = F.cosine_similarity(s_i.unsqueeze(0), s_j.unsqueeze(0).detach())  # [1]
                pos_sim_ij = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0).detach())

                numerator = torch.exp(pos_sim_ij / self.temperature)
                denominator = numerator + torch.sum(torch.exp(F.cosine_similarity(s_i.unsqueeze(0), negatives) / self.temperature))
                loss_i = -torch.log(numerator / denominator)

                # Similarly handle anchor s_j with positive s_i
                pos_sim_ij = F.cosine_similarity(g2.unsqueeze(0), g1.unsqueeze(0).detach())

                denominator_j = torch.exp(pos_sim_ji / self.temperature) + torch.sum(torch.exp(F.cosine_similarity(s_j.unsqueeze(0), negatives) / self.temperature))
                loss_j = -torch.log(torch.exp(pos_sim_ji / self.temperature) / denominator_j)

                total_loss += (loss_i + loss_j)

        total_loss = torch.mean(total_loss)
        # Average loss
        # total_loss /= len(positive_pairs) * 2  # Each pair contributes two loss terms
        return total_loss


class KCLWONegLoss(nn.Module):

    def __init__(self, L, K, temperature=0.1, hard_neg=True):
        super().__init__()
        self.L = L  # Layer interval
        self.K = K  # Number of negative samples
        self.temperature = temperature  # Temperature parameter
        self.hard_neg = hard_neg

    def forward(self, I_embeddings, g0, g1, g2):
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

        N = g0.shape[0]
        candidate_neg_indices = list(range(N))
        candidate_neg_indices_tensor = torch.tensor(candidate_neg_indices, device=g0.device)

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
            num_neg_candidates = len(candidate_neg_indices_tensor)

            if num_candidates == 0:
                continue

            if self.hard_neg:
                if num_candidates < self.K:
                    # Sample with replacement
                    repeats = (self.K + num_candidates - 1) // num_candidates
                    indices = candidate_indices_tensor.repeat(repeats)[:self.K]
                    neg_indices1 = candidate_neg_indices_tensor[torch.randperm(num_neg_candidates, device=s_i.device)[:self.K]]
                    neg_indices2 = candidate_neg_indices_tensor[torch.randperm(num_neg_candidates, device=s_i.device)[:self.K]]
                else:
                    indices = candidate_indices_tensor[torch.randperm(num_candidates, device=s_i.device)[:self.K]]
                    neg_indices1 = candidate_neg_indices_tensor[torch.randperm(num_neg_candidates, device=s_i.device)[:self.K]]
                    neg_indices2 = candidate_neg_indices_tensor[torch.randperm(num_neg_candidates, device=s_i.device)[:self.K]]

                selected_indices = indices.tolist()
                selected_neg_indices_1 = neg_indices1.tolist()
                selected_neg_indices_2 = neg_indices2.tolist()

                # Calculate positive similarity
                pos_sim_ij = F.cosine_similarity(s_i.unsqueeze(0), s_j.unsqueeze(0).detach())

                # Get negative tensors
                negatives = g2[selected_neg_indices_2]              # [K, hidden_dim]
                anchor_i = g1[selected_neg_indices_1].unsqueeze(0)  # [1, hidden_dim]
                sim_i_neg = F.cosine_similarity(anchor_i, negatives)  # [K]

                # Calculate denominator: exp(pos_sim_ij / temp) + sum(exp(sim_i_neg / temp))
                numerator = torch.exp(pos_sim_ij / self.temperature)
                denominator = numerator + torch.sum(torch.exp(sim_i_neg / self.temperature))
                loss_i = -torch.log(numerator / denominator)

                # Similarly handle anchor s_j with positive s_i
                pos_sim_ji = F.cosine_similarity(s_j.unsqueeze(0).detach(), s_i.unsqueeze(0))

                # anchor_j = s_j.unsqueeze(0)
                negatives = g1[selected_neg_indices_1]              # [K, hidden_dim]
                anchor_j = g2[selected_neg_indices_2].unsqueeze(0)  # [1, hidden_dim]
                sim_j_neg = F.cosine_similarity(anchor_j, negatives)

                denominator_j = torch.exp(pos_sim_ji / self.temperature) + torch.sum(torch.exp(sim_j_neg / self.temperature))
                loss_j = -torch.log(torch.exp(pos_sim_ji / self.temperature) / denominator_j)

                total_loss += loss_i + loss_j
            else:
                # Calculate positive similarity
                pos_sim_ij = F.cosine_similarity(s_i.unsqueeze(0), s_j.unsqueeze(0).detach())  # [1]
                numerator = torch.exp(pos_sim_ij / self.temperature)
                denominator = numerator + torch.sum(torch.exp(F.cosine_similarity(s_i.unsqueeze(0), negatives) / self.temperature))
                loss_i = -torch.log(numerator / denominator)

                # Similarly handle anchor s_j with positive s_i
                pos_sim_ji = F.cosine_similarity(s_j.unsqueeze(0).detach(), s_i.unsqueeze(0))
                denominator_j = torch.exp(pos_sim_ji / self.temperature) + torch.sum(torch.exp(F.cosine_similarity(s_j.unsqueeze(0), negatives) / self.temperature))
                loss_j = -torch.log(torch.exp(pos_sim_ji / self.temperature) / denominator_j)

                total_loss += (loss_i + loss_j)

        total_loss = torch.mean(total_loss)
        return total_loss