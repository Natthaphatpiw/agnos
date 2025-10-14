import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import List, Dict, Tuple
import numpy as np


class SymptomRecommender(nn.Module):
    """
    Hybrid symptom recommendation model combining:
    - Content-based filtering (symptom embeddings + cosine similarity)
    - Collaborative filtering (NCF: GMF + MLP)
    - Graph neural network (GAT for symptom co-occurrence)
    """

    def __init__(
        self,
        num_symptoms: int,
        num_patients: int,
        symptom_embed_dim: int = 128,
        demog_embed_dim: int = 16,
        gat_heads: int = 4,
        gat_hidden_dim: int = 128,
        mlp_layers: List[int] = [256, 128, 64, 32],
        dropout: float = 0.3
    ):
        super(SymptomRecommender, self).__init__()

        self.num_symptoms = num_symptoms
        self.num_patients = num_patients
        self.symptom_embed_dim = symptom_embed_dim

        # Symptom embeddings
        self.symptom_embeddings = nn.Embedding(num_symptoms, symptom_embed_dim)

        # Patient embeddings for CF
        self.patient_embeddings = nn.Embedding(num_patients, symptom_embed_dim)

        # Demographics encoder (gender: 2 one-hot + age_bin: 4 one-hot = 6 input)
        self.demog_encoder = nn.Sequential(
            nn.Linear(6, demog_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GAT layers for symptom co-occurrence graph
        self.gat1 = GATConv(symptom_embed_dim, gat_hidden_dim, heads=gat_heads, dropout=dropout)
        self.gat2 = GATConv(gat_hidden_dim * gat_heads, symptom_embed_dim, heads=1, concat=False, dropout=dropout)

        # GMF (Generalized Matrix Factorization) for CF
        self.gmf_linear = nn.Linear(symptom_embed_dim, 1)

        # MLP for CF
        mlp_input_dim = symptom_embed_dim * 2 + demog_embed_dim
        mlp = []
        prev_dim = mlp_input_dim
        for hidden_dim in mlp_layers:
            mlp.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        mlp.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*mlp)

        # Final fusion weights
        self.fusion_weights = nn.Parameter(torch.tensor([0.4, 0.4, 0.2]))  # CF, CB, Graph

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with Xavier uniform"""
        nn.init.xavier_uniform_(self.symptom_embeddings.weight)
        nn.init.xavier_uniform_(self.patient_embeddings.weight)

    def forward_gat(self, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        Propagate symptom embeddings through GAT
        Args:
            edge_index: [2, num_edges] graph connectivity
            edge_attr: [num_edges, feat_dim] optional edge features
        Returns:
            updated_embeddings: [num_symptoms, embed_dim]
        """
        x = self.symptom_embeddings.weight  # [num_symptoms, embed_dim]
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.gat2(x, edge_index)
        return x

    def encode_demographics(self, gender: torch.Tensor, age_bin: torch.Tensor) -> torch.Tensor:
        """
        Encode demographics to embedding
        Args:
            gender: [batch_size, 2] one-hot
            age_bin: [batch_size, 4] one-hot
        Returns:
            demog_embed: [batch_size, demog_embed_dim]
        """
        demog_input = torch.cat([gender, age_bin], dim=-1)  # [batch_size, 6]
        return self.demog_encoder(demog_input)

    def compute_cf_score(
        self,
        patient_idx: torch.Tensor,
        symptom_idx: torch.Tensor,
        demog_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute collaborative filtering score using GMF and MLP
        Args:
            patient_idx: [batch_size] patient indices
            symptom_idx: [batch_size] symptom indices
            demog_embed: [batch_size, demog_embed_dim]
        Returns:
            gmf_score: [batch_size, 1]
            mlp_score: [batch_size, 1]
        """
        patient_embed = self.patient_embeddings(patient_idx)  # [batch_size, embed_dim]
        symptom_embed = self.symptom_embeddings(symptom_idx)  # [batch_size, embed_dim]

        # GMF: element-wise product
        gmf_vector = patient_embed * symptom_embed  # [batch_size, embed_dim]
        gmf_score = self.gmf_linear(gmf_vector)  # [batch_size, 1]

        # MLP: concatenation
        mlp_input = torch.cat([patient_embed, symptom_embed, demog_embed], dim=-1)
        mlp_score = self.mlp(mlp_input)  # [batch_size, 1]

        # Combine GMF and MLP
        cf_score = gmf_score + mlp_score  # [batch_size, 1]
        return cf_score

    def compute_cb_score(
        self,
        query_symptoms: torch.Tensor,
        candidate_symptoms: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute content-based score using cosine similarity
        Args:
            query_symptoms: [batch_size, num_query_symptoms] indices of input symptoms
            candidate_symptoms: [batch_size, num_candidates] indices of candidate symptoms
        Returns:
            cb_scores: [batch_size, num_candidates]
        """
        # Get embeddings
        query_embeds = self.symptom_embeddings(query_symptoms)  # [batch_size, num_query, embed_dim]
        candidate_embeds = self.symptom_embeddings(candidate_symptoms)  # [batch_size, num_candidates, embed_dim]

        # Average query embeddings
        query_avg = query_embeds.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]

        # Cosine similarity
        query_norm = F.normalize(query_avg, p=2, dim=-1)
        candidate_norm = F.normalize(candidate_embeds, p=2, dim=-1)
        cb_scores = (query_norm * candidate_norm).sum(dim=-1)  # [batch_size, num_candidates]

        return cb_scores

    def compute_graph_score(
        self,
        query_symptoms: torch.Tensor,
        candidate_symptoms: torch.Tensor,
        gat_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute graph-based score using GAT-propagated embeddings
        Args:
            query_symptoms: [batch_size, num_query_symptoms]
            candidate_symptoms: [batch_size, num_candidates]
            gat_embeddings: [num_symptoms, embed_dim]
        Returns:
            graph_scores: [batch_size, num_candidates]
        """
        query_embeds = gat_embeddings[query_symptoms]  # [batch_size, num_query, embed_dim]
        candidate_embeds = gat_embeddings[candidate_symptoms]  # [batch_size, num_candidates, embed_dim]

        query_avg = query_embeds.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]

        # Cosine similarity with GAT embeddings
        query_norm = F.normalize(query_avg, p=2, dim=-1)
        candidate_norm = F.normalize(candidate_embeds, p=2, dim=-1)
        graph_scores = (query_norm * candidate_norm).sum(dim=-1)

        return graph_scores

    def forward(
        self,
        patient_idx: torch.Tensor,
        gender: torch.Tensor,
        age_bin: torch.Tensor,
        query_symptoms: torch.Tensor,
        edge_index: torch.Tensor,
        candidate_symptoms: torch.Tensor = None,
        return_all_scores: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for scoring symptom recommendations
        Args:
            patient_idx: [batch_size] patient index
            gender: [batch_size, 2] one-hot gender
            age_bin: [batch_size, 4] one-hot age bin
            query_symptoms: [batch_size, num_query] input symptom indices
            edge_index: [2, num_edges] graph edges
            candidate_symptoms: [batch_size, num_candidates] or None (use all)
            return_all_scores: whether to return individual scores
        Returns:
            final_scores: [batch_size, num_candidates]
        """
        batch_size = patient_idx.shape[0]

        # Encode demographics
        demog_embed = self.encode_demographics(gender, age_bin)

        # Propagate GAT
        gat_embeddings = self.forward_gat(edge_index)

        # If no candidates specified, use all symptoms
        if candidate_symptoms is None:
            candidate_symptoms = torch.arange(self.num_symptoms, device=patient_idx.device).unsqueeze(0).expand(batch_size, -1)

        num_candidates = candidate_symptoms.shape[1]

        # Expand patient_idx and demog_embed for all candidates
        patient_idx_expanded = patient_idx.unsqueeze(1).expand(-1, num_candidates).reshape(-1)
        demog_embed_expanded = demog_embed.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, demog_embed.shape[-1])
        candidate_symptoms_flat = candidate_symptoms.reshape(-1)

        # Compute CF scores
        cf_scores = self.compute_cf_score(patient_idx_expanded, candidate_symptoms_flat, demog_embed_expanded)
        cf_scores = cf_scores.reshape(batch_size, num_candidates)

        # Compute CB scores
        cb_scores = self.compute_cb_score(query_symptoms, candidate_symptoms)

        # Compute Graph scores
        graph_scores = self.compute_graph_score(query_symptoms, candidate_symptoms, gat_embeddings)

        # Normalize scores to [0, 1]
        cf_scores_norm = torch.sigmoid(cf_scores)
        cb_scores_norm = (cb_scores + 1) / 2  # cosine is in [-1, 1]
        graph_scores_norm = (graph_scores + 1) / 2

        # Weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        final_scores = (
            weights[0] * cf_scores_norm +
            weights[1] * cb_scores_norm +
            weights[2] * graph_scores_norm
        )

        if return_all_scores:
            return final_scores, cf_scores_norm, cb_scores_norm, graph_scores_norm

        return final_scores

    def recommend(
        self,
        patient_idx: int,
        gender: str,
        age: int,
        query_symptoms: List[str],
        symptom_to_idx: Dict[str, int],
        idx_to_symptom: Dict[int, str],
        edge_index: torch.Tensor,
        age_bins: List[int],
        top_k: int = 5,
        device: str = 'cpu'
    ) -> List[str]:
        """
        Recommend top-k symptoms for a given query
        Args:
            patient_idx: patient index (use 0 for new patient)
            gender: 'male' or 'female'
            age: patient age
            query_symptoms: list of symptom names
            symptom_to_idx: dict mapping symptom name to index
            idx_to_symptom: dict mapping index to symptom name
            edge_index: graph edges
            age_bins: list of age bin edges
            top_k: number of recommendations
            device: 'cpu' or 'cuda'
        Returns:
            recommendations: list of symptom names
        """
        self.eval()
        with torch.no_grad():
            # Encode gender
            gender_vec = torch.zeros(1, 2, device=device)
            gender_vec[0, 0 if gender.lower() == 'male' else 1] = 1.0

            # Encode age bin
            age_bin_vec = torch.zeros(1, 4, device=device)
            bin_idx = np.digitize(age, age_bins) - 1
            bin_idx = max(0, min(3, bin_idx))
            age_bin_vec[0, bin_idx] = 1.0

            # Encode query symptoms
            query_indices = []
            for sym in query_symptoms:
                if sym in symptom_to_idx:
                    query_indices.append(symptom_to_idx[sym])

            if not query_indices:
                # Cold start: return most popular symptoms (highest embedding norms)
                embedding_norms = torch.norm(self.symptom_embeddings.weight, dim=1)
                top_indices = torch.topk(embedding_norms, k=top_k).indices
                return [idx_to_symptom[idx.item()] for idx in top_indices]

            query_tensor = torch.tensor([query_indices], device=device)
            patient_idx_tensor = torch.tensor([patient_idx], device=device)

            # Get all candidate symptoms (exclude query symptoms)
            all_symptom_indices = list(range(self.num_symptoms))
            candidate_indices = [idx for idx in all_symptom_indices if idx not in query_indices]
            candidate_tensor = torch.tensor([candidate_indices], device=device)

            # Forward pass
            scores = self.forward(
                patient_idx_tensor,
                gender_vec,
                age_bin_vec,
                query_tensor,
                edge_index,
                candidate_tensor
            )

            # Get top-k
            top_k_actual = min(top_k, len(candidate_indices))
            top_scores, top_local_indices = torch.topk(scores[0], k=top_k_actual)
            top_indices = [candidate_indices[idx.item()] for idx in top_local_indices]

            recommendations = [idx_to_symptom[idx] for idx in top_indices]

            return recommendations
