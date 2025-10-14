"""
Lightweight model architecture for small datasets (< 10,000 samples)
This model is designed to prevent overfitting on limited data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import numpy as np


class LightweightSymptomRecommender(nn.Module):
    """
    Simplified symptom recommender with much fewer parameters
    Designed for datasets with 1,000-10,000 samples

    Key changes from original:
    - Smaller embedding dimensions (32 vs 128)
    - No complex MLP (just 1 layer)
    - No GAT (too complex for small data)
    - Higher dropout (0.5 vs 0.3)
    - Simpler fusion (just CF + CB, no graph)
    """

    def __init__(
        self,
        num_symptoms: int,
        num_patients: int,
        symptom_embed_dim: int = 32,  # Reduced from 128
        demog_embed_dim: int = 8,     # Reduced from 16
        dropout: float = 0.5           # Increased from 0.3
    ):
        super(LightweightSymptomRecommender, self).__init__()

        self.num_symptoms = num_symptoms
        self.num_patients = num_patients
        self.symptom_embed_dim = symptom_embed_dim

        # Symptom embeddings with dropout
        self.symptom_embeddings = nn.Embedding(num_symptoms, symptom_embed_dim)
        self.symptom_dropout = nn.Dropout(dropout)

        # Patient embeddings with dropout
        self.patient_embeddings = nn.Embedding(num_patients, symptom_embed_dim)
        self.patient_dropout = nn.Dropout(dropout)

        # Demographics encoder (simpler)
        self.demog_encoder = nn.Sequential(
            nn.Linear(6, demog_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Simple collaborative filtering (just dot product + bias)
        self.cf_bias = nn.Parameter(torch.zeros(1))

        # Simple MLP for fusion (1 layer only)
        fusion_input_dim = symptom_embed_dim + demog_embed_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        # Fusion weights (only CF and CB, no graph)
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with smaller values to prevent overfitting"""
        nn.init.normal_(self.symptom_embeddings.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.patient_embeddings.weight, mean=0.0, std=0.01)

    def compute_cf_score(
        self,
        patient_idx: torch.Tensor,
        symptom_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Simple collaborative filtering using dot product
        No complex MLP to reduce overfitting
        """
        patient_embed = self.patient_embeddings(patient_idx)
        patient_embed = self.patient_dropout(patient_embed)

        symptom_embed = self.symptom_embeddings(symptom_idx)
        symptom_embed = self.symptom_dropout(symptom_embed)

        # Simple dot product
        cf_score = (patient_embed * symptom_embed).sum(dim=-1, keepdim=True) + self.cf_bias

        return cf_score

    def compute_cb_score(
        self,
        query_symptoms: torch.Tensor,
        candidate_symptoms: torch.Tensor
    ) -> torch.Tensor:
        """
        Content-based score using cosine similarity
        Same as original but with smaller embeddings
        """
        query_embeds = self.symptom_embeddings(query_symptoms)
        query_embeds = self.symptom_dropout(query_embeds)

        candidate_embeds = self.symptom_embeddings(candidate_symptoms)
        candidate_embeds = self.symptom_dropout(candidate_embeds)

        query_avg = query_embeds.mean(dim=1, keepdim=True)

        query_norm = F.normalize(query_avg, p=2, dim=-1)
        candidate_norm = F.normalize(candidate_embeds, p=2, dim=-1)

        cb_scores = (query_norm * candidate_norm).sum(dim=-1)

        return cb_scores

    def forward(
        self,
        patient_idx: torch.Tensor,
        gender: torch.Tensor,
        age_bin: torch.Tensor,
        query_symptoms: torch.Tensor,
        candidate_symptoms: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass - simplified version
        No edge_index needed (no GAT)
        """
        batch_size = patient_idx.shape[0]

        # Encode demographics
        demog_input = torch.cat([gender, age_bin], dim=-1)
        demog_embed = self.demog_encoder(demog_input)

        # If no candidates specified, use all symptoms
        if candidate_symptoms is None:
            candidate_symptoms = torch.arange(
                self.num_symptoms,
                device=patient_idx.device
            ).unsqueeze(0).expand(batch_size, -1)

        num_candidates = candidate_symptoms.shape[1]

        # Expand for all candidates
        patient_idx_expanded = patient_idx.unsqueeze(1).expand(-1, num_candidates).reshape(-1)
        candidate_symptoms_flat = candidate_symptoms.reshape(-1)

        # Compute CF scores
        cf_scores = self.compute_cf_score(patient_idx_expanded, candidate_symptoms_flat)
        cf_scores = cf_scores.reshape(batch_size, num_candidates)

        # Compute CB scores
        cb_scores = self.compute_cb_score(query_symptoms, candidate_symptoms)

        # Normalize scores
        cf_scores_norm = torch.sigmoid(cf_scores)
        cb_scores_norm = (cb_scores + 1) / 2

        # Weighted fusion (only CF and CB)
        weights = F.softmax(self.fusion_weights, dim=0)
        final_scores = (
            weights[0] * cf_scores_norm +
            weights[1] * cb_scores_norm
        )

        return final_scores

    def recommend(
        self,
        patient_idx: int,
        gender: str,
        age: int,
        query_symptoms: List[str],
        symptom_to_idx: Dict[str, int],
        idx_to_symptom: Dict[int, str],
        age_bins: List[int],
        top_k: int = 5,
        device: str = 'cpu'
    ) -> List[str]:
        """
        Recommend top-k symptoms
        Same interface as original model
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
                # Cold start: return most similar to random symptoms
                embedding_norms = torch.norm(self.symptom_embeddings.weight, dim=1)
                top_indices = torch.topk(embedding_norms, k=top_k).indices
                return [idx_to_symptom[idx.item()] for idx in top_indices]

            query_tensor = torch.tensor([query_indices], device=device)
            patient_idx_tensor = torch.tensor([patient_idx], device=device)

            # Get candidate symptoms (exclude query)
            all_symptom_indices = list(range(self.num_symptoms))
            candidate_indices = [idx for idx in all_symptom_indices if idx not in query_indices]
            candidate_tensor = torch.tensor([candidate_indices], device=device)

            # Forward pass
            scores = self.forward(
                patient_idx_tensor,
                gender_vec,
                age_bin_vec,
                query_tensor,
                candidate_tensor
            )

            # Get top-k
            top_k_actual = min(top_k, len(candidate_indices))
            top_scores, top_local_indices = torch.topk(scores[0], k=top_k_actual)
            top_indices = [candidate_indices[idx.item()] for idx in top_local_indices]

            recommendations = [idx_to_symptom[idx] for idx in top_indices]

            return recommendations


class TraditionalMLRecommender:
    """
    Traditional Machine Learning approach using Matrix Factorization
    Best for very small datasets (< 5,000 samples)
    """

    def __init__(self, n_components: int = 50):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.neighbors import NearestNeighbors

        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.knn = NearestNeighbors(n_neighbors=20, metric='cosine')
        self.fitted = False

    def fit(self, interaction_matrix: np.ndarray, symptom_to_idx: Dict[str, int]):
        """Fit the model on interaction matrix"""
        # SVD for dimensionality reduction
        self.patient_features = self.svd.fit_transform(interaction_matrix)
        self.symptom_features = self.svd.components_.T

        # KNN for finding similar patients
        self.knn.fit(self.patient_features)

        self.interaction_matrix = interaction_matrix
        self.symptom_to_idx = symptom_to_idx
        self.idx_to_symptom = {idx: sym for sym, idx in symptom_to_idx.items()}
        self.fitted = True

    def recommend(
        self,
        query_symptoms: List[str],
        top_k: int = 5
    ) -> List[str]:
        """Recommend symptoms based on query"""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Create query vector
        query_vector = np.zeros(len(self.symptom_to_idx))
        for sym in query_symptoms:
            if sym in self.symptom_to_idx:
                query_vector[self.symptom_to_idx[sym]] = 1

        # Transform to latent space
        query_features = self.svd.transform(query_vector.reshape(1, -1))

        # Find similar patients
        distances, indices = self.knn.kneighbors(query_features)

        # Aggregate symptoms from similar patients
        symptom_scores = np.zeros(len(self.symptom_to_idx))
        for idx in indices[0]:
            symptom_scores += self.interaction_matrix[idx]

        # Remove query symptoms
        for sym in query_symptoms:
            if sym in self.symptom_to_idx:
                symptom_scores[self.symptom_to_idx[sym]] = -999

        # Get top-k
        top_indices = np.argsort(symptom_scores)[::-1][:top_k]
        recommendations = [self.idx_to_symptom[idx] for idx in top_indices]

        return recommendations


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Compare model sizes
    print("Model Size Comparison:")
    print("=" * 60)

    # Original model (from model.py)
    from model import SymptomRecommender

    original_model = SymptomRecommender(
        num_symptoms=338,
        num_patients=1000,
        symptom_embed_dim=128
    )

    lightweight_model = LightweightSymptomRecommender(
        num_symptoms=338,
        num_patients=1000,
        symptom_embed_dim=32
    )

    original_params = count_parameters(original_model)
    lightweight_params = count_parameters(lightweight_model)

    print(f"Original Model:    {original_params:,} parameters")
    print(f"Lightweight Model: {lightweight_params:,} parameters")
    print(f"Reduction:         {(1 - lightweight_params/original_params)*100:.1f}%")
    print("=" * 60)

    print(f"\nParameters per sample (1,000 training samples):")
    print(f"Original:    {original_params/1000:.1f} params/sample")
    print(f"Lightweight: {lightweight_params/1000:.1f} params/sample")
    print(f"\nRecommended: < 100 params/sample for good generalization")
