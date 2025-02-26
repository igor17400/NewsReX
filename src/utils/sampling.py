from typing import Dict, List, Optional

import numpy as np
from omegaconf import DictConfig


class ImpressionSampler:
    """Handles sampling strategies for news impressions."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.max_length = cfg.max_impressions_length
        self.strategy = cfg.strategy
        np.random.seed(cfg.random_seed)

    def sample_impressions(
        self,
        impressions: List[str],
        news_info: Optional[Dict] = None,
        timestamp: Optional[str] = None,
    ) -> List[List[str]]:
        """Sample impressions based on configured strategy.

        Args:
            impressions: List of impression IDs in format "newsID-clickStatus"
            news_info: Dictionary containing news metadata (for topic-based sampling)
            timestamp: Timestamp of the impression (for temporal sampling)

        Returns:
            List of impression groups, where each group contains one positive sample
            and (max_length-1) negative samples
        """
        # Split impressions into positives and negatives
        positives = []
        negatives = []
        for imp in impressions:
            news_id, label = imp.split("-")
            if int(label) == 1:
                positives.append(imp)
            else:
                negatives.append(imp)

        # If no positives, return original list up to max_length
        if not positives:
            if len(impressions) <= self.max_length:
                return [impressions]
            return [impressions[: self.max_length]]

        # Number of negatives needed per positive
        neg_per_pos = self.max_length - 1

        # Sample groups
        sampled_groups = []
        for pos in positives:
            if len(negatives) <= neg_per_pos:
                # If not enough negatives, use all available
                group = [pos] + negatives
                # Pad if necessary
                if len(group) < self.max_length:
                    group += [negatives[0]] * (self.max_length - len(group))
                sampled_groups.append(group)
            else:
                # Sample negatives based on strategy
                if self.strategy == "random":
                    sampled_negs = self._random_sample_negatives(negatives, neg_per_pos)
                else:
                    # Default to random for now
                    sampled_negs = self._random_sample_negatives(negatives, neg_per_pos)

                sampled_groups.append([pos] + sampled_negs)

        return sampled_groups

    def _random_sample_negatives(self, negatives: List[str], n: int) -> List[str]:
        """Randomly sample n negatives."""
        return list(np.random.choice(negatives, size=n, replace=len(negatives) < n))

    def _random_sample(self, impressions: List[str]) -> List[str]:
        """Random sampling strategy."""
        return np.random.choice(impressions, size=self.max_length, replace=self.cfg.random.replace).tolist()

    def _topic_diverse_sample(self, impressions: List[str], news_info: Dict) -> List[str]:
        """Topic-diverse sampling strategy."""
        # Implementation for topic diversity
        categories = [news_info[imp]["category"] for imp in impressions]
        unique_cats = set(categories)

        # Ensure minimum category diversity
        sampled = []
        for cat in unique_cats:
            cat_impressions = [imp for imp, c in zip(impressions, categories) if c == cat]
            weight = self.cfg.topic_diverse.category_weights.get(cat, 1.0)
            n_samples = max(1, int(self.max_length * weight / len(unique_cats)))
            if cat_impressions:
                sampled.extend(np.random.choice(cat_impressions, size=min(n_samples, len(cat_impressions))))

        # Fill remaining slots randomly
        if len(sampled) < self.max_length:
            remaining = [imp for imp in impressions if imp not in sampled]
            sampled.extend(np.random.choice(remaining, size=self.max_length - len(sampled)))

        return sampled[: self.max_length]

    # Add other sampling strategies as needed
