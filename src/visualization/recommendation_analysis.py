"""
Comprehensive Visualization Suite for News Recommendation System Analysis

This module provides publication-ready visualizations for analyzing news recommendation
system behavior, including score distributions, accuracy metrics, diversity analysis,
and model comparisons.

Author: NewsReX Project
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings

warnings.filterwarnings('ignore')

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RecommendationVisualizer:
    """
    A comprehensive visualization suite for analyzing news recommendation systems.
    
    This class provides methods to generate publication-ready visualizations that help
    researchers understand model behavior, recommendation quality, and system performance.
    """
    
    def __init__(self, output_dir: str = "visualization_outputs", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
            figsize: Default figure size for plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 8)
        
    def load_prediction_data(self, prediction_file: str) -> pd.DataFrame:
        """
        Load prediction data from file and parse into structured format.
        
        Args:
            prediction_file: Path to prediction file
            
        Returns:
            DataFrame with parsed prediction data
        """
        print(f"  [load_prediction_data] Loading data from {prediction_file}")
        data = []
        
        with open(prediction_file, 'r') as f:
            lines = f.readlines()
        
        print(f"  [load_prediction_data] Processing {len(lines)-1} prediction lines...")
        
        # Skip header
        for line_num, line in enumerate(lines[1:], 1):
            if line_num % 10000 == 0:
                print(f"    Processing line {line_num}/{len(lines)-1}... ({line_num/(len(lines)-1)*100:.1f}%)")
            
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                # Parse impression IDs (handle numpy array format)
                imp_id_str = parts[0].strip('[]')
                impression_ids = [str(x) for x in imp_id_str.split()]  # Keep as string for consistency
                # Parse ground truth
                ground_truth = eval(parts[1])
                # Parse prediction scores
                prediction_scores = eval(parts[2])
                
                # Create records for each impression
                for i, (imp_id, gt, score) in enumerate(zip(impression_ids, ground_truth, prediction_scores)):
                    data.append({
                        'impression_id': imp_id,  # Keep as string
                        'ground_truth': gt,
                        'prediction_score': score,
                        'position': i,
                        'session_id': len(data) // len(impression_ids)
                    })
        
        print(f"  [load_prediction_data] Created DataFrame with {len(data)} records")
        return pd.DataFrame(data)
    
    def load_news_metadata(self, news_file: str) -> pd.DataFrame:
        """
        Load news metadata from TSV file.
        
        Args:
            news_file: Path to news.tsv file
            
        Returns:
            DataFrame with news metadata
        """
        try:
            news_df = pd.read_csv(
                news_file, 
                sep='\t', 
                names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
                dtype={'news_id': str}
            )
            return news_df
        except Exception as e:
            print(f"Warning: Could not load news metadata: {e}")
            return pd.DataFrame()
    
    def load_training_summary(self, summary_file: str) -> Dict:
        """
        Load training summary JSON file.
        
        Args:
            summary_file: Path to training_run_summary.json
            
        Returns:
            Dictionary with training summary data
        """
        try:
            with open(summary_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load training summary: {e}")
            return {}
    
    def plot_score_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Plot distribution of recommendation scores.
        
        Args:
            df: DataFrame with prediction data
            save_path: Optional path to save the plot
        """
        print(f"  [plot_score_distribution] Creating score distribution plots...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Recommendation Score Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Overall score distribution
        axes[0, 0].hist(df['prediction_score'], bins=50, alpha=0.7, color=self.colors[0], edgecolor='black')
        axes[0, 0].axvline(df['prediction_score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["prediction_score"].mean():.3f}')
        axes[0, 0].axvline(df['prediction_score'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df["prediction_score"].median():.3f}')
        axes[0, 0].set_xlabel('Prediction Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Overall Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Score distribution by ground truth
        clicked_scores = df[df['ground_truth'] == 1]['prediction_score']
        not_clicked_scores = df[df['ground_truth'] == 0]['prediction_score']
        
        axes[0, 1].hist(not_clicked_scores, bins=50, alpha=0.6, label='Not Clicked', color=self.colors[1], density=True)
        axes[0, 1].hist(clicked_scores, bins=50, alpha=0.6, label='Clicked', color=self.colors[2], density=True)
        axes[0, 1].set_xlabel('Prediction Score')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Score Distribution by Click Status')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot by position
        position_data = [df[df['position'] == i]['prediction_score'].values for i in range(min(10, df['position'].max() + 1))]
        bp = axes[1, 0].boxplot(position_data, patch_artist=True)
        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
        axes[1, 0].set_xlabel('Position in Impression')
        axes[1, 0].set_ylabel('Prediction Score')
        axes[1, 0].set_title('Score Distribution by Position')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_scores = np.sort(df['prediction_score'])
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        axes[1, 1].plot(sorted_scores, cumulative, linewidth=2, color=self.colors[3])
        axes[1, 1].set_xlabel('Prediction Score')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution of Scores')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_topk_accuracy(self, df: pd.DataFrame, max_k: int = 10, save_path: Optional[str] = None) -> None:
        """
        Plot Top-K accuracy metrics.
        
        Args:
            df: DataFrame with prediction data
            max_k: Maximum K value to compute
            save_path: Optional path to save the plot
        """
        # Group by session and compute top-k accuracy
        sessions = df.groupby('session_id')
        k_values = list(range(1, max_k + 1))
        accuracies = []
        
        for k in k_values:
            session_accuracies = []
            for session_id, session_data in sessions:
                # Sort by prediction score descending
                sorted_session = session_data.sort_values('prediction_score', ascending=False)
                # Get top-k predictions
                top_k = sorted_session.head(k)
                # Check if any top-k item was clicked
                accuracy = 1 if top_k['ground_truth'].sum() > 0 else 0
                session_accuracies.append(accuracy)
            accuracies.append(np.mean(session_accuracies))
        
        # Compute NDCG@K as well
        ndcg_scores = []
        for k in k_values:
            session_ndcgs = []
            for session_id, session_data in sessions:
                ndcg = self._compute_ndcg(session_data, k)
                session_ndcgs.append(ndcg)
            ndcg_scores.append(np.mean(session_ndcgs))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Top-K Performance Metrics', fontsize=16, fontweight='bold')
        
        # Top-K Accuracy
        ax1.plot(k_values, accuracies, marker='o', linewidth=3, markersize=8, color=self.colors[0])
        ax1.fill_between(k_values, accuracies, alpha=0.3, color=self.colors[0])
        ax1.set_xlabel('K (Top-K)')
        ax1.set_ylabel('Accuracy@K')
        ax1.set_title('Top-K Accuracy')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add value annotations
        for i, (k, acc) in enumerate(zip(k_values, accuracies)):
            ax1.annotate(f'{acc:.3f}', (k, acc), textcoords="offset points", xytext=(0,10), ha='center')
        
        # NDCG@K
        ax2.plot(k_values, ndcg_scores, marker='s', linewidth=3, markersize=8, color=self.colors[1])
        ax2.fill_between(k_values, ndcg_scores, alpha=0.3, color=self.colors[1])
        ax2.set_xlabel('K (Top-K)')
        ax2.set_ylabel('NDCG@K')
        ax2.set_title('Normalized Discounted Cumulative Gain')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add value annotations
        for i, (k, ndcg) in enumerate(zip(k_values, ndcg_scores)):
            ax2.annotate(f'{ndcg:.3f}', (k, ndcg), textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'topk_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _compute_ndcg(self, session_data: pd.DataFrame, k: int) -> float:
        """Compute NDCG@K for a single session."""
        sorted_session = session_data.sort_values('prediction_score', ascending=False).head(k)
        relevance = sorted_session['ground_truth'].values
        
        # DCG
        dcg = relevance[0]
        for i in range(1, len(relevance)):
            dcg += relevance[i] / np.log2(i + 2)
        
        # IDCG (perfect ranking)
        ideal_relevance = sorted(session_data['ground_truth'].values, reverse=True)[:k]
        idcg = ideal_relevance[0] if len(ideal_relevance) > 0 else 0
        for i in range(1, len(ideal_relevance)):
            idcg += ideal_relevance[i] / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0
    
    def plot_interaction_heatmap(self, df: pd.DataFrame, news_df: pd.DataFrame = None, 
                                sample_users: int = 50, sample_items: int = 50, 
                                save_path: Optional[str] = None) -> None:
        """
        Plot user-item interaction heatmap.
        
        Args:
            df: DataFrame with prediction data
            news_df: Optional news metadata
            sample_users: Number of users to sample for visualization
            sample_items: Number of items to sample for visualization
            save_path: Optional path to save the plot
        """
        # Create user-item interaction matrix
        pivot_data = df.pivot_table(
            index='session_id', 
            columns='impression_id', 
            values='prediction_score',
            fill_value=0
        )
        
        # Sample for visualization
        if len(pivot_data) > sample_users:
            pivot_data = pivot_data.sample(n=sample_users, random_state=42)
        if len(pivot_data.columns) > sample_items:
            pivot_data = pivot_data.loc[:, pivot_data.columns[:sample_items]]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('User-Item Interaction Analysis', fontsize=16, fontweight='bold')
        
        # Prediction scores heatmap
        sns.heatmap(pivot_data, cmap='viridis', ax=axes[0, 0], cbar_kws={'label': 'Prediction Score'})
        axes[0, 0].set_title('Prediction Scores Heatmap')
        axes[0, 0].set_xlabel('Item ID')
        axes[0, 0].set_ylabel('User Session ID')
        
        # Click pattern heatmap
        click_data = df.pivot_table(
            index='session_id',
            columns='impression_id',
            values='ground_truth',
            fill_value=0
        )
        
        if len(click_data) > sample_users:
            click_data = click_data.sample(n=sample_users, random_state=42)
        if len(click_data.columns) > sample_items:
            click_data = click_data.loc[:, click_data.columns[:sample_items]]
        
        sns.heatmap(click_data, cmap='Reds', ax=axes[0, 1], cbar_kws={'label': 'Clicked (1) / Not Clicked (0)'})
        axes[0, 1].set_title('Actual Click Patterns')
        axes[0, 1].set_xlabel('Item ID')
        axes[0, 1].set_ylabel('User Session ID')
        
        # User activity distribution
        user_activity = df.groupby('session_id')['ground_truth'].sum()
        axes[1, 0].hist(user_activity, bins=20, alpha=0.7, color=self.colors[2], edgecolor='black')
        axes[1, 0].set_xlabel('Number of Clicks per Session')
        axes[1, 0].set_ylabel('Number of Users')
        axes[1, 0].set_title('User Activity Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Item popularity distribution
        item_popularity = df.groupby('impression_id')['ground_truth'].sum()
        axes[1, 1].hist(item_popularity, bins=20, alpha=0.7, color=self.colors[3], edgecolor='black')
        axes[1, 1].set_xlabel('Number of Clicks per Item')
        axes[1, 1].set_ylabel('Number of Items')
        axes[1, 1].set_title('Item Popularity Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'interaction_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_category_analysis(self, df: pd.DataFrame, news_df: pd.DataFrame, 
                              save_path: Optional[str] = None) -> None:
        """
        Analyze recommendation patterns by news categories.
        
        Args:
            df: DataFrame with prediction data
            news_df: DataFrame with news metadata
            save_path: Optional path to save the plot
        """
        if news_df.empty:
            print("Warning: No news metadata available for category analysis")
            return
        
        # Ensure both columns are string type for merging
        print(f"  [plot_category_analysis] Preparing data for merge...")
        df['impression_id_str'] = df['impression_id'].astype(str)
        news_df['news_id_str'] = news_df['news_id'].astype(str)
        
        # Merge prediction data with news metadata
        print(f"  [plot_category_analysis] Merging {len(df)} predictions with {len(news_df)} news items...")
        df_with_categories = df.merge(
            news_df[['news_id', 'category', 'subcategory']], 
            left_on='impression_id_str', 
            right_on='news_id', 
            how='left'
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Category-wise Recommendation Analysis', fontsize=16, fontweight='bold')
        
        # Category distribution in recommendations
        category_counts = df_with_categories['category'].value_counts()
        axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', 
                       colors=self.colors[:len(category_counts)])
        axes[0, 0].set_title('Distribution of Recommended News by Category')
        
        # Click-through rate by category
        ctr_by_category = df_with_categories.groupby('category').agg({
            'ground_truth': ['sum', 'count']
        }).round(3)
        ctr_by_category.columns = ['clicks', 'impressions']
        ctr_by_category['ctr'] = ctr_by_category['clicks'] / ctr_by_category['impressions']
        
        bars = axes[0, 1].bar(ctr_by_category.index, ctr_by_category['ctr'], 
                              color=self.colors[:len(ctr_by_category)])
        axes[0, 1].set_xlabel('Category')
        axes[0, 1].set_ylabel('Click-Through Rate')
        axes[0, 1].set_title('Click-Through Rate by Category')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, ctr in zip(bars, ctr_by_category['ctr']):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                            f'{ctr:.3f}', ha='center', va='bottom')
        
        # Average prediction score by category
        avg_scores = df_with_categories.groupby('category')['prediction_score'].mean()
        bars = axes[1, 0].bar(avg_scores.index, avg_scores.values, 
                              color=self.colors[:len(avg_scores)])
        axes[1, 0].set_xlabel('Category')
        axes[1, 0].set_ylabel('Average Prediction Score')
        axes[1, 0].set_title('Average Prediction Score by Category')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Score distribution by category (box plot)
        unique_categories = [cat for cat in df_with_categories['category'].unique() if pd.notna(cat) and cat != '']
        category_data = []
        category_labels = []
        
        for cat in unique_categories:
            cat_scores = df_with_categories[df_with_categories['category'] == cat]['prediction_score'].values
            if len(cat_scores) > 0:  # Only include categories with data
                category_data.append(cat_scores)
                category_labels.append(cat)
        
        if len(category_data) > 0:
            bp = axes[1, 1].boxplot(category_data, labels=category_labels, patch_artist=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(self.colors[i % len(self.colors)])
        else:
            axes[1, 1].text(0.5, 0.5, 'No category data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_xlabel('Category')
        axes[1, 1].set_ylabel('Prediction Score')
        axes[1, 1].set_title('Score Distribution by Category')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'category_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, model_results: Dict[str, Dict], save_path: Optional[str] = None) -> None:
        """
        Compare multiple models' performance.
        
        Args:
            model_results: Dictionary with model names as keys and metrics as values
            save_path: Optional path to save the plot
        """
        if not model_results:
            print("Warning: No model results provided for comparison")
            return
        
        # Extract metrics
        models = list(model_results.keys())
        metrics = ['auc', 'mrr', 'ndcg@5', 'ndcg@10']
        
        metric_data = {}
        for metric in metrics:
            metric_data[metric] = []
            for model in models:
                if 'final_test_metrics' in model_results[model]:
                    value = model_results[model]['final_test_metrics'].get(metric, 0)
                elif 'best_validation_summary' in model_results[model]:
                    value = model_results[model]['best_validation_summary'].get(f'val_{metric}', 0)
                else:
                    value = 0
                metric_data[metric].append(value)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            bars = ax.bar(models, metric_data[metric], color=self.colors[:len(models)])
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_ylabel(metric.upper())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, metric_data[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_diversity_metrics(self, df: pd.DataFrame, news_df: pd.DataFrame, 
                              save_path: Optional[str] = None) -> None:
        """
        Analyze recommendation diversity metrics.
        
        Args:
            df: DataFrame with prediction data
            news_df: DataFrame with news metadata
            save_path: Optional path to save the plot
        """
        if news_df.empty:
            print("Warning: No news metadata available for diversity analysis")
            return
        
        # Merge with news metadata
        df_with_categories = df.merge(
            news_df[['news_id', 'category']], 
            left_on='impression_id', 
            right_on='news_id', 
            how='left'
        )
        
        # Compute diversity metrics per session
        session_diversity = []
        sessions = df_with_categories.groupby('session_id')
        
        for session_id, session_data in sessions:
            # Get top-5 recommendations
            top_recs = session_data.nlargest(5, 'prediction_score')
            
            # Category diversity (number of unique categories in top-5)
            unique_categories = top_recs['category'].nunique()
            total_items = len(top_recs)
            category_diversity = unique_categories / total_items if total_items > 0 else 0
            
            # Intra-list diversity (average pairwise distance)
            scores = top_recs['prediction_score'].values
            if len(scores) > 1:
                pairwise_distances = []
                for i in range(len(scores)):
                    for j in range(i + 1, len(scores)):
                        pairwise_distances.append(abs(scores[i] - scores[j]))
                intra_list_diversity = np.mean(pairwise_distances)
            else:
                intra_list_diversity = 0
            
            session_diversity.append({
                'session_id': session_id,
                'category_diversity': category_diversity,
                'intra_list_diversity': intra_list_diversity,
                'num_recommendations': total_items
            })
        
        diversity_df = pd.DataFrame(session_diversity)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Recommendation Diversity Analysis', fontsize=16, fontweight='bold')
        
        # Category diversity distribution
        axes[0, 0].hist(diversity_df['category_diversity'], bins=20, alpha=0.7, 
                        color=self.colors[0], edgecolor='black')
        axes[0, 0].set_xlabel('Category Diversity Score')
        axes[0, 0].set_ylabel('Number of Sessions')
        axes[0, 0].set_title('Distribution of Category Diversity')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Intra-list diversity distribution
        axes[0, 1].hist(diversity_df['intra_list_diversity'], bins=20, alpha=0.7,
                        color=self.colors[1], edgecolor='black')
        axes[0, 1].set_xlabel('Intra-list Diversity Score')
        axes[0, 1].set_ylabel('Number of Sessions')
        axes[0, 1].set_title('Distribution of Intra-list Diversity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Diversity vs Accuracy trade-off
        session_accuracy = df.groupby('session_id')['ground_truth'].max()  # Did user click anything?
        diversity_accuracy = diversity_df.merge(
            session_accuracy.reset_index(), on='session_id'
        )
        
        # Scatter plot
        clicked = diversity_accuracy[diversity_accuracy['ground_truth'] == 1]
        not_clicked = diversity_accuracy[diversity_accuracy['ground_truth'] == 0]
        
        axes[1, 0].scatter(not_clicked['category_diversity'], not_clicked['intra_list_diversity'], 
                          alpha=0.6, label='No Click', color=self.colors[2])
        axes[1, 0].scatter(clicked['category_diversity'], clicked['intra_list_diversity'], 
                          alpha=0.6, label='Clicked', color=self.colors[3])
        axes[1, 0].set_xlabel('Category Diversity')
        axes[1, 0].set_ylabel('Intra-list Diversity')
        axes[1, 0].set_title('Diversity vs Click Success')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Average diversity by click status
        diversity_by_click = diversity_accuracy.groupby('ground_truth').agg({
            'category_diversity': 'mean',
            'intra_list_diversity': 'mean'
        })
        
        x = ['No Click', 'Clicked']
        width = 0.35
        x_pos = np.arange(len(x))
        
        axes[1, 1].bar(x_pos - width/2, diversity_by_click['category_diversity'], 
                       width, label='Category Diversity', color=self.colors[4])
        axes[1, 1].bar(x_pos + width/2, diversity_by_click['intra_list_diversity'], 
                       width, label='Intra-list Diversity', color=self.colors[5])
        axes[1, 1].set_xlabel('Click Status')
        axes[1, 1].set_ylabel('Average Diversity Score')
        axes[1, 1].set_title('Average Diversity by Click Status')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(x)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'diversity_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cold_start_analysis(self, df: pd.DataFrame, news_df: pd.DataFrame = None, 
                                save_path: Optional[str] = None) -> None:
        """
        Analyze cold start performance (new users/items).
        
        Args:
            df: DataFrame with prediction data
            news_df: Optional news metadata
            save_path: Optional path to save the plot
        """
        # Simulate cold start by analyzing performance for users/items with few interactions
        user_interaction_counts = df.groupby('session_id').size()
        item_interaction_counts = df.groupby('impression_id').size()
        
        # Define cold start thresholds
        cold_user_threshold = user_interaction_counts.quantile(0.25)
        cold_item_threshold = item_interaction_counts.quantile(0.25)
        
        # Categorize users and items
        df['user_type'] = df['session_id'].map(
            lambda x: 'Cold' if user_interaction_counts[x] <= cold_user_threshold else 'Warm'
        )
        df['item_type'] = df['impression_id'].map(
            lambda x: 'Cold' if item_interaction_counts.get(x, 0) <= cold_item_threshold else 'Warm'
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cold Start Analysis', fontsize=16, fontweight='bold')
        
        # Performance by user type
        user_performance = df.groupby('user_type').agg({
            'prediction_score': 'mean',
            'ground_truth': ['sum', 'count']
        })
        user_performance.columns = ['avg_score', 'clicks', 'impressions']
        user_performance['ctr'] = user_performance['clicks'] / user_performance['impressions']
        
        x = list(user_performance.index)
        axes[0, 0].bar(x, user_performance['avg_score'], color=self.colors[:len(x)])
        axes[0, 0].set_title('Average Prediction Score by User Type')
        axes[0, 0].set_ylabel('Average Prediction Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # CTR by user type
        axes[0, 1].bar(x, user_performance['ctr'], color=self.colors[:len(x)])
        axes[0, 1].set_title('Click-Through Rate by User Type')
        axes[0, 1].set_ylabel('CTR')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance by item type
        item_performance = df.groupby('item_type').agg({
            'prediction_score': 'mean',
            'ground_truth': ['sum', 'count']
        })
        item_performance.columns = ['avg_score', 'clicks', 'impressions']
        item_performance['ctr'] = item_performance['clicks'] / item_performance['impressions']
        
        x = list(item_performance.index)
        axes[1, 0].bar(x, item_performance['avg_score'], color=self.colors[:len(x)])
        axes[1, 0].set_title('Average Prediction Score by Item Type')
        axes[1, 0].set_ylabel('Average Prediction Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # CTR by item type
        axes[1, 1].bar(x, item_performance['ctr'], color=self.colors[:len(x)])
        axes[1, 1].set_title('Click-Through Rate by Item Type')
        axes[1, 1].set_ylabel('CTR')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'cold_start_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_and_calibration(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve and calibration analysis.
        
        Args:
            df: DataFrame with prediction data
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Calibration and ROC Analysis', fontsize=16, fontweight='bold')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(df['ground_truth'], df['prediction_score'])
        roc_auc = auc(fpr, tpr)
        
        axes[0].plot(fpr, tpr, color=self.colors[0], lw=3, 
                     label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('Receiver Operating Characteristic')
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)
        
        # Calibration Plot
        # Bin predictions and compute calibration
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        predicted_probs = (df['prediction_score'] - df['prediction_score'].min()) / \
                         (df['prediction_score'].max() - df['prediction_score'].min())
        
        bin_means = []
        bin_true_probs = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
            if in_bin.sum() > 0:
                bin_means.append(predicted_probs[in_bin].mean())
                bin_true_probs.append(df[in_bin]['ground_truth'].mean())
        
        axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        axes[1].plot(bin_means, bin_true_probs, 'o-', color=self.colors[1], 
                     label='Model calibration', linewidth=2, markersize=8)
        axes[1].set_xlabel('Mean Predicted Probability')
        axes[1].set_ylabel('Fraction of Positives')
        axes[1].set_title('Calibration Plot')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'roc_calibration.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, prediction_files: List[str], 
                                    news_files: List[str] = None,
                                    summary_files: List[str] = None,
                                    model_names: List[str] = None) -> None:
        """
        Generate a comprehensive visualization report.
        
        Args:
            prediction_files: List of prediction file paths
            news_files: Optional list of news metadata files
            summary_files: Optional list of training summary files
            model_names: Optional list of model names
        """
        print("Generating comprehensive recommendation system analysis report...")
        
        # Load all data
        all_predictions = []
        model_summaries = {}
        
        for i, pred_file in enumerate(prediction_files):
            print(f"Loading predictions from {pred_file}...")
            df = self.load_prediction_data(pred_file)
            
            model_name = model_names[i] if model_names and i < len(model_names) else f"Model_{i+1}"
            df['model'] = model_name
            all_predictions.append(df)
            
            # Load corresponding summary if available
            if summary_files and i < len(summary_files):
                model_summaries[model_name] = self.load_training_summary(summary_files[i])
        
        # Combine all predictions
        combined_df = pd.concat(all_predictions, ignore_index=True)
        
        # Load news metadata if available
        news_df = pd.DataFrame()
        if news_files:
            news_dfs = []
            for news_file in news_files:
                df_news = self.load_news_metadata(news_file)
                if not df_news.empty:
                    news_dfs.append(df_news)
            if news_dfs:
                news_df = pd.concat(news_dfs, ignore_index=True).drop_duplicates('news_id')
        
        # Generate all visualizations
        print("\n=== Starting Visualization Generation ===")
        
        # 1. Score distribution analysis
        print("\n[1/9] Score Distribution Analysis")
        self.plot_score_distribution(combined_df, self.output_dir / 'comprehensive_score_distribution.png')
        print("  ✓ Score distribution plots saved")
        
        # 2. Top-K accuracy
        print("\n[2/9] Top-K Accuracy Analysis")
        self.plot_topk_accuracy(combined_df, save_path=self.output_dir / 'comprehensive_topk_accuracy.png')
        print("  ✓ Top-K accuracy plots saved")
        
        # 3. Interaction heatmap
        print("\n[3/9] Interaction Heatmap")
        self.plot_interaction_heatmap(combined_df, news_df, 
                                     save_path=self.output_dir / 'comprehensive_interaction_heatmap.png')
        print("  ✓ Interaction heatmap saved")
        
        # 4. Category analysis (if news metadata available)
        if not news_df.empty:
            print("\n[4/9] Category Distribution Analysis")
            self.plot_category_analysis(combined_df, news_df, 
                                       save_path=self.output_dir / 'comprehensive_category_analysis.png')
            print("  ✓ Category analysis plots saved")
            
            # 5. Diversity metrics
            print("\n[5/9] Diversity Metrics")
            self.plot_diversity_metrics(combined_df, news_df,
                                       save_path=self.output_dir / 'comprehensive_diversity_metrics.png')
            print("  ✓ Diversity metrics plots saved")
        else:
            print("\n[4/9] Category Distribution Analysis - SKIPPED (no news metadata)")
            print("\n[5/9] Diversity Metrics - SKIPPED (no news metadata)")
        
        # 6. Cold start analysis
        print("\n[6/9] Cold Start Analysis")
        self.plot_cold_start_analysis(combined_df, 
                                     save_path=self.output_dir / 'comprehensive_cold_start_analysis.png')
        print("  ✓ Cold start analysis plots saved")
        
        # 7. ROC and calibration
        print("\n[7/9] ROC and Calibration Analysis")
        self.plot_roc_and_calibration(combined_df,
                                     save_path=self.output_dir / 'comprehensive_roc_calibration.png')
        print("  ✓ ROC and calibration plots saved")
        
        # 8. Model comparison (if multiple models)
        if len(all_predictions) > 1:
            print("\n[8/9] Model Comparison")
            self.plot_model_comparison(combined_df,
                                     save_path=self.output_dir / 'comprehensive_model_comparison.png')
            print("  ✓ Model comparison plots saved")
        else:
            print("\n[8/9] Model Comparison - SKIPPED (single model)")
            
        # 9. Summary statistics
        print("\n[9/9] Summary Statistics")
        self._generate_summary_statistics(combined_df, news_df, model_summaries)
        print("  ✓ Summary statistics generated")
        
        print(f"\n🎉 Comprehensive report generated! Check {self.output_dir} for all visualizations.")
    
    def _generate_summary_statistics(self, df: pd.DataFrame, news_df: pd.DataFrame, 
                                   model_summaries: Dict) -> None:
        """Generate and save summary statistics."""
        stats = {
            'dataset_stats': {
                'total_impressions': len(df),
                'total_sessions': df['session_id'].nunique(),
                'total_items': df['impression_id'].nunique(),
                'overall_ctr': df['ground_truth'].mean(),
                'avg_session_length': df.groupby('session_id').size().mean(),
            },
            'score_stats': {
                'mean_score': df['prediction_score'].mean(),
                'std_score': df['prediction_score'].std(),
                'min_score': df['prediction_score'].min(),
                'max_score': df['prediction_score'].max(),
            }
        }
        
        if not news_df.empty and 'category' in news_df.columns:
            df_with_cat = df.merge(news_df[['news_id', 'category']], 
                                   left_on='impression_id', right_on='news_id', how='left')
            category_stats = df_with_cat.groupby('category').agg({
                'ground_truth': ['sum', 'count', 'mean'],
                'prediction_score': 'mean'
            }).round(4)
            
            # Flatten column names and convert to simple dict
            category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns.values]
            stats['category_stats'] = category_stats.to_dict()
        
        # Save statistics
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print("Summary statistics saved to summary_statistics.json")


def main():
    """
    Example usage of the RecommendationVisualizer.
    
    Modify the file paths below to match your data structure.
    """
    
    # Initialize visualizer
    visualizer = RecommendationVisualizer(output_dir="recommendation_analysis_output")
    
    # Example file paths - modify these according to your data
    base_path = "/home/itachi/Programming/Utokyo/LabSuzumura/NewsReX/outputs"
    
    prediction_files = [
        f"{base_path}/nrms_mind_small/2025-08-13-07-49-07/predictions/test/test_predictions_epoch_1.txt"
    ]
    
    news_files = [
        "/home/itachi/Programming/Utokyo/LabSuzumura/NewsReX/data/mind/small/train/news.tsv"
    ]
    
    summary_files = [
        f"{base_path}/nrms_mind_small/2025-08-13-07-49-07/training_run_summary.json"
    ]
    
    model_names = ["NRMS"]
    
    # Generate comprehensive report
    visualizer.generate_comprehensive_report(
        prediction_files=prediction_files,
        news_files=news_files,
        summary_files=summary_files,
        model_names=model_names
    )


if __name__ == "__main__":
    main()