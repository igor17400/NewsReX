from src.visualization.utils.preprocess import filter_stats_by_percentile
from utils.input import parse_recommendation_stream
from utils.preprocess import article_stats
from utils.graphs import create_exposure_click_heatmap
import matplotlib.pyplot as plt
import os
from datetime import datetime


def analyze_article_performance(
        df,
        min_impressions=1,
        min_clicks=0,
        percentile_to_show=95,
        figsize=(12, 8),
        save_plots=True,
        output_dir="./src/visualization/figures/"
):
    """
    Complete workflow: prepare data and create enhanced visualizations

    Parameters:
    df: DataFrame with 'ImpressionIDs' and 'GroundTruths' columns
    min_impressions: minimum impressions filter
    min_clicks: minimum clicks filter
    percentile_to_show: percentile for optional filtering
    figsize: figure size
    save_plots: whether to save plots automatically
    output_dir: directory to save plots

    Returns:
    stats_df: prepared statistics DataFrame
    saved_files: list of saved file paths
    """
    print("🚀 Starting Article Performance Analysis")
    print("=" * 50)

    # Create output directory if it doesn't exist
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    # Prepare data
    stats_df = article_stats(df, min_impressions, min_clicks)

    # 1. FULL DATASET HEATMAP
    print("\nCreating full dataset heatmap visualization...")
    fig1, ax1 = create_exposure_click_heatmap(
        stats_df,
        figsize=figsize,
        title='Article Performance: Full Dataset - Impressions vs Clicks with CTR Regions'
    )

    if save_plots and fig1 is not None:
        filename1 = f"{output_dir}heatmap_full_dataset.png"
        fig1.savefig(filename1, dpi=300, bbox_inches='tight', facecolor='white')
        saved_files.append(filename1)
        print(f"   ✅ Saved: {filename1}")

    plt.show()

    # 2. FILTERED DATASET HEATMAP
    print(f"\nCreating filtered dataset heatmap ({percentile_to_show}% percentile)...")
    bottom_df, top_df = filter_stats_by_percentile(
        stats_df,
        percentile_to_show=percentile_to_show,
        filter_by='both',
        filter_type='both'
    )

    fig2, ax2 = create_exposure_click_heatmap(
        bottom_df,
        figsize=figsize,
        title=f'Article Performance: Bottom {percentile_to_show}% - Impressions vs Clicks with CTR Regions'
    )

    if save_plots and fig2 is not None:
        filename2 = f"{output_dir}heatmap_filtered_{percentile_to_show}p.png"
        fig2.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white')
        saved_files.append(filename2)
        print(f"   ✅ Saved: {filename2}")

    plt.show()

    # 3. CREATE COMPARISON PLOT
    print(f"\nCreating side-by-side comparison...")
    fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))

    # Panel 1: Full dataset
    create_exposure_click_heatmap(
        stats_df,
        title=f'Full Dataset\n({len(stats_df):,} articles)',
        ax=ax1,
        show_ctr_regions=True
    )

    # Panel 2: Bottom percentile
    create_exposure_click_heatmap(
        bottom_df,
        title=f'Bottom {percentile_to_show}%\n({len(bottom_df):,} articles)',
        ax=ax2,
        show_ctr_regions=True
    )

    # Panel 3: Top percentile
    create_exposure_click_heatmap(
        top_df,
        title=f'Top {100 - percentile_to_show}%\n({len(top_df):,} articles)',
        ax=ax3,
        show_ctr_regions=True
    )

    # Add overall title
    fig3.suptitle(
        f'Article Performance Analysis: Full vs Bottom {percentile_to_show}% vs Top {100 - percentile_to_show}%',
        fontsize=16, fontweight='bold', y=1.02)

    # Adjust layout
    plt.tight_layout()

    # Save plots
    if save_plots:
        filename3 = f"{output_dir}heatmap_comparison_{percentile_to_show}p.png"
        fig3.savefig(filename3, dpi=300, bbox_inches='tight', facecolor='white')
        saved_files.append(filename3)
        print(f"   ✅ Saved: {filename3}")

    plt.show()

    # Summary of saved files
    if save_plots and saved_files:
        print(f"\n💾 SAVED FILES:")
        for i, file_path in enumerate(saved_files, 1):
            print(f"   {i}. {os.path.basename(file_path)}")
        print(f"\n📁 All files saved to: {output_dir}")

    return stats_df, saved_files


if __name__ == "__main__":
    predictions_path = "./outputs/nrms_mind_small/2025-08-13-07-49-07/predictions/test/test_predictions_epoch_1.txt"
    predictions_df = parse_recommendation_stream(predictions_path)
    print(f"Predictions Dataframe Shape: {predictions_df.shape}")
    print("--- Predictions Dataframe ---")
    print(predictions_df)

    stats_df, saved_files = analyze_article_performance(
        predictions_df,
        save_plots=True,
        output_dir="./src/visualization/figures/"
    )

    print(f"\n🎉 Analysis complete! {len(saved_files)} plots saved.")
