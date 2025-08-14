import pandas as pd
import numpy as np


def article_stats(df, min_impressions=1, min_clicks=0):
    """
    Prepare article statistics from dataframe using the efficient flattening approach

    Parameters:
        df: DataFrame with 'ImpressionIDs' and 'GroundTruths' columns
        min_impressions: minimum impressions to include article
        min_clicks: minimum clicks to include article (0 includes articles with no clicks)

    Returns:
        DataFrame with columns ['article_id', 'impressions', 'clicks', 'ctr']
    """
    print("🔄 Preparing article statistics...")

    # Flatten the Dataframe efficiently
    exploded = pd.DataFrame({
        "article_id": [id_ for ids in df["ImpressionIDs"] for id_ in ids],
        "clicked": [gt for gts in df["GroundTruths"] for gt in gts]
    })

    # Compute aggregated stats
    stats = exploded.groupby("article_id").agg(
        impressions=("clicked", "count"),
        clicks=("clicked", "sum"),
    ).assign(ctr=lambda x: x["clicks"] / x["impressions"])

    # Apply filters
    stats = stats[stats["impressions"] >= min_impressions]
    if min_clicks > 0:
        stats = stats[stats["clicks"] >= min_clicks]

    print(f"✅ Stats prepared! Found {len(stats)} articles")
    print(f"   📊 Impressions range: {stats['impressions'].min()}-{stats['impressions'].max()}")
    print(f"   📊 Clicks range: {stats['clicks'].min()}-{stats['clicks'].max()}")
    print(f"   📊 CTR range: {stats['ctr'].min():.4f}-{stats['ctr'].max():.4f}")

    return stats


def filter_stats_by_percentile(stats_df, percentile_to_show=95, filter_by='both', filter_type='bottom'):
    """
    Filter statistics by percentile to focus on specific data ranges

    Parameters:
    stats_df: DataFrame with article statistics
    percentile_to_show: percentile threshold (e.g., 95 for bottom 95% or top 5%)
    filter_by: 'impressions', 'clicks', 'both', or 'none'
    filter_type: 'bottom', 'top', or 'both' (returns tuple for both)

    Returns:
    Filtered DataFrame or tuple of (bottom_df, top_df) if filter_type='both'
    """
    if filter_by == 'none':
        if filter_type == 'both':
            return stats_df, stats_df
        return stats_df

    original_count = len(stats_df)

    if filter_type == 'both':
        # Return both bottom and top percentiles
        bottom_df = stats_df.copy()
        top_df = stats_df.copy()

        if filter_by in ['impressions', 'both']:
            impression_threshold = stats_df['impressions'].quantile(percentile_to_show / 100)
            bottom_df = bottom_df[bottom_df['impressions'] <= impression_threshold]
            top_df = top_df[top_df['impressions'] > impression_threshold]

        if filter_by in ['clicks', 'both']:
            clicks_threshold = stats_df['clicks'].quantile(percentile_to_show / 100)
            bottom_df = bottom_df[bottom_df['clicks'] <= clicks_threshold]
            top_df = top_df[top_df['clicks'] > clicks_threshold]

        print(f"Bottom {percentile_to_show}%: {len(bottom_df):,} articles")
        print(f"Top {100 - percentile_to_show}%: {len(top_df):,} articles")

        return bottom_df, top_df

    elif filter_type == 'bottom':
        # Original behavior - bottom percentile
        filtered_df = stats_df.copy()

        if filter_by in ['impressions', 'both']:
            impression_limit = stats_df['impressions'].quantile(percentile_to_show / 100)
            filtered_df = filtered_df[filtered_df['impressions'] <= impression_limit]

        if filter_by in ['clicks', 'both']:
            click_limit = stats_df['clicks'].quantile(percentile_to_show / 100)
            filtered_df = filtered_df[filtered_df['clicks'] <= click_limit]

        print(f"Filtered from {original_count:,} to {len(filtered_df):,} articles (bottom {percentile_to_show}%)")
        return filtered_df

    elif filter_type == 'top':
        # Top percentile (e.g., top 5%)
        filtered_df = stats_df.copy()
        top_percentile = 100 - percentile_to_show  # Convert to top percentile

        if filter_by in ['impressions', 'both']:
            impression_limit = stats_df['impressions'].quantile(top_percentile / 100)
            filtered_df = filtered_df[filtered_df['impressions'] >= impression_limit]

        if filter_by in ['clicks', 'both']:
            click_limit = stats_df['clicks'].quantile(top_percentile / 100)
            filtered_df = filtered_df[filtered_df['clicks'] >= click_limit]

        print(f"Filtered from {original_count:,} to {len(filtered_df):,} articles (top {100 - percentile_to_show}%)")
        return filtered_df
