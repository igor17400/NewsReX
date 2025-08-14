import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def create_exposure_click_heatmap(
        stats_df,
        figsize=(12, 8),
        title='Article Performance: Impressions vs Clicks',
        show_ctr_regions=True,
        ctr_levels=[1, 0.1, 0.05, 0.01, 0.001],
        region_colors=['lightblue', 'lightgreen', 'yellow', 'orange', 'lightcoral', 'white'],
        cmap='tab20c',
        ax=None
):
    """
    Create a standalone scatter plot heatmap showing exposure vs clicks with ctr color coding

    Parameters:
    stats_df: DataFrame with columns ['article_id', 'impressions', 'clicks', 'ctr']
    figsize: tuple, figure size
    title: string, plot title
    show_ctr_regions: bool, whether to show ctr region fills
    ctr_levels: list, ctr levels for region boundaries
    region_colors: list, colors for each ctr region
    cmap: string, colormap for scatter points

    Returns:
    fig, ax: matplotlib figure and axis objects
    """
    if stats_df.empty:
        print("⚠️ The provided DataFrame is empty. Nothing to plot.")
        return None, None

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False

    # Calculate plot limits with some padding
    x_limit = stats_df["impressions"].max() * 1.05
    y_limit = stats_df["clicks"].max() * 1.05

    # --- ctr REGION FILLS (optional) ---
    if show_ctr_regions and len(ctr_levels) > 0:
        x_lin = np.linspace(0, x_limit, 200)
        y_upper_bound = np.full_like(x_lin, y_limit)

        # Ensure we have enough colors
        while len(region_colors) < len(ctr_levels):
            region_colors.extend(region_colors)

        for i, (ctr_level, color) in enumerate(zip(ctr_levels, region_colors)):
            y_line = ctr_level * x_lin

            # Fill region
            ax.fill_between(
                x_lin, y_line, y_upper_bound,
                color=color, alpha=0.15,
                where=(y_line < y_upper_bound)
            )

            # Draw ctr line
            valid_mask = y_line <= y_limit
            if np.any(valid_mask):
                ax.plot(
                    x_lin[valid_mask],
                    y_line[valid_mask],
                    '--', label=f'ctr={ctr_level}',
                    linewidth=1.5, alpha=0.3, color=color
                )

            y_upper_bound = np.minimum(y_upper_bound, y_line)

    # --- SCATTER PLOT ---
    # Relative sizes for better visualization
    min_clicks = stats_df['clicks'].min()
    max_clicks = stats_df['clicks'].max()
    if min_clicks == max_clicks:
        bubble_sizes = 50
    else:
        bubble_sizes = 20 + ((stats_df['clicks'] - min_clicks) / (max_clicks - min_clicks)) * 480

    # Color normalization
    vmin = stats_df["ctr"].min()
    vmax = stats_df["ctr"].max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Create scatter plot
    scatter = ax.scatter(
        stats_df["impressions"],
        stats_df["clicks"],
        s=bubble_sizes,
        c=stats_df["ctr"],
        cmap=cmap,
        alpha=0.7,
        norm=norm,
        edgecolors='grey',
        linewidth=0.5,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Clicks-Through Impression (ctr)', rotation=270, labelpad=20)

    # --- TREND LINE ---
    if len(stats_df) > 1:
        z = np.polyfit(stats_df["impressions"], stats_df["clicks"], 1)
        p = np.poly1d(z)
        trend_x = np.linspace(
            stats_df["impressions"].min(),
            stats_df["impressions"].max(),
            100
        )
        ax.plot(
            trend_x,
            p(trend_x),
            "r--",
            alpha=0.8,
            linewidth=2,
            label="Trend line"
        )

    # -- CUSTOMIZATION --
    ax.set_xlim(0, x_limit)
    ax.set_ylim(0, y_limit)
    ax.set_xlabel('Total Impressions', fontsize=12)
    ax.set_ylabel('Total Clicks', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Legend
    if show_ctr_regions or len(stats_df) > 1:
        ax.legend(loc='upper right', fontsize=10)

    # --- STATISTICS BOX ---
    stats_text = f"""
    Dataset Statistics:
        Total Articles: {len(stats_df):,}
        Avg CTR: {stats_df['ctr'].mean():.4f}
        Median CTR: {stats_df['ctr'].median():.4f}
        Max CTR: {stats_df['ctr'].max():.4f}
        Total Clicks: {stats_df['clicks'].sum():,}
        Total Impressions: {stats_df['impressions'].sum():,}
    """

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.3, edgecolor='gray'))

    plt.tight_layout()

    if return_fig:
        return fig, ax
    else:
        return None, ax
