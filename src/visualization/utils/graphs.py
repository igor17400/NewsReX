import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def boundary_line(ax, x_lin, boundary_ctr, ctr_legend_handles, color):
    label_text = f'CTR > {boundary_ctr * 100:.1f}%'
    line, = ax.plot(
        x_lin, boundary_ctr * x_lin, '--',
        linewidth=1.0, alpha=0.7,
        color=color,
        label=label_text
    )
    ctr_legend_handles.append(line)


def create_exposure_click_heatmap(
        stats_df,
        figsize=(12, 12),
        title='Article Performance: Impressions vs Clicks',
        show_ctr_regions=True,
        ctr_levels=[0.1, 0.05, 0.01, 0.001],
        cmap='plasma',
        ctr_legend_loc='upper left',
        ax=None
):
    """
    Create a scatter plot heatmap with a custom-placed legend for CTR lines.
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

    x_limit = stats_df["impressions"].max() * 1.05
    y_limit = stats_df["clicks"].max() * 1.05

    min_val = min(stats_df["ctr"].min(), min(ctr_levels))
    max_val = max(stats_df["ctr"].max(), max(ctr_levels))

    epsilon = 1e-9
    norm = LogNorm(vmin=max(min_val, epsilon), vmax=max_val)
    cmap_obj = plt.get_cmap(cmap)

    # Prepare lists to store handles and labels for the custom legend
    ctr_legend_handles = []

    if show_ctr_regions and len(ctr_levels) > 0:
        x_lin = np.linspace(0.01, x_limit, 200)
        sorted_levels = sorted(ctr_levels, reverse=True)

        upper_bound = max_val
        mid_point_ctr = (sorted_levels[0] + upper_bound) / 2
        region_color = cmap_obj(norm(mid_point_ctr))
        ax.fill_between(x_lin, sorted_levels[0] * x_lin, y_limit, color=region_color, alpha=0.25)

        for i in range(len(sorted_levels) - 1):
            upper_ctr, lower_ctr = sorted_levels[i], sorted_levels[i + 1]
            mid_point_ctr = (upper_ctr + lower_ctr) / 2
            region_color = cmap_obj(norm(mid_point_ctr))
            ax.fill_between(x_lin, lower_ctr * x_lin, upper_ctr * x_lin, color=region_color, alpha=0.25)
            boundary_line(ax, x_lin, lower_ctr, ctr_legend_handles, region_color)

        lower_bound = min_val
        mid_point_ctr = (sorted_levels[-1] + lower_bound) / 2
        region_color = cmap_obj(norm(mid_point_ctr))
        ax.fill_between(x_lin, 0, sorted_levels[-1] * x_lin, color=region_color, alpha=0.25)
        boundary_line(ax, x_lin, lower_bound, ctr_legend_handles, region_color)

    min_clicks, max_clicks = stats_df['clicks'].min(), stats_df['clicks'].max()
    bubble_sizes = 50 if min_clicks == max_clicks else 20 + (
            (stats_df['clicks'] - min_clicks) / (max_clicks - min_clicks)) * 480
    scatter = ax.scatter(
        stats_df["impressions"], stats_df["clicks"], s=bubble_sizes, c=stats_df["ctr"],
        cmap=cmap_obj, alpha=0.8, norm=norm, edgecolors='white', linewidth=0.5,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Click-Through Rate (CTR)', rotation=270, labelpad=20)

    # -- CUSTOMIZATION --
    ax.set_xlim(0, x_limit)
    ax.set_ylim(0, y_limit)
    ax.set_xlabel('Total Impressions', fontsize=12)
    ax.set_ylabel('Total Clicks', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

    # --- STATISTICS BOX ---
    stats_text = f"""
    Total Articles: {len(stats_df):,}
    Avg CTR: {stats_df['ctr'].mean():.4f}
    Median CTR: {stats_df['ctr'].median():.4f}
    """
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='gray'))

    if ctr_legend_handles:
        if ctr_legend_loc == 'upper left':
            bbox_to_anchor = (0.02, 0.85)
        else:
            bbox_to_anchor = (1, 1)

        ax.legend(
            handles=ctr_legend_handles,
            loc=ctr_legend_loc,
            bbox_to_anchor=bbox_to_anchor,
            fontsize=9,
            title='CTR Levels',
            title_fontsize='9'
        )

    plt.tight_layout()

    if return_fig:
        return fig, ax
    else:
        return None, ax
