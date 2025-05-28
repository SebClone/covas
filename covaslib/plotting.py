"""
plotting.py

This module provides enhanced SHAP decision plotting capabilities with
custom overlays such as mean paths, standard deviation bounds, percentile fills,
and scatter markers for better visualization and interpretation of model explanations.

Functions:
----------
- custom_decision_plot: Generates a SHAP decision plot with optional overlays.

Dependencies:
-------------
- matplotlib
- numpy
- shap

Author: Sebastian Roth
"""
import matplotlib.pyplot as plt
import numpy as np
import shap

def custom_decision_plot(shap_dictonary, X_test, feature_names,
                         scatter_levels=None, line_levels=None, fill_levels=None, class_name=None):
    """
    Create a custom SHAP decision plot with optional overlays for mean path,
    standard deviation bounds, percentile fills, and scatter markers.

    Parameters
    ----------
    shap_base : float
        The expected value (base value) from the SHAP explainer.
    shap_vals : np.ndarray
        SHAP values for the samples to plot (shape: [samples, features]).
    X_test : np.ndarray or pd.DataFrame
        The feature values for the samples (used for SHAP plotting).
    feature_names : list of str
        List of feature names (order must match shap_vals columns).
    scatter_levels : list of str, optional
        Levels at which to plot scatter markers (e.g., ['mean', '1 std', '2 std', '3 std']).
    line_levels : list of str, optional
        Which lines to draw (e.g., ['mean', '1 std', '2 std']).
    fill_levels : list of str, optional
        Which percentile bands to fill (e.g., ['68%', '95%']).

    Returns
    -------
    None
        Displays a matplotlib plot.

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If input values are invalid or inconsistent.
    """
    # Input validation
    if class_name not in shap_dictonary:
        raise ValueError(f"class_name {class_name!r} not found in shap_dictionary keys")
    if not hasattr(plt, "figure"):
        raise TypeError("matplotlib.pyplot must be imported as plt")
    # Retrieve the SHAP base value (expected value) and SHAP values for the given class
    shap_base = shap_dictonary[class_name]['base value']
    shap_vals = shap_dictonary[class_name]['values']

    # If no overlays are specified, generate a standard SHAP decision plot
    if scatter_levels is None and line_levels is None and fill_levels is None:
        shap.decision_plot(shap_base, shap_vals, X_test, feature_names, show=True)
        return
    if scatter_levels is None:
        scatter_levels = []
    if line_levels is None:
        line_levels = []
    if fill_levels is None:
        fill_levels = []

    # Map string levels to numeric standard deviation values
    std_map = {'1 std': 1, '2 std': 2, '3 std': 3}
    if 'all' in line_levels:
        std_levels = [1, 2, 3]
    else:
        std_levels = [std_map[s] for s in line_levels if s in std_map]

    plot_scatter = {'mean': 'mean' in scatter_levels}
    plot_scatter.update({f'{i} std': (f'{i} std' in scatter_levels) for i in [1,2,3]})

    plot_fill = {'1': '68%' in fill_levels, '2': '95%' in fill_levels, '3': '99%' in fill_levels}

    height_in_inches = 10 #placeholder
    width_in_pixels = 2926
    DPI = 300
    # Set figure size for plotting, converting width from pixels to inches
    width_in_inches = width_in_pixels / DPI

    # Prepare figure with desired size
    plt.figure(figsize=(width_in_inches, height_in_inches))
    # Generate the decision plot but suppress immediate display
    shap.decision_plot(
        shap_base,
        shap_vals,
        X_test,
        feature_names,
        show=False,
        ignore_warnings=True
    )
    ax = plt.gca()
    # Initialize cumulative sums for means and std deviations (not used but kept for clarity)
    cumulative_mean = shap_base
    cumulative_neg_std = shap_base
    cumulative_pos_std = shap_base

    ax = plt.gca()
    # Determine feature order used in the SHAP decision plot
    order = [tick.get_text() for tick in ax.get_yticklabels() if tick.get_text()]
    shap_vals_ordered = shap_vals[:, [feature_names.index(f) for f in order]]

    # Compute cumulative SHAP paths, mean and standard deviation for overlays
    base = shap_base
    cum_paths = base + np.cumsum(shap_vals_ordered, axis=1)
    mean_path = np.mean(cum_paths, axis=0)
    std_path = np.std(cum_paths, axis=0)

    # Plot the mean/std lines and fill regions if enabled
    if 'mean' in line_levels:
        ax.plot(mean_path, range(len(order)), linestyle='-', linewidth=2, zorder=4, color='#333333', label='Mean Path Line')
    # Plot the ±1 Std lines and fill
    if 1 in std_levels:
        ax.plot(mean_path - std_path, range(len(order)), linestyle='--', linewidth=2, zorder=3, color='#82A582', label='±1 Std Line')
        ax.plot(mean_path + std_path, range(len(order)), linestyle='--', linewidth=2, zorder=3, color='#82A582', label='_nolegend_')
    if 1 in std_levels and plot_fill['1']:
        ax.fill_betweenx(
            range(len(order)),
            mean_path - std_path,
            mean_path + std_path,
            color='#82A582',
            alpha=0.4,
            zorder=4,
            label='68% Perzentil'
        )
    # Plot the ±2 Std lines and fill
    if 2 in std_levels:
        ax.plot(mean_path - 2*std_path, range(len(order)), linestyle='--', linewidth=2, zorder=2, color='#517551', label='±2 Std Line')
        ax.plot(mean_path + 2*std_path, range(len(order)), linestyle='--', linewidth=2, zorder=2, color='#517551', label='_nolegend_')
    if 2 in std_levels and plot_fill['2']:
        ax.fill_betweenx(
            range(len(order)),
            mean_path - 2*std_path,
            mean_path + 2*std_path,
            color='#517551',
            alpha=0.4,
            zorder=4,
            label='95% Perzentil'
        )
    # Plot the ±3 Std lines and fill
    if 3 in std_levels:
        ax.plot(mean_path - 3*std_path, range(len(order)), linestyle='--', linewidth=2, zorder=1, color='#2F4F2F', label='±3 Std Line')
        ax.plot(mean_path + 3*std_path, range(len(order)), linestyle='--', linewidth=2, zorder=1, color='#2F4F2F', label='_nolegend_')
    if 3 in std_levels and plot_fill['3']:
        ax.fill_betweenx(
            range(len(order)),
            mean_path - 3*std_path,
            mean_path + 3*std_path,
            color='#2F4F2F',
            alpha=0.4,
            zorder=4,
            label='99.7% Perzentil'
        )

    # Plot scatter markers for mean and std bounds at each feature position
    for idx, feature in enumerate(order):
        # Plot mean cumulative SHAP
        if plot_scatter['mean']:
            if idx == 0:
                ax.scatter(
                    mean_path[idx],
                    idx,
                    marker='D',
                    s=50,
                    zorder=5,
                    color='#333333',
                    label='Mean Path'
                )
            else:
                ax.scatter(
                    mean_path[idx],
                    idx,
                    marker='D',
                    s=50,
                    zorder=5,
                    color='#333333'
                )
        # Plot symmetric std bounds
        if 1 in std_levels and plot_scatter['1 std']:
            if idx == 0:
                ax.scatter(
                    [mean_path[idx] - std_path[idx], mean_path[idx] + std_path[idx]],
                    [idx, idx],
                    marker='X',
                    s=50,
                    zorder=5,
                    color='#82A582',
                    label='±1 Std'
                )
            else:
                ax.scatter(
                    [mean_path[idx] - std_path[idx], mean_path[idx] + std_path[idx]],
                    [idx, idx],
                    marker='X',
                    s=50,
                    zorder=5,
                    color='#82A582'
                )
        # Plot ±2 Std bounds
        if 2 in std_levels and plot_scatter['2 std']:
            if idx == 0:
                ax.scatter(
                    [mean_path[idx] - 2*std_path[idx], mean_path[idx] + 2*std_path[idx]],
                    [idx, idx],
                    marker='s',
                    s=50,
                    zorder=5,
                    color='#517551',
                    label='±2 Std'
                )
            else:
                ax.scatter(
                    [mean_path[idx] - 2*std_path[idx], mean_path[idx] + 2*std_path[idx]],
                    [idx, idx],
                    marker='s',
                    s=50,
                    zorder=5,
                    color='#517551'
                )
        # Plot ±3 Std bounds
        if 3 in std_levels and plot_scatter['3 std']:
            if idx == 0:
                ax.scatter(
                    [mean_path[idx] - 3*std_path[idx], mean_path[idx] + 3*std_path[idx]],
                    [idx, idx],
                    marker='^',
                    s=50,
                    zorder=5,
                    color='#2F4F2F',
                    label='±3 Std'
                )
            else:
                ax.scatter(
                    [mean_path[idx] - 3*std_path[idx], mean_path[idx] + 3*std_path[idx]],
                    [idx, idx],
                    marker='^',
                    s=50,
                    zorder=5,
                    color='#2F4F2F'
                )
    # Create a clean legend with unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right')
    # Set plot title based on the class
    if class_name is not None:
        ax.set_title(f"SHAP Decision Plot for class {class_name} with Mean Path", fontsize=26)
    else:
        ax.set_title(f"SHAP Decision Plot with Mean Path", fontsize=26)
    plt.show()

