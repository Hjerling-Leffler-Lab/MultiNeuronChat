import math

import warnings

from decimal import Decimal

import numpy as np
import xarray as xr

import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import Line2D

from . import MultiNeuronChatObject

def plot_circle_plot(
    axs: plt.Axes,
    cell_cell_counts: dict[tuple[str, str], int],
    cell_types: list[str],
    cell_type_colors: dict[str, str] | list[str] | None = None,
    cell_type_labels_short: dict[str, str] | None = None,
    radius_of_nodes: float = 0.05,
    show_labels: bool = False,
    label_font_size: int = 12,
    legend_axs: plt.Axes | None = None,
    legend_marker_size: float = 13,
    legend_marker_edge_width: float = 1,
    legend_font_size: float = 12,
    legend_n_cols: int = 1,
    title: str | None = None,
    title_font_size: float = 12,
    arrow_thickness_factor: float = 2,
    arrow_head_length: float = 5,
    arrow_head_width: float = 2.5,
    arrow_legend_axs: plt.Axes | None = None,
    arrow_legend_font_size: float = 12,
    arrow_legend_n_cols: int = 1,
    arrow_normalization_factor: float | None = None,
    show_counts: bool = False
) -> None:
    """
    Create a circle plot showing the interactions between different cell-types.

    :param axs: Matplotlib Axes object where the circle plot will be drawn.
    :param cell_cell_counts: Dictionary with keys as tuples of (source_cell_type, receiver_cell_type) and values as the number of connections.
    :param cell_types: List of cell-types to be included in the plot.
    :param cell_type_colors: Dictionary or list of colors for each cell-type. If None, default colors will be used.
    :param cell_type_labels_short: Dictionary mapping cell-types to their short labels for display.
    :param radius_of_nodes: Radius of the nodes representing cell-types.
    :param show_labels: Whether to show cell-type labels on the plot.
    :param label_font_size: Font size for the cell-type labels.
    :param legend_axs: Matplotlib Axes object where the legend will be drawn. If None, no legend is drawn.
    :param legend_marker_size: Size of the markers in the legend.
    :param legend_marker_edge_width: Edge width of the markers in the legend.
    :param legend_font_size: Font size for the legend text.
    :param legend_n_cols: Number of columns in the legend.
    :param title: Title of the plot. If None, no title is set.
    :param title_font_size: Font size for the title.
    :param arrow_thickness_factor: Factor to scale the thickness of the arrows representing connections.
    :param arrow_head_length: Length of the arrow heads.
    :param arrow_head_width: Width of the arrow heads.
    :param arrow_legend_axs: Matplotlib Axes object where the arrow legend will be drawn. If None, no arrow legend is drawn.
    :param arrow_legend_font_size: Font size for the arrow legend text.
    :param arrow_legend_n_cols: Number of columns in the arrow legend.
    :param arrow_normalization_factor: Normalization factor for the arrow thickness. If None, the maximum number of connections is used.
    :param show_counts: Whether to show incoming and outgoing connection counts for each cell-type.

    :return: None
    """
    if cell_type_colors is None:
        cell_type_colors = list(sns.color_palette('husl', len(cell_types)))
    elif len(cell_type_colors) != len(cell_types):
        raise ValueError('The number of colors must match the number of cell-types.')

    # Compute coordinates on a unit-circle for each cell-type
    angle: float = 2 * np.pi / len(cell_types)
    shift: float = 2 * (len(cell_types) * radius_of_nodes) / np.pi
    cell_type_to_coordinates: dict[str, np.array] = {
        cell_type: np.array([shift * np.cos(i * angle), shift * np.sin(i * angle)])
        for i, cell_type in enumerate(cell_types)
    }

    for i, cell_type in enumerate(cell_types):
        color = cell_type_colors[i] if type(cell_type_colors) is list else cell_type_colors[cell_type]

        node = plt.Circle(
            xy=cell_type_to_coordinates[cell_type],
            radius=radius_of_nodes,
            color=color
        )

        axs.add_patch(node)

    # Only iterate through the connections if the cell_cell_counts is not empty
    if len(cell_cell_counts) > 0:
        if arrow_normalization_factor is None:
            arrow_normalization_factor: float = max(cell_cell_counts.values())

        for source_cell_type, receiver_cell_type in cell_cell_counts.keys():
            source_pos: np.array = cell_type_to_coordinates[source_cell_type]
            receiver_pos: np.array = cell_type_to_coordinates[receiver_cell_type]

            # Get the number of connections
            number_of_connections: float = cell_cell_counts[(source_cell_type, receiver_cell_type)]

            if number_of_connections > arrow_normalization_factor:
                # Send out a warning that the normalization factor is too low
                warnings.warn(f'The number of connections between {source_cell_type} and {receiver_cell_type} is higher than the normalization factor. Please choose a value higher or equal to {number_of_connections} to ensure that the arrow thickness is proportional to the number of connections.')

            normalized_number_of_connections: float = (number_of_connections / arrow_normalization_factor) * arrow_thickness_factor

            if source_cell_type == receiver_cell_type:
                length_of_origin_vector: float = np.linalg.norm(source_pos)

                normalized_radius_vector: np.array = (source_pos / length_of_origin_vector) * radius_of_nodes

                angle_to_rotate: float = 80 * (np.pi / 180)
                x, y = normalized_radius_vector
                rotation_clockwise_vector: np.array = np.array([
                    np.cos(-angle_to_rotate) * x - np.sin(-angle_to_rotate) * y,
                    np.sin(-angle_to_rotate) * x + np.cos(-angle_to_rotate) * y
                ])
                rotation_anticlockwise_vector: np.array = np.array([
                    np.cos(angle_to_rotate) * x - np.sin(angle_to_rotate) * y,
                    np.sin(angle_to_rotate) * x + np.cos(angle_to_rotate) * y
                ])

                start_arrow_pos: np.array = source_pos + rotation_anticlockwise_vector
                end_arrow_pos: np.array = receiver_pos + rotation_clockwise_vector

                edge: patches.FancyArrowPatch = patches.FancyArrowPatch(
                    posA=start_arrow_pos,
                    posB=end_arrow_pos,
                    arrowstyle=f'-|>,head_length={arrow_head_length},head_width={arrow_head_width}',
                    connectionstyle=f'arc3,rad={-2}',
                    color='#000000',
                    linewidth=normalized_number_of_connections,
                )

            else:
                connecting_vector: np.array = receiver_pos - source_pos
                length: float = np.linalg.norm(connecting_vector)
                radius_vector: np.array = (connecting_vector / length) * radius_of_nodes

                source_arrow_pos: np.array = source_pos + radius_vector
                receiver_arrow_pos: np.array = receiver_pos - radius_vector

                # Determine direction to change angle of the circle
                theta1: float = np.arctan2(source_arrow_pos[1], source_arrow_pos[0])
                theta2: float = np.arctan2(receiver_arrow_pos[1], receiver_arrow_pos[0])
                delta_theta: float = theta2 - theta1

                if delta_theta > np.pi:
                    delta_theta = delta_theta - 2 * np.pi
                elif delta_theta < -np.pi:
                    delta_theta = delta_theta + 2 * np.pi

                clockwise = 1 if delta_theta < 0 else -2

                # Check if the source cell-type is adjacent to the receiver cell; then ignore the arc
                index_cell_type_a = cell_types.index(source_cell_type)
                index_cell_type_b = cell_types.index(receiver_cell_type)
                if index_cell_type_a == index_cell_type_b + 1 or index_cell_type_a == index_cell_type_b - 1:
                    clockwise = -1

                edge: patches.FancyArrowPatch = patches.FancyArrowPatch(
                    posA=source_arrow_pos,
                    posB=receiver_arrow_pos,
                    arrowstyle=f'-|>,head_length={arrow_head_length},head_width={arrow_head_width}',
                    connectionstyle=f'arc3,rad={clockwise * 0.25}',
                    color='#000000',
                    linewidth=normalized_number_of_connections,
                )

            axs.add_patch(edge)

    if show_labels:
        for cell_type in cell_types:
            label = cell_type_labels_short[cell_type] if cell_type_labels_short is not None else cell_type

            axs.text(
                x=cell_type_to_coordinates[cell_type][0],
                y=cell_type_to_coordinates[cell_type][1],
                s=label,
                ha='center',
                va='center',
                fontsize=label_font_size,
            )

            # If user wants the counts, show them above (incoming) and below (outgoing)
            if show_counts:
                # Sum of incoming
                incoming_count = sum(
                    count
                    for (src, dst), count in cell_cell_counts.items()
                    if dst == cell_type
                )
                # Sum of outgoing
                outgoing_count = sum(
                    count
                    for (src, dst), count in cell_cell_counts.items()
                    if src == cell_type
                )

                # Above the label (incoming)
                axs.text(
                    x=cell_type_to_coordinates[cell_type][0],
                    y=cell_type_to_coordinates[cell_type][1] + radius_of_nodes*0.4,  # small offset upward
                    s=str(incoming_count),
                    ha='center',
                    va='bottom',
                    fontsize=label_font_size,
                    color='black'
                )
                # Below the label (outgoing)
                axs.text(
                    x=cell_type_to_coordinates[cell_type][0],
                    y=cell_type_to_coordinates[cell_type][1] - radius_of_nodes*0.4,  # small offset downward
                    s=str(outgoing_count),
                    ha='center',
                    va='top',
                    fontsize=label_font_size,
                    color='black'
                )

    axs.set_xlim(-shift - radius_of_nodes * 2.5, shift + radius_of_nodes * 2.5)
    axs.set_ylim(-shift - radius_of_nodes * 2.5, shift + radius_of_nodes * 2.5)

    axs.axis('off')

    if title is not None:
        axs.set_title(title, fontsize=title_font_size)

    if legend_axs is not None:
        legend_patches = []

        for i, cell_type in enumerate(cell_types):
            if cell_type_labels_short is not None:
                label = f'{cell_type_labels_short[cell_type]} - {cell_type}'
            else:
                label = cell_type

            color = cell_type_colors[i] if type(cell_type_colors) is list else cell_type_colors[cell_type]

            legend_patch = Line2D(
                [0], [0],
                label=label,
                marker='o',
                markersize=legend_marker_size,
                markerfacecolor=color,
                markeredgecolor='black',
                markeredgewidth=legend_marker_edge_width,
                linestyle='',
            )
            legend_patches.append(legend_patch)

        # center legend
        legend_axs.legend(handles=legend_patches, loc='center', prop={'size': legend_font_size}, ncol=legend_n_cols)

        legend_axs.axis('off')

    if arrow_legend_axs is not None:
        arrow_legend_patches = []

        def get_integer_steps(min_value, max_value, max_steps=5):
            unique_values = np.arange(min_value, max_value + 1)
            num_unique_values = len(unique_values)
            if num_unique_values <= max_steps:
                steps = unique_values
            else:
                indices = np.linspace(0, num_unique_values - 1, num=max_steps)
                indices = np.round(indices).astype(int)
                steps = unique_values[indices]
            return steps.tolist()

        # Check if there are even connections
        if cell_cell_counts.values():
            max_number_of_connections = max(cell_cell_counts.values())
            min_number_of_connections = min(cell_cell_counts.values())

            # Generate integer steps for the legend
            arrow_step_size = get_integer_steps(
                min_number_of_connections,
                max_number_of_connections,
                max_steps=5
            )

            for arrow_number in arrow_step_size:
                normalized_arrow_number = (arrow_number / arrow_normalization_factor) * arrow_thickness_factor

                arrow_legend_patch = Line2D(
                    [0], [0],
                    color='#000000',
                    linewidth=normalized_arrow_number,
                    linestyle='-',
                )
                arrow_legend_patches.append(arrow_legend_patch)

            arrow_legend_axs.legend(
                arrow_legend_patches,
                [f'{int(arrow_number)}' for arrow_number in arrow_step_size],
                loc='center',
                prop={'size': arrow_legend_font_size},
                ncol=arrow_legend_n_cols
            )

            arrow_legend_axs.axis('off')


def plot_p_value_differential_communication_circle_plot(
        axs: plt.Axes,
        mnc_object: MultiNeuronChatObject,
        significance_test_to_use: str,
        significance_threshold: float = 0.05,
        use_adj_p_values: bool = True,
        cell_types: list[str] | None = None,
        cell_type_colors: dict[str, str] | list[str] | None = None,
        cell_type_labels_short: dict[str, str] | None = None,
        radius_of_nodes: float = 0.05,
        show_labels: bool = False,
        label_font_size: int = 12,
        legend_axs: plt.Axes | None = None,
        legend_marker_size: float = 13,
        legend_marker_edge_width: float = 1,
        legend_font_size: float = 12,
        legend_n_cols: int = 1,
        title: str | None = None,
        title_font_size: float = 12,
        arrow_thickness_factor: float = 2,
        arrow_head_length: float = 5,
        arrow_head_width: float = 2.5,
        arrow_legend_axs: plt.Axes | None = None,
        arrow_legend_font_size: float = 12,
        arrow_legend_n_cols: int = 1,
        arrow_normalization_factor: float | None = None,
        show_counts: bool = False
):
    """
    Create a circle plot showing the significant interactions between different cell-types based on p-values from a MultiNeuronChatObject.

    :param axs: Matplotlib Axes object where the circle plot will be drawn.
    :param mnc_object: MultiNeuronChatObject containing the p-values for differential communication analysis.
    :param significance_test_to_use: Name of the significance test to use for plotting (e.g., 'KS', 'Anderson', etc.).
    :param significance_threshold: Significance threshold for p-values to consider an interaction significant (default: 0.05).
    :param use_adj_p_values: Whether to use adjusted p-values (default: True).
    :param cell_types: List of cell-types to be included in the plot.
    :param cell_type_colors: Dictionary or list of colors for each cell-type. If None, default colors will be used.
    :param cell_type_labels_short: Dictionary mapping cell-types to their short labels for display.
    :param radius_of_nodes: Radius of the nodes representing cell-types.
    :param show_labels: Whether to show cell-type labels on the plot.
    :param label_font_size: Font size for the cell-type labels.
    :param legend_axs: Matplotlib Axes object where the legend will be drawn. If None, no legend is drawn.
    :param legend_marker_size: Size of the markers in the legend.
    :param legend_marker_edge_width: Edge width of the markers in the legend.
    :param legend_font_size: Font size for the legend text.
    :param legend_n_cols: Number of columns in the legend.
    :param title: Title of the plot. If None, no title is set.
    :param title_font_size: Font size for the title.
    :param arrow_thickness_factor: Factor to scale the thickness of the arrows representing connections.
    :param arrow_head_length: Length of the arrow heads.
    :param arrow_head_width: Width of the arrow heads.
    :param arrow_legend_axs: Matplotlib Axes object where the arrow legend will be drawn. If None, no arrow legend is drawn.
    :param arrow_legend_font_size: Font size for the arrow legend text.
    :param arrow_legend_n_cols: Number of columns in the arrow legend.
    :param arrow_normalization_factor: Normalization factor for the arrow thickness. If None, the maximum number of connections is used.
    :param show_counts: Whether to show incoming and outgoing connection counts for each cell-type.

    :return: None
    """
    if cell_types is None:
        # Get the number of cell-types
        cell_types: list[str] = list(set(mnc_object.source_cell_types).union(set(mnc_object.receiver_cell_types)))
        cell_types.sort()
    else:
        # Check if cell_types are valid
        if not all([cell_type in mnc_object.source_cell_types or cell_type in mnc_object.receiver_cell_types for cell_type in cell_types]):
            not_present_cell_types: set[str] = set(cell_types).difference(set(mnc_object.source_cell_types).union(set(mnc_object.receiver_cell_types)))
            raise ValueError(f'All cell-types must be present in the MultiNeuronChatObject!'
                             f'The following cell_types are not present: {not_present_cell_types}')


    if use_adj_p_values and significance_test_to_use not in mnc_object.p_values_adj.keys():
        raise ValueError(f'The significance test "{significance_test_to_use}" was not run.')
    elif significance_test_to_use not in mnc_object.p_values.keys():
        raise ValueError(f'The significance test "{significance_test_to_use}" was not run.')

    # Get the p-values to use for creating the graph
    if use_adj_p_values:
        p_values: xr.DataArray = mnc_object.p_values_adj[significance_test_to_use]
    else:
        p_values: xr.DataArray = mnc_object.p_values[significance_test_to_use]

    significant_interactions: xr.DataArray = p_values < significance_threshold

    # Sum over the ligand target interactions to get the counts of significant interactions
    significant_interactions_counts: xr.DataArray = significant_interactions.sum(dim='interaction')
    significant_interactions_counts_idx: np.array = np.where(significant_interactions_counts > 0)

    # Get the cell-cell counts
    source_receiver_pair_counts_dict: dict[tuple[str, str], int] = {
        (significant_interactions.coords['source'][source_cell_idx].values.item(), significant_interactions.coords['receiver'][receiver_cell_idx].values.item()): significant_interactions_counts[source_cell_idx, receiver_cell_idx].values.item()
        for source_cell_idx, receiver_cell_idx in zip(*significant_interactions_counts_idx)
    }

    # Get the cell-type colors
    plot_circle_plot(
        axs=axs,
        cell_cell_counts=source_receiver_pair_counts_dict,
        cell_types=cell_types,
        cell_type_colors=cell_type_colors,
        cell_type_labels_short=cell_type_labels_short,
        radius_of_nodes=radius_of_nodes,
        show_labels=show_labels,
        label_font_size=label_font_size,
        legend_axs=legend_axs,
        legend_marker_size=legend_marker_size,
        legend_marker_edge_width=legend_marker_edge_width,
        legend_font_size=legend_font_size,
        legend_n_cols=legend_n_cols,
        title=title,
        title_font_size=title_font_size,
        arrow_thickness_factor=arrow_thickness_factor,
        arrow_legend_axs=arrow_legend_axs,
        arrow_legend_font_size=arrow_legend_font_size,
        arrow_head_width=arrow_head_width,
        arrow_head_length=arrow_head_length,
        arrow_legend_n_cols=arrow_legend_n_cols,
        arrow_normalization_factor=arrow_normalization_factor,
        show_counts=show_counts
    )


def plot_wasserstein_ranked_differential_communication_circle_plot(
        axs: plt.Axes,
        wasserstein_distances: xr.DataArray,
        top_n_edges: int = 10,
        cell_types: list[str] | None = None,
        cell_type_colors: dict[str, str] | list[str] | None = None,
        cell_type_labels_short: dict[str, str] | None = None,
        radius_of_nodes: float = 0.05,
        show_labels: bool = False,
        label_font_size: int = 12,
        legend_axs: plt.Axes | None = None,
        legend_marker_size: float = 13,
        legend_marker_edge_width: float = 1,
        legend_font_size: float = 12,
        legend_n_cols: int = 1,
        title: str | None = None,
        title_font_size: float = 12,
        arrow_thickness_factor: float = 2,
        arrow_head_length: float = 5,
        arrow_head_width: float = 2.5,
        arrow_legend_axs: plt.Axes | None = None,
        arrow_legend_font_size: float = 12,
        arrow_legend_n_cols: int = 1,
        arrow_normalization_factor: float | None = None,
        show_counts: bool = False,
):
    """
    Create a circle plot showing the top N interactions between different cell-types based on Wasserstein distances.

    :param axs: Matplotlib Axes object where the circle plot will be drawn.
    :param wasserstein_distances: xarray DataArray containing the Wasserstein distances between cell-types.
    :param top_n_edges: Number of top interactions (edges) to display in the plot (default: 10).
    :param cell_types: List of cell-types to be included in the plot.
    :param cell_type_colors: Dictionary or list of colors for each cell-type. If None, default colors will be used.
    :param cell_type_labels_short: Dictionary mapping cell-types to their short labels for display.
    :param radius_of_nodes: Radius of the nodes representing cell-types.
    :param show_labels: Whether to show cell-type labels on the plot.
    :param label_font_size: Font size for the cell-type labels.
    :param legend_axs: Matplotlib Axes object where the legend will be drawn. If None, no legend is drawn.
    :param legend_marker_size: Size of the markers in the legend.
    :param legend_marker_edge_width: Edge width of the markers in the legend.
    :param legend_font_size: Font size for the legend text.
    :param legend_n_cols: Number of columns in the legend.
    :param title: Title of the plot. If None, no title is set.
    :param title_font_size: Font size for the title.
    :param arrow_thickness_factor: Factor to scale the thickness of the arrows representing connections.
    :param arrow_head_length: Length of the arrow heads.
    :param arrow_head_width: Width of the arrow heads.
    :param arrow_legend_axs: Matplotlib Axes object where the arrow legend will be drawn. If None, no arrow legend is drawn.
    :param arrow_legend_font_size: Font size for the arrow legend text.
    :param arrow_legend_n_cols: Number of columns in the arrow legend.
    :param arrow_normalization_factor: Normalization factor for the arrow thickness. If None, the maximum number of connections is used.
    :param show_counts: Whether to show incoming and outgoing connection counts for each cell-type.

    :return: None
    """
    if cell_types is None:
        # Get the number of cell-types
        cell_types: list[str] = list(set(wasserstein_distances.coords['source'].values.tolist()).union(
            set(wasserstein_distances.coords['receiver'].values.tolist())))
        cell_types.sort()
    else:
        # Check if cell_types are valid
        if not all([cell_type in wasserstein_distances.coords['source'].values.tolist() or cell_type in
                    wasserstein_distances.coords['receiver'].values.tolist() for cell_type in cell_types]):
            not_present_cell_types: set[str] = set(cell_types).difference(
                set(wasserstein_distances.coords['source'].values.tolist()).union(
                    set(wasserstein_distances.coords['receiver'].values.tolist())))
            raise ValueError(f'All cell-types must be present in the MultiNeuronChatObject!'
                             f'The following cell_types are not present: {not_present_cell_types}')

    if cell_type_colors is None:
        cell_type_colors = list(sns.color_palette('husl', len(cell_types)))
    elif len(cell_type_colors) != len(cell_types):
        raise ValueError('The number of colors must match the number of cell-types.')

    # Globally sorted wasserstein distances indices
    sorted_wasserstein_distances_idxs = np.argsort(wasserstein_distances.values.flatten())[::-1]
    # Remove NaNs
    sorted_wasserstein_distances_idxs = sorted_wasserstein_distances_idxs[
        ~np.isnan(wasserstein_distances.values.flatten()[sorted_wasserstein_distances_idxs])]

    # Get the top_n_edges indices
    top_n_edges_idxs = sorted_wasserstein_distances_idxs[:top_n_edges]

    # Unravel the indices
    source_idxs, receiver_idxs, interaction_idxs = np.unravel_index(top_n_edges_idxs, wasserstein_distances.shape)

    # Get the source and receiver cell-types
    source_cell_types = wasserstein_distances.coords['source'].values[source_idxs]
    receiver_cell_types = wasserstein_distances.coords['receiver'].values[receiver_idxs]
    interaction_cell_types = wasserstein_distances.coords['interaction'].values[interaction_idxs]

    df: pd.DataFrame = pd.DataFrame({
        'source': source_cell_types,
        'receiver': receiver_cell_types,
        'interaction': interaction_cell_types,
        'wasserstein distance': wasserstein_distances.values.flatten()[top_n_edges_idxs]
    })

    # Count the source-receiver pairs
    source_receiver_pair_counts: pd.Series = df.groupby(['source', 'receiver']).size()

    # Convert series of source-receiver pair counts to a dictionary of tuples to int
    source_receiver_pair_counts_dict: dict[tuple[str, str], int] = source_receiver_pair_counts.to_dict()

    plot_circle_plot(
        axs=axs,
        cell_cell_counts=source_receiver_pair_counts_dict,
        cell_types=cell_types,
        cell_type_colors=cell_type_colors,
        cell_type_labels_short=cell_type_labels_short,
        radius_of_nodes=radius_of_nodes,
        show_labels=show_labels,
        label_font_size=label_font_size,
        legend_axs=legend_axs,
        legend_marker_size=legend_marker_size,
        legend_marker_edge_width=legend_marker_edge_width,
        legend_font_size=legend_font_size,
        legend_n_cols=legend_n_cols,
        title=title,
        title_font_size=title_font_size,
        arrow_thickness_factor=arrow_thickness_factor,
        arrow_legend_axs=arrow_legend_axs,
        arrow_legend_font_size=arrow_legend_font_size,
        arrow_head_width=arrow_head_width,
        arrow_head_length=arrow_head_length,
        arrow_legend_n_cols=arrow_legend_n_cols,
        arrow_normalization_factor=arrow_normalization_factor,
        show_counts=show_counts
    )


def plot_aula_medica_plot(
        axs: plt.Axes,
        mnc_object: MultiNeuronChatObject,
        wasserstein_distance_matrix: xr.DataArray,
        statistical_test: str = 'KS',
        use_adjusted_p_values: bool = True,
        cell_types: list[str] | list[tuple[str, list[str]]] | None = None,
        cell_type_labels_short: dict[str, str] | None = None,
        ligand_target_interactions: list[str] | None = None,
        scale: float = 1.0,
        triangle_line_width: float = 0.5,
        source_cell_type_split_line_width: float = 1.0,
        marker_size: float = 10,
        tick_font_size: float = 20,
        wasserstein_color_map: str = 'seagreen',
        p_value_color_map: str = 'salmon',
        not_tested_color: str = '#CCCCCC',
        wasserstein_legend_axs: plt.Axes | None = None,
        p_value_legend_axs: plt.Axes | None = None,
        wasserstein_rounding: int | None = None,
        p_value_rounding: int | None = None,
        annotate_source_cell_types: bool = True,
        annotate_receiver_cell_types: bool = True,
        annotate_interactions: bool = True,
        annotate_significance: bool = True,
        significance_threshold: float = 0.05,
        max_log_p_value: float | None = None,
        max_wasserstein_distance: float | None = None,
):
    """
    Create an Aula Medica plot showing the Wasserstein distances and p-values for differential communication analysis.

    :param axs: Matplotlib Axes object where the Aula Medica plot will be drawn.
    :param mnc_object: MultiNeuronChatObject containing the p-values for differential communication analysis.
    :param wasserstein_distance_matrix: xarray DataArray containing the Wasserstein distances between cell-types.
    :param statistical_test: Name of the statistical test to use for plotting (e.g 'KS', 'Anderson', etc.).
    :param use_adjusted_p_values: Whether to use adjusted p-values (default: True).
    :param cell_types: List of cell-types or list of tuples (source_cell_type, list of receiver_cell_types) to be included in the plot. If a list is provided all interactions between these cells will be plotted. If a list of tuples is given, the first entry of the tuple represents the sender-cell-type and the second entry the list of receiver-cell-types. If None, all cell-types will be used, and the second entry is a list of all the receiving cell-types to be plotted.
    :param cell_type_labels_short: Dictionary mapping cell-types to their short labels for display.
    :param ligand_target_interactions: List of ligand-target interactions to be included in the plot.
    :param scale: Scale factor for the plot.
    :param triangle_line_width: Line width for the triangles in the plot.
    :param source_cell_type_split_line_width: Line width for the split lines between source cell-types.
    :param marker_size: Size of the markers for annotations.
    :param tick_font_size: Font size for the tick labels.
    :param wasserstein_color_map: Color map for the Wasserstein distances.
    :param p_value_color_map: Color map for the p-values.
    :param not_tested_color: Color for interactions that were not tested.
    :param wasserstein_legend_axs: Matplotlib Axes object where the Wasserstein legend will be drawn. If None, no legend is drawn.
    :param p_value_legend_axs: Matplotlib Axes object where the p-value legend will be drawn. If None, no legend is drawn.
    :param wasserstein_rounding: Number of decimal places to round the Wasserstein distances in the legend. If None, no rounding is applied.
    :param p_value_rounding: Number of decimal places to round the p-values in the legend. If None, no rounding is applied.
    :param annotate_source_cell_types: Whether to annotate the source cell-types.
    :param annotate_receiver_cell_types: Whether to annotate the receiver cell-types.
    :param annotate_interactions: Whether to annotate the ligand-target interactions.
    :param annotate_significance: Whether to annotate the significance based on the significance threshold.
    :param significance_threshold: Significance threshold for p-values to consider an interaction significant (default: 0.05).
    :param max_log_p_value: Maximum -log10(p-value) to use for normalization. This is especially useful if you plot several lines of the Aula Medica plot to have the same color scale. If None, the maximum value in the data is used.
    :param max_wasserstein_distance: Maximum Wasserstein distance to use for normalization. If None, the maximum value in the data is used.

    :return: None
    """
    # If no cell-types are provided -> use all cell-types
    all_cell_types: list[str] = list(set(mnc_object.source_cell_types + mnc_object.receiver_cell_types))
    all_cell_types.sort()
    if cell_types is None:
        cell_types: list[tuple[str, list[str]]] = [
            (cell_type, all_cell_types)
            for cell_type in all_cell_types
        ]
    else:  # If cell-types are provided -> check if the selection is valid
        if type(cell_types[0]) is str:
            # The provided type is a list of strings -> first check if all cell-types are valid, then convert to a list of tuples
            if not all([cell_type in all_cell_types for cell_type in cell_types]):
                raise ValueError('Invalid cell types')

            cell_types: list[tuple[str, list[str]]] = [
                (cell_type, cell_types)
                for cell_type in cell_types
            ]
        elif type(cell_types[0]) is tuple:
            # The provided type is a list of tuples -> check if all cell-types are valid
            selected_cell_types: list[str] = list(set(
                [cell_type for cell_type, _ in cell_types] +
                [cell_type for _, receiver_cell_types in cell_types for cell_type in receiver_cell_types]
            ))

            if not all([cell_type in all_cell_types for cell_type in selected_cell_types]):
                raise ValueError('Invalid cell types')

    # If no ligand-target interactions are provided -> use all ligand-target interactions
    if ligand_target_interactions is None:
        ligand_target_interactions = mnc_object.interaction_names
    else:  # If ligand-target interactions are provided -> check if the selection is valid
        if not all([interaction in mnc_object.interaction_names for interaction in ligand_target_interactions]):
            raise ValueError('Invalid ligand-target interactions')

    if statistical_test not in ['KS', 'Anderson', 'CVM', 'Wilcoxon', 'MannWhitneyU']:
        raise ValueError('Invalid statistical test')
    elif (not use_adjusted_p_values) and (statistical_test not in mnc_object.p_values.keys()):
        raise ValueError('The p-values have not been yet calculated for the selected statistical test')
    elif use_adjusted_p_values and (statistical_test not in mnc_object.p_values_adj.keys()):
        raise ValueError('The adjusted p-values have not been yet calculated for the selected statistical test')

    n_x: int = sum([len(cell_type[1]) for cell_type in cell_types])
    n_y: int = len(ligand_target_interactions)

    p_values: xr.DataArray = mnc_object.p_values_adj[statistical_test] if use_adjusted_p_values else \
    mnc_object.p_values[statistical_test]
    neg_log_p_values: xr.DataArray = -np.log10(p_values)

    if max_log_p_value is None:
        max_log_p_value: float = np.nanmax(neg_log_p_values.to_numpy())

    if max_wasserstein_distance is None:
        max_wasserstein_distance: float = np.nanmax(wasserstein_distance_matrix.to_numpy())

    # Define the points of the triangle to draw
    uniform_lower_triangle_corners_x: np.array = np.array([0, 1, 0, 0]) * scale
    uniform_lower_triangle_corners_y: np.array = np.array([0, 0, 1, 0]) * scale
    uniform_upper_triangle_corners_x: np.array = np.array([1, 1, 0, 1]) * scale
    uniform_upper_triangle_corners_y: np.array = np.array([0, 1, 1, 0]) * scale

    triangle_positions_x: np.array = np.arange(n_x) * scale
    triangle_positions_y: np.array = np.arange(n_y) * scale

    triangle_coordinates: np.array = np.stack(np.meshgrid(triangle_positions_x, triangle_positions_y)).transpose(2, 1,
                                                                                                                 0)

    wasserstein_cmap: sns.color_palette = sns.light_palette(wasserstein_color_map, as_cmap=True)
    p_value_cmap: sns.color_palette = sns.light_palette(p_value_color_map, as_cmap=True)

    # Iterate through
    x_pos: int = 0
    for source_idx, (source_cell_type, receiver_cell_types) in enumerate(cell_types):
        for receiver_idx, receiver_cell_type in enumerate(receiver_cell_types):
            for interaction_idx, interaction in enumerate(ligand_target_interactions):
                y_pos: int = interaction_idx

                p_value: float = p_values.loc[
                    {'source': source_cell_type, 'receiver': receiver_cell_type,
                     'interaction': interaction}].values.item()
                neg_log_p_value: float = neg_log_p_values.loc[
                    {'source': source_cell_type, 'receiver': receiver_cell_type,
                     'interaction': interaction}].values.item()
                wasserstein_distance: float = wasserstein_distance_matrix.loc[
                    {'source': source_cell_type, 'receiver': receiver_cell_type,
                     'interaction': interaction}].values.item()

                normalized_neg_log_p_value: float = neg_log_p_value / max_log_p_value
                normalized_wasserstein_distance: float = wasserstein_distance / max_wasserstein_distance

                x_start, y_start = triangle_coordinates[x_pos, y_pos]

                axs.plot(
                    uniform_lower_triangle_corners_x + x_start,
                    uniform_lower_triangle_corners_y + y_start,
                    color='black',
                    linewidth=triangle_line_width
                )
                axs.plot(
                    uniform_upper_triangle_corners_x + x_start,
                    uniform_upper_triangle_corners_y + y_start,
                    color='black',
                    linewidth=triangle_line_width
                )

                if math.isnan(normalized_neg_log_p_value):
                    p_value_color: str = not_tested_color
                else:
                    p_value_color: tuple = p_value_cmap(normalized_neg_log_p_value)

                if math.isnan(wasserstein_distance):
                    wasserstein_color: str = not_tested_color
                else:
                    wasserstein_color: tuple = wasserstein_cmap(normalized_wasserstein_distance)

                axs.fill(
                    uniform_lower_triangle_corners_x + x_start,
                    uniform_lower_triangle_corners_y + y_start,
                    color=wasserstein_color,
                )
                axs.fill(
                    uniform_upper_triangle_corners_x + x_start,
                    uniform_upper_triangle_corners_y + y_start,
                    color=p_value_color,
                )

                if annotate_significance and p_value < significance_threshold:
                    mid_point_x = mid_point_y = 0.75 * scale
                    axs.scatter(
                        x=x_start + mid_point_x,
                        y=y_start + mid_point_y,
                        marker='.',
                        color='black',
                        s=marker_size,
                    )

            x_pos += 1

    # Annotate x-axis
    axs.set_xticks(triangle_positions_x + (scale / 2))
    if annotate_receiver_cell_types:
        receiver_cell_type_list: list[str] = [cell_type for _, receiver_cell_types in cell_types for cell_type in
                                              receiver_cell_types]
        if cell_type_labels_short is not None:
            receiver_cell_type_list = [cell_type_labels_short[cell_type] for cell_type in receiver_cell_type_list]

        axs.set_xticklabels(
            receiver_cell_type_list,
            fontsize=tick_font_size,
            rotation=90
        )
    else:
        axs.set_xticklabels([])

    if annotate_source_cell_types:
        current_count_cell_types: int = 0
        for i, (source_cell_type, receiver_cell_types) in enumerate(cell_types):
            n_receiver_cell_types: int = len(receiver_cell_types)
            label_x_position = (current_count_cell_types + (n_receiver_cell_types / 2)) * scale

            current_count_cell_types += n_receiver_cell_types

            if cell_type_labels_short is not None:
                source_cell_type = cell_type_labels_short[source_cell_type]

            # Plot text at the top of the plot
            axs.text(
                label_x_position,
                n_y * scale + 0.5,
                source_cell_type,
                fontsize=tick_font_size,
                ha='center'
            )

    # Plot lines that separate the cell-types
    current_count_cell_types: int = 0
    for i, (source_cell_type, receiver_cell_types) in enumerate(cell_types[:-1]):
        current_count_cell_types += len(receiver_cell_types)
        axs.plot(
            [current_count_cell_types * scale, current_count_cell_types * scale],
            [0, n_y * scale],
            color='black',
            linewidth=source_cell_type_split_line_width
        )

    # Annotate y-axis
    axs.set_yticks(triangle_positions_y + (scale / 2))
    if annotate_interactions:
        axs.set_yticklabels(
            ligand_target_interactions,
            fontsize=tick_font_size
        )
    else:
        axs.set_yticklabels([])

    axs.grid(False)

    axs.set_xlim(0, n_x * scale)
    axs.set_ylim(0, n_y * scale)

    # Remove borders
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.spines['left'].set_visible(False)

    axs.set_aspect('equal')

    if wasserstein_legend_axs is not None:
        # Define the normalization from 0 to 1
        norm = mpl.colors.Normalize(vmin=0, vmax=max_wasserstein_distance)
        wasserstein_cb = mpl.colorbar.ColorbarBase(
            wasserstein_legend_axs,  # The axis to draw the colorbar on
            cmap=wasserstein_cmap,  # Your custom colormap
            norm=norm,  # Normalization from 0 to 1
            orientation='vertical'  # Orientation of the colorbar ('vertical' or 'horizontal')
        )

        wasserstein_tick_labels = [0, max_wasserstein_distance / 2, max_wasserstein_distance]
        if wasserstein_rounding is not None:
            wasserstein_tick_labels = [round(label, wasserstein_rounding) for label in wasserstein_tick_labels]

        # Optionally, set the label and ticks
        wasserstein_cb.set_label('Wasserstein distance')
        wasserstein_cb.set_ticks(wasserstein_tick_labels)  # Set custom tick positions
        wasserstein_cb.set_ticklabels(wasserstein_tick_labels)  # Set custom tick labels

    if p_value_legend_axs is not None:
        # Define the normalization from 0 to 1
        norm = mpl.colors.Normalize(vmin=0, vmax=max_log_p_value)

        # Create the colorbar on your custom axis
        p_val_cb = mpl.colorbar.ColorbarBase(
            p_value_legend_axs,  # The axis to draw the colorbar on
            cmap=p_value_cmap,  # Your custom colormap
            norm=norm,  # Normalization from 0 to 1
            orientation='vertical'  # Orientation of the colorbar ('vertical' or 'horizontal')
        )

        p_value_tick_labels = [0, max_log_p_value / 2, max_log_p_value]
        if p_value_rounding is not None:
            p_value_tick_labels = [round(label, p_value_rounding) for label in p_value_tick_labels]

        # Optionally, set the label and ticks
        if use_adjusted_p_values:
            p_val_cb.set_label('-log10(adjusted p-value)')
        else:
            p_val_cb.set_label('-log10(p-value)')

        p_val_cb.set_ticks(p_value_tick_labels)  # Set custom tick positions
        p_val_cb.set_ticklabels(p_value_tick_labels)


def plot_communication_score_distribution(
        axs: plt.Axes,
        mnc_object: MultiNeuronChatObject,
        cell_type_pair: tuple[str, str],
        ligand_target_interaction: str,
        n_bins: int = 25,
        plot_p_value: bool = False,
        use_adjusted_p_value: bool = True,
        statistical_test: str | None = None,
        show_legend: bool = False,
        legend_position: str = 'best',
        legend_axs: plt.Axes | None = None,
        condition_colors: tuple[str, str] | None = None,
        alpha: float = 0.5,
        title_font_size: float = 20,
        annotation_font_size: float = 20,
        tick_font_size: float = 20,
        p_value_font_size: float = 15,
        legend_font_size: float = 20,
        min_x: float = 0,
        max_x: float = -1,
        title: str = '',
):
    """
    Plot the distribution of communication scores between two cell types for a specific ligand-target interaction.

    :param axs: Matplotlib Axes object where the histogram will be drawn.
    :param mnc_object: MultiNeuronChatObject containing the communication scores.
    :param cell_type_pair: Tuple containing the source and receiver cell types.
    :param ligand_target_interaction: Name of the ligand-target interaction to plot.
    :param n_bins: Number of bins for the histogram (default: 25).
    :param plot_p_value: Whether to plot the p-value on the histogram (default: False).
    :param use_adjusted_p_value: Whether to use adjusted p-values (default: True).
    :param statistical_test: Name of the statistical test to use for p-value retrieval (e.g., 'KS', 'Anderson', etc.). Required if plot_p_value is True.
    :param show_legend: Whether to show the legend (default: False).
    :param legend_position: Position of the legend (default: 'best').
    :param legend_axs: Matplotlib Axes object where the legend will be drawn. If None, the legend will be drawn on the main axes.
    :param condition_colors: Tuple containing the colors for the two conditions. If None, default colors will be used.
    :param alpha: Transparency level for the histogram bars (default: 0.5).
    :param title_font_size: Font size for the plot title (default: 20).
    :param annotation_font_size: Font size for the axis labels (default: 20).
    :param tick_font_size: Font size for the tick labels (default: 20).
    :param p_value_font_size: Font size for the p-value text (default: 15).
    :param legend_font_size: Font size for the legend text (default: 20).
    :param min_x: Minimum x-axis value for the histogram (default: 0).
    :param max_x: Maximum x-axis value for the histogram. If -1, it will be set to the maximum communication score (default: -1).
    :param title: Title of the plot (default: '').

    :return: None
    """
    if plot_p_value:
        if statistical_test is None:
            raise ValueError("You selected to plot the p-value of this comparision, but have not selected a statistical test! Please select a valid statistical test!")

        if use_adjusted_p_value and (statistical_test not in mnc_object.p_values_adj.keys()):
            raise ValueError(f'The statistical test "{statistical_test}" is not available in the current MultiNeuronChat adjusted p-values! Please select one of the valid tests: {mnc_object.p_values_adj.keys()}')
        elif (not use_adjusted_p_value) and (statistical_test not in mnc_object.p_values.keys()):
            raise ValueError(f'The statistical test "{statistical_test}" is not available in the current MultiNeuronChat p-values! Please select one of the valid tests: {mnc_object.p_values.keys()}')

    # Check if the selected cell-type pair and ligand-target pair exists in the MultiNeuronChat object
    mnc_cell_types: set[str] = set(mnc_object.source_cell_types + mnc_object.receiver_cell_types)

    if (not cell_type_pair[0] in mnc_cell_types) or (not cell_type_pair[1] in mnc_cell_types):
        raise ValueError(f'One of the two selected cell-types {cell_type_pair} is not valid! Available cell-types: {mnc_cell_types}')

    if ligand_target_interaction not in mnc_object.interaction_names:
        raise ValueError(f'The selected ligand-target interaction {ligand_target_interaction} is not valid! Available interactions: {mnc_object.interaction_names}')

    condition_a_communication_scores: np.array = mnc_object.communication_score_dict[mnc_object.condition_names[0]].sel(
        source=cell_type_pair[0],
        receiver=cell_type_pair[1],
        interaction=ligand_target_interaction,
    ).to_numpy()
    condition_b_communication_scores: np.array = mnc_object.communication_score_dict[mnc_object.condition_names[1]].sel(
        source=cell_type_pair[0],
        receiver=cell_type_pair[1],
        interaction=ligand_target_interaction,
    ).to_numpy()

    condition_a_communication_scores = condition_a_communication_scores[~np.isnan(condition_a_communication_scores)]
    condition_b_communication_scores = condition_b_communication_scores[~np.isnan(condition_b_communication_scores)]

    # Uniform binning
    if max_x < 0:
        max_x = max(np.max(condition_a_communication_scores), np.max(condition_b_communication_scores))

    bins = np.linspace(min_x, max_x, n_bins + 1)

    axs.hist(
        x=condition_a_communication_scores,
        bins=bins,
        color=condition_colors[0],
        alpha=alpha,
        label=mnc_object.condition_names[0]
    )
    axs.hist(
        x=condition_b_communication_scores,
        bins=bins,
        color=condition_colors[1],
        alpha=alpha,
        label=mnc_object.condition_names[1]
    )

    top_y_value: float = axs.get_ylim()[1]
    y_ticks: np.array = np.arange(0, top_y_value)
    axs.set_yticks(y_ticks)
    axs.set_yticklabels(y_ticks)

    axs.tick_params(axis='both', which='major', labelsize=tick_font_size)

    if plot_p_value:
        if use_adjusted_p_value:
            p_value: float = mnc_object.p_values_adj[statistical_test].sel(
                source=cell_type_pair[0],
                receiver=cell_type_pair[1],
                interaction=ligand_target_interaction,
            ).values.tolist()

        else:
            p_value: float = mnc_object.p_values[statistical_test].sel(
                source=cell_type_pair[0],
                receiver=cell_type_pair[1],
                interaction=ligand_target_interaction,
            ).values.tolist()
        axs.text(
            x=0.8,
            y=0.8,
            s=f'{"adj. " if use_adjusted_p_value else ""}p-value:\n{Decimal(p_value):.4E}',
            ha='center',
            fontsize=p_value_font_size,
            transform=axs.transAxes
        )


    axs.set_ylabel('#Donors', fontsize=annotation_font_size)
    axs.set_xlabel('Communication score', fontsize=annotation_font_size)
    axs.set_title(title, fontsize=title_font_size)

    if show_legend:
        if legend_axs is not None:
            legend_axs.plot([], [], marker='s', color=condition_colors[0], label=mnc_object.condition_names[0])
            legend_axs.plot([], [], marker='s', color=condition_colors[0], label=mnc_object.condition_names[1])
            legend_axs.legend(loc='center', fontsize=legend_font_size)
        else:
            axs.legend(loc=legend_position, fontsize=legend_font_size)
