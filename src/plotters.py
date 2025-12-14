from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import logging

from typing import Union


logger = logging.getLogger(__name__)


def create_plotter_fn(
    plot_class,
    x: str,
    y: str,
    title=None,
    param_filters: dict | None = None,
    **facet_grid_params,
):
    """
    Factory function to create plotting functions that use seaborn's FacetGrid
    to create plots across multiple simulation facets. This little wrapper
    is not, frankly, particularly necessary, but it does make it a bit easier to
    create consistent plots across different simulation runs. Custom plotters
    are fully supported; effectively you just need to provide a plotting
    function which takes the arbitrary evaluation_data and, optionally, the
    simulation directory, as arguments, and outputs the resulting plots.

    Note: We may be able to make this an arbitrary class, but honestly,
    there's no reason to throw decorators on it because we just use the provided
    targets in the other functions to decide what to plot on Y

    Args:
        plot_class: A seaborn plotting function, e.g., sns.lineplot, sns.scatterplot.
        x (str): The column name to use for the x-axis.
        y (str): The column name to use for the y-axis.
        title (str, optional): Title for the entire plot. Defaults to None.
        param_filters (dict, optional): A dictionary of parameter filters to apply to the data before plotting.
            Defaults to None.
        **facet_grid_params: Additional keyword arguments to pass to seaborn's FacetGrid.
    Returns:
        function: A plotting function that takes evaluation_data and an optional simulation_dir.
    """

    def _drawer(
        evaluation_data: pd.DataFrame, simulation_dir: Union[str, Path, None] = None
    ):
        # TODO: Decide whether this really needs to be here.
        """
        Thin wrapper around seaborn's FacetGrid to plot lineplots across multiple facets. This is not strictly required, and
        arguably, it would just be easier if end users called seaborn directly.
        """
        data: pd.DataFrame = evaluation_data.copy()
        if param_filters:
            # Gemini generated
            # Initialize a boolean series where all rows are True
            filter_mask = pd.Series(True, index=data.index)

            for col, value in param_filters.items():
                if col not in data.columns:
                    # Optional: Add a warning or error if the column is missing
                    logging.warning(
                        f"Warning: Filter column '{col}' not found in data. Skipping."
                    )
                    continue

                if pd.isna(value):
                    # Handle np.nan or float('nan'): Select rows where the data column is also NaN
                    # This is done using data[col].isna() or pd.isna(data[col])
                    col_mask = data[col].isna()
                else:
                    # Handle regular values: Select rows where the data column equals the filter value
                    col_mask = data[col] == value

                # Combine the current column's mask with the overall filter mask
                filter_mask = filter_mask & col_mask

            # Apply the combined filter mask to the data
            data = data[filter_mask]

        # Check if any data remains after filtering
        if data.empty:
            logging.warning(
                "Warning: Data is empty after applying filters. Skipping plot generation."
            )
            return  # Exit the drawer function early

        # End Gemini generation

        grid = sns.FacetGrid(data, **facet_grid_params)
        grid.map_dataframe(plot_class, x=x, y=y)
        grid.add_legend()

        if title:
            plt.suptitle(title)
        if simulation_dir is not None:
            fig_dir = Path(simulation_dir) / "plots"
            fig_dir.mkdir(parents=True, exist_ok=True)
            save_path = fig_dir / f"{plot_class.__name__}_{x}_vs_{y}.png"
            plt.savefig(save_path)

    return _drawer
