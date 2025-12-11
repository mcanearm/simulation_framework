from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

from typing import Union


def create_plotter_fn(
    plot_class,
    x: str,
    y: str,
    title=None,
    param_filters: dict | None = None,
    **facet_grid_params,
):
    def _drawer(
        evaluation_data: pd.DataFrame, simulation_dir: Union[str, Path, None] = None
    ):
        # TODO: Decide whether this really needs to be here.
        """
        Thin wrapper around seaborn's FacetGrid to plot lineplots across multiple facets. This is not strictly required, and
        arguably, it would just be easier if end users called seaborn directly.
        """
        data = evaluation_data.copy()

        # TODO: add funcitonality to filter data, because we have some stacking going on
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
