import pandas as pd

from typing import Any, Dict, Optional, Sequence, Union

from copy import deepcopy
from IPython.display import display, HTML

from gca.config import SEED

from gca.data import get_data
from gca.evaluate import causal_determinants, evaluate_Z, evaluate_pag
from gca.model import get_pipeline, train_and_apply
from gca.pag import create_pag
from gca.result import Result
from gca.utils import title, huggingface_login


def generate_and_evaluate(
        df: pd.DataFrame,
        alpha: float = 0.05,
        depth: int = -1,
        times: int = 5,
        percentage: float = 0.8,
        n_trials: int = 0,
        continuous: Sequence[Union[int, str]] = [],
        continuous_cuts: int = 10,
        categorical_max: Optional[int] = 10,
        multiprocess: bool = True,
        color: bool = False,
        verbosity_level: int = 1,
        seed: int = SEED,
        ) -> Result:
    """Generate a global explanatory graph with GCA, and perform the automatic evaluation steps.

    Args:
        df (pd.DataFrame): Dataset for evaluation.
        alpha (float, optional): Chi-Squared statistic. Defaults to 0.05.
        depth (int, optional): Maximum conditional independence depth for FCI (-1 = unlimited). Defaults to -1.
        times (int, optional): Number of neighborhood graphs for structural fit and stability. Defaults to 5.
        percentage (float, optional): Percentage of data for each neighborhood graph. Defaults to 0.8.
        n_trials (int, optional): Number of trials for causal determinant estimation (typically 50). Defaults to 0.
        continuous (Sequence[Union[int, str]], optional): Names of continuous features, to cut. Defaults to [].
        continuous_cuts (int, optional): Number of cuts for continuous features. Defaults to 10.
        categorical_max (Optional[int], optional): Maximum number of categories in categorical variables. Defaults to 10.
        multiprocess (bool, optional): Apply automatic evaluation methods with multiprocessing. Defaults to True.
        color (bool, optional): Color the resulting explanatory graph. Defaults to False.
        verbosity_level (int, optional): Print verbosity level (0+). Defaults to 1.
        seed (int, optional): Seed for reproduciblity. Defaults to SEED.

    Returns:
        Result: Summary of explanatory graph and automatic evaluation results.
    """
    from time import process_time

    df = deepcopy(df)
    verbose = verbosity_level > 0

    label = [col for col in df.columns if col.startswith('PRED_')][0]
    features = [col for col in df.columns if not col.startswith('PRED_')]

    print('*' * 30)
    print(f'* {label: <26} *')
    print('*' * 30)

    if continuous:
        # convert continuous columns to ordinal in continuous_cuts equal splits
        for col in continuous:
            if verbose:
                print(f'(i) Replaced continuous values of "{col}" with {continuous_cuts} equal-sized intervals')
            df[col] = pd.qcut(df[col], continuous_cuts).cat.codes

    if categorical_max:
        # maximum number of values for each categorical value (excluding text and PRED_*):
        for col in df.select_dtypes(include=['category', 'object']).columns:
            if col != 'text' and not col.startswith('PRED') and len(df.groupby(col)) > categorical_max:
                if verbose:
                    print(f'(i) Reduced categories of "{col}" with {categorical_max} and grouped remainder in Other')
                df = df.replace(df.groupby(col).count().iloc[:, 0].sort_values(ascending=False).index[categorical_max:], 'Other')

    # 1. Compare g'(Z) = \hat{Y}' to \hat{Y}
    evaluation_Z = evaluate_Z(
        df[features],
        df[label].values.ravel(),
        verbose=verbose,
    )

    # 2. Create global PAG on V=(Z, \hat{Y}) and perform sanity checks
    _time = process_time()
    G, edges, x_map, test_results, sanity_check, svg = create_pag(
        df,
        alpha=alpha,
        depth=depth,
        verbose=verbose,
        verbosity_level=verbosity_level,
        color=color,
        view_svg=False,
    )
    elapsed_time = process_time() - _time

    print(title('ELAPSED TIME'))
    print(f'{elapsed_time}s')

    # 3. Determine causal determinants Z_i with Z_i --> \hat{Y} and Z o-> \hat{Y}
    if n_trials > 0:
        evaluation_determinants = causal_determinants(
            df,
            n_trials=n_trials,
            depth=depth,
            multiprocess=multiprocess,
            seed=seed
        )
    else:
        evaluation_determinants = {}

    # 4. Evaluate PAG with InterVal and MVEE
    if times > 0:
        evaluation_PAG = evaluate_pag(
            G,
            df,
            alpha=alpha,
            depth=depth,
            times=times,
            percentage=percentage,
            multiprocess=multiprocess,
            verbose=verbose,
        )
    else:
        evaluation_PAG = [{'strategy': strategy, 'phd': None, 'adj_precision': None, 'adj_recall': None}
                          for strategy in ['InterVal', 'MVEE']]

    # Show resulting SVG
    def svg_to_html(svg, width='100%'):
        import base64
        import re

        b64 = base64.b64encode(re.sub(b'transform=\"(scale\(.*?\))\s+', b'transform="', svg)).decode("utf=8")
        return f'<img width="{width}" src="data:image/svg+xml;base64,{b64}" >'

    print(title('RESULTING SVG'))
    display(HTML(svg_to_html(svg)))

    return Result(
        label=label,
        features=features,
        graph=G,
        edges=edges,
        sanity_check=sanity_check,
        pag_test_results=test_results,
        z_fidelity=evaluation_Z,
        phd=evaluation_PAG,
        causal_determinants=evaluation_determinants,
        elapsed_time=elapsed_time,
        svg=svg,
    )


__all__ = [
    'generate_and_evaluate',
    'get_data',
    'get_pipeline',
    'huggingface_login',
    'train_and_apply'
]


if __name__ == '__main__':
    from gca.config import FORCE_TRAIN

    huggingface_login()

    dataset, _ = train_and_apply(force_train=FORCE_TRAIN)

    generate_and_evaluate(dataset, verbosity_level=1)
