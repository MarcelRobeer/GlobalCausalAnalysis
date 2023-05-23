import numpy as np
import pandas as pd

from functools import partial
from multiprocessing import Pool
from tqdm.auto import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance 
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline

from causallearn.graph.Endpoint import Endpoint

from gca.agreement_graph import AgreementGraph
from gca.config import SEED
from gca.pag import create_pag
from gca.utils import create_prettytable, silence_inner_tqdm, subtitle, title


def evaluate_Z(Z,
               y,
               clf_method=RandomForestClassifier,
               metrics=[accuracy_score, precision_score, recall_score, f1_score],
               n_components=None,
               gamma=None,
               verbose=True,
               seed=SEED):
    # Consistent Fold splitting
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(cv.split(X=Z, y=y))

    # One-hot encode
    if isinstance(Z, pd.DataFrame):
        Z = pd.get_dummies(Z)

    # Instantiate classifier
    if n_components is None:
        clf = clf_method(random_state=seed)
    else:
        if gamma is None:
            gamma = .2 #1. / (Z.shape[1] * Z.var(axis=-1)) 
        clf = make_pipeline(
            Nystroem(n_components=n_components, gamma=gamma, random_state=seed),
            clf_method(random_state=seed)
        )

    # Cross val predictions
    y_hat = cross_val_predict(clf, Z, y, cv=splits)

    # Scores
    average = 'macro' if len(np.unique(y)) > 2 else 'binary'
    metrics = {'method': clf_method.__name__, 'weighting': average} | \
              {metric.__name__: metric(y, y_hat, average=average) if 'accuracy' not in metric.__name__ else metric(y, y_hat)
               for metric in metrics}
    clf.fit(Z, y)
    importances = {'feature': list(Z.columns)} | {k: list(v.round(decimals=5)) for k, v in permutation_importance(clf, Z, y, n_repeats=5, random_state=seed).items()
                   if k in ['importances_mean', 'importances_std']}

    if verbose:
        print(title('Z-FIDELITY: EVALUATION OF Z ON Y_HAT'))
        print(create_prettytable({k: round(v, 5) if isinstance(v, float) else v
                                  for k, v in metrics.items()}))
        print(subtitle('PERMUTATION IMPORTANCE SCORES'))
        print(create_prettytable([dict(zip(importances, zv)) for zv in zip(*importances.values())]).get_string(sortby='importances_mean', reversesort=True))
    return metrics


def get_determinants(sampled_df: pd.DataFrame, depth: int = -1):
    # evaluate PAG with bootstrapping (same size as df with replacement)
    G, _, x_map, _, _, _ = create_pag(sampled_df, depth=depth, verbose=False)  
    node = G.nodes[G.num_vars - 1]
    return [x_map[edge.node1.name] for edge in G.get_node_edges(node)
            if edge.node2 == node and edge.endpoint1 in (Endpoint.TAIL, Endpoint.CIRCLE) and edge.endpoint2 == Endpoint.ARROW]


def causal_determinants(df,
                        n_trials: int = 50,
                        depth: int = -1,
                        seed: int = SEED,
                        verbose: bool = True,
                        multiprocess: bool = True,
                        **kwargs):
    count = {var: 0 for var in df.columns}
    trials = [df.sample(len(df), replace=True, random_state=seed + i) for i in range(n_trials)]
    with silence_inner_tqdm():
        desc = 'Calculating causal determinants'
        if multiprocess:
            with Pool() as pool:
                for result in tqdm(pool.imap_unordered(partial(get_determinants, depth=depth, verbose=False, show_progress=False), trials),
                                total=n_trials, desc=desc):
                    for var in result:
                        count[var] += 1
        else:
            for trial in tqdm(trials, total=n_trials, desc=desc):
                for var in get_determinants(trial, depth=depth, verbose=False, show_progress=False):
                    count[var] += 1

    determinants = {k: v / n_trials for k, v in sorted(count.items(), key=lambda x: x[1], reverse=True)
                    if not k.startswith('PRED_')}

    if verbose:
        print(title(f'CAUSAL DETERMINANTS (Zi -> Y_HAT or Zi o-> Y_HAT; over {n_trials} trials)'))
        print(create_prettytable(determinants))

    return determinants


def evaluate_pag(pag,
                 df: pd.DataFrame,
                 times: int = 5,
                 percentage: float = 0.8,
                 alpha: float = 0.05,
                 depth: int = -1,
                 verbose: bool = True,
                 multiprocess: bool = True,
                 seed: int = SEED):
    samples = [df.sample(int(percentage * len(df)), replace=False, random_state=seed + i) for i in range(times)]

    with silence_inner_tqdm():
        desc = 'Calculating InterVal/MVEE'
        if multiprocess:
            with Pool() as pool:
                subgraphs = []
                for result in tqdm(pool.imap_unordered(partial(create_pag, alpha=alpha, depth=depth, verbose=False, show_progress=False), samples),
                                total=times, desc=desc):
                    subgraphs.append(result[0])
        else:
            subgraphs = [create_pag(sample, alpha=alpha, depth=depth, verbose=False, show_progress=False)[0]
                         for sample in tqdm(samples, total=times, desc=desc)]

    agreement_graph_interval = AgreementGraph(subgraphs, default_strategy='interval')
    agreement_graph_mvee = agreement_graph_interval.apply_strategy(strategy='mvee')

    evaluations = [agreement_graph_interval.full_evaluation(pag),
                   agreement_graph_mvee.full_evaluation(pag)]

    if verbose:
        print(title(f'PAG STRUCTURAL QUALITY & STABILITY EVALUATION (InterVal/MVEE) over {times} neighbors'))
        print(create_prettytable(evaluations))

    return evaluations
