from typing import Any, Dict, List

import pandas as pd

from gca.utils import get_graph_for_print


class Result:
    def __init__(self,
                 label: str,
                 features: List[str],
                 graph,
                 edges,
                 sanity_check: bool,
                 pag_test_results: List[Dict[str, Any]],
                 z_fidelity: Dict[str, Any],
                 phd: List[Dict[str, Any]],
                 causal_determinants: Dict[str, float],
                 elapsed_time: float,
                 svg: bytes):
        self.label = label
        self.features = features
        self._xmap = {f'X{i}': f for i, f in enumerate(self.features)}
        self.graph = graph
        self.edges = edges
        self.sanity_check = sanity_check
        self.elapsed_time = elapsed_time
        self.svg = svg
        self._pag_test_results = pag_test_results
        self.z_fidelity = z_fidelity
        self._phd = phd
        self._causal_determinants = causal_determinants

    @property
    def n_features(self):
        return len(self.features)

    @property
    def pag_test_results(self):
        return pd.DataFrame(self._pag_test_results)

    @property
    def z_accuracy(self):
        return self.z_fidelity['accuracy_score']

    @property
    def z_precision(self):
        return self.z_fidelity['precision_score']

    @property
    def z_recall(self):
        return self.z_fidelity['recall_score']

    @property
    def z_f1(self):
        return self.z_fidelity['f1_score']

    def __get_strategy_element(self, name: str, element: str):
        return [strategy[element] for strategy in self._phd if str.lower(strategy['strategy']) == str.lower(name)][0]

    def __relative_adjacency_score(self, absolute_score: int):
        return 1.0 - absolute_score / ((self.n_features + 1) * self.n_features / 2)

    @property
    def mvee(self):
        return self.__get_strategy_element('mvee', 'phd')

    @property
    def mvee_adj(self):
        return self.__get_strategy_element('mvee', 'adj_f1')

    @property
    def mvee_relative(self):
        return self.__relative_adjacency_score(self.mvee)

    @property
    def interval(self):
        return self.__get_strategy_element('interval', 'phd')

    @property
    def interval_adj(self):
        return self.__get_strategy_element('interval', 'adj_f1')

    @property
    def interval_relative(self):
        return self.__relative_adjacency_score(self.interval)

    @property
    def structural_quality(self):
        return self.mvee

    @property
    def causal_determinants(self):
        return pd.Series(self._causal_determinants)

    def save_svg(self, path: str):
        if not path.endswith('.svg') or not path.endswith('.html'):
            path += '.svg'
        with open(path, 'wb') as f:
            f.write(self.svg)

    def __repr__(self):
        return f'{self.__class__.__name__}(\n"' + str(get_graph_for_print(self.graph, self._xmap)) + '",' + \
            f'\n{self.sanity_check=},\n{self.z_accuracy=},\n{self.z_precision=},\n{self.z_recall=},' + \
            f'\n{self.z_f1=},\n{self.mvee=},\n{self.interval=},\n{self.elapsed_time=})'
