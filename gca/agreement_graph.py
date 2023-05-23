from typing import Dict, List, Union
from copy import deepcopy

import itertools
import numpy as np

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Graph import Graph
from causallearn.graph.GraphNode import GraphNode


def adjacency_skeleton(adjacency_matrix: np.ndarray) -> np.ndarray:
    return (np.abs(adjacency_matrix) == 0).astype(int) - 1


def to_skeleton(graph: GeneralGraph) -> GeneralGraph:
    skeleton = deepcopy(graph)
    skeleton.graph = adjacency_skeleton(skeleton.graph)
    return skeleton


# Special edge type for partial agreement: ?-?
class AgreementGraph(Graph):
    def __init__(self,
                 graphs: List[GeneralGraph],
                 default_strategy: str = 'interval',
                 strict: bool = True):
        self.graphs = dict(enumerate(graphs))
        self.nodes = sorted(set(node for nodes in [g.get_nodes() for g in self.graphs.values()] for node in nodes))
        self.node_map = {v: k for k, v in enumerate(self.nodes)}
        self.node_index = {node.name: node for node in self.nodes}
        self.num_vars = len(self.nodes)
        self.default_strategy = default_strategy

        if strict and any(set(self.nodes) != set(g.get_nodes()) for g in self.graphs.values()):
            raise NotImplementedError('Both strategies assume equal node names for all graphs')
        else:
            for graph in self.graphs.values():
                for node in self.nodes:
                    if node not in graph.nodes:
                        graph.add_node(node)

        self.skeletons = {i: to_skeleton(g) for i, g in self.graphs.items()}
        self._last_strategy = self.default_strategy
        self._graph = self.agreement_graph(strategy=self._last_strategy)

    def get_nodes(self):
        return self.nodes

    def get_num_nodes(self) -> int:
        return self.num_vars

    @property
    def default_strategy(self):
        return self._default_strategy

    @default_strategy.setter
    def default_strategy(self, default_strategy):
        self._default_strategy = str.lower(default_strategy)

    def get_edge(self, node1: GraphNode, node2: GraphNode) -> Union[Edge, None]:
        i = self.node_map[node1]
        j = self.node_map[node2]

        end_1 = self.graph[i, j]
        end_2 = self.graph[j, i]

        if end_1 == 0:
            return None

        return Edge(node1, node2, Endpoint(end_1), Endpoint(end_2))

    def get_endpoint(self, node1: GraphNode, node2: GraphNode) -> Union[Endpoint, None]:
        i = self.node_map[node1]
        j = self.node_map[node2]

        if self.graph[i, j] == 0:
            return None
        return Endpoint(self.graph[j, i])

    def get_node(self, node: str) -> GraphNode:
        return self.node_index[node]  

    def agreement_graph(self, strategy=None):
        if strategy is None:
            strategy = self.default_strategy
        if strategy not in ['interval', 'mvee']:
            raise ValueError(f'Unknown strategy "{strategy}", choose from [interval, mvee]')
        self.default_strategy = strategy

        graph: np.ndarray = np.zeros((self.num_vars, self.num_vars), np.dtype(int)) + 5

        # Intersection-Validation (InterVal)
        # https://proceedings.mlr.press/v84/viinikka18a.html
        if strategy == 'interval':
            if len(self.graphs) == 1:
                return self.graphs[0].graph
            graphs = [g.graph for g in self.graphs.values()]

            # Get the ends all graphs agree on
            ends_agreed = np.logical_and.reduce(
                [np.equal(graphs[0], graphs[i]) for i in range(1, len(graphs))]
                )
            np.putmask(graph, ends_agreed, graphs[0])
            return graph

        # Modal Value of Edges Existence (MVEE)
        # https://proceedings.mlr.press/v138/handhayani20a.html
        elif strategy == 'mvee':
            if len(self.graphs) == 1:
                return self.skeletons[0].graph
            skeletons = np.dstack([s.graph for s in self.skeletons.values()])
            for i in range(graph.shape[0]):
                for j in range(graph.shape[1]):
                    majority_vote = skeletons[i, j, :].mean()

                    # If most vote for -1 (A --- B) choose -1
                    if majority_vote < -0.5:
                        graph[i, j] = -1
                    # If most vote for 0 (A -/- B) choose 0
                    elif majority_vote > -0.5:
                        graph[i, j] = 0

                    # Otherwise keep default value
            return graph

    def apply_strategy(self, strategy: str) -> 'AgreementGraph':
        new_graph = deepcopy(self)
        new_graph.agreement_graph(strategy=strategy)
        return new_graph

    @property
    def graph(self):
        if self._last_strategy != self.default_strategy or self._graph is None:
            self._graph = self.agreement_graph(strategy=self.default_strategy)
            self._last_strategy = self.default_strategy
        return self._graph

    def is_adjacent_to(self,
                       node1: GraphNode,
                       node2: GraphNode,
                       strict: bool = True) -> bool:
        i, j = self.node_map[node1], self.node_map[node2]
        graph = self.graph 
        if not strict:
            graph[graph == 5] = -1  # special
        return graph[j, i] != 0

    def phd(self, graph: GeneralGraph) -> int:
        """Partial Hamming Distance (PHD) to agreement graph."""
        if self.default_strategy == 'mvee':
            graph = to_skeleton(graph)

        assert len(graph.graph) == len(self.graph)
        return sum((graph.graph[i, j], graph.graph[j, i]) != (self.graph[i, j], self.graph[j, i])
                   for i, j in itertools.combinations(range(len(self.graph)), 2))

    def adjacency_confusion(self, graph: GeneralGraph) -> Dict[str, int]:
        """Confusion matrix based on adjacency/non-adjacency."""
        def to_pos_neg(G):
            skeleton = adjacency_skeleton(G.graph)
            return [skeleton[i, j] == skeleton[j, i] == -1
                    for i, j in itertools.combinations(range(len(skeleton)), 2)]
        pn_graph = to_pos_neg(graph)
        pn_agreement_graph = to_pos_neg(self)
        return {'tp': sum(x == y == 1 for x, y in zip(pn_graph, pn_agreement_graph)),
                'fp': sum(x == 1 and y == 0 for x, y in zip(pn_graph, pn_agreement_graph)),
                'tn': sum(x == y == 0 for x, y in zip(pn_graph, pn_agreement_graph)),
                'fn': sum(x == 0 and y == 1 for x, y in zip(pn_graph, pn_agreement_graph))}

    def full_evaluation(self, graph: GeneralGraph):
        adj = self.adjacency_confusion(graph)

        try:
            precision = adj['tp'] / (adj['tp'] + adj['fp'])
        except ZeroDivisionError:
            precision = 0.0

        try:
            recall = adj['tp'] / (adj['tp'] + adj['fn'])
        except ZeroDivisionError:
            recall = 1.0

        return {'strategy': {'interval': 'InterVal', 'mvee': 'MVEE'}[self.default_strategy],
                'phd': self.phd(graph),
                'adj_precision': precision,
                'adj_recall': recall,
                'adj_f1': (2 * precision * recall) / (precision + recall)}
