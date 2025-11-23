from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import networkx as nx
import numpy as np

try:
    from qiskit.quantum_info import Statevector
except Exception as exc:  # pragma: no cover
    raise RuntimeError("需要安装 qiskit 以运行评估模块") from exc

try:
    from scipy.optimize import minimize
except Exception as exc:  # pragma: no cover
    raise RuntimeError("需要安装 scipy 以运行优化器") from exc


@dataclass
class SearchResult:
    gate_sequence: Tuple[str, ...]
    p: int
    energy: float
    approximation_ratio: float
    circuit: Any  # QuantumCircuit
    param_assignment: Dict[Any, float]


def _normalise_graph(graph: nx.Graph) -> nx.Graph:
    mapping = {node: idx for idx, node in enumerate(sorted(graph.nodes()))}
    return nx.relabel_nodes(graph, mapping, copy=True)


def _bitstring_cost(graph: nx.Graph, bitstring: str) -> int:
    """计算给定位串对应的 cut 大小。"""

    n = graph.number_of_nodes()
    bits = [int(b) for b in bitstring]
    total = 0
    for u, v in graph.edges():
        # qiskit 位序为 q_{n-1} ... q_0
        if bits[n - 1 - u] != bits[n - 1 - v]:
            total += 1
    return total


def _quantum_expectation(graph: nx.Graph, state: Statevector) -> float:
    probs = state.probabilities_dict()
    exp_value = 0.0
    for bitstring, prob in probs.items():
        exp_value += prob * _bitstring_cost(graph, bitstring)
    return exp_value


def _classical_maxcut(graph: nx.Graph) -> int:
    n = graph.number_of_nodes()
    best = 0
    edges = list(graph.edges())
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best


class Evaluator:
    """负责训练与评估 QAOA 电路。"""

    def __init__(self, maxiter: int = 200, tol: float = 1e-3):
        self.maxiter = maxiter
        self.tol = tol

    def _optimise_parameters(
        self,
        circuit,
        graph: nx.Graph,
        beta_params,
        gamma_params,
    ) -> Tuple[Dict[Any, float], float]:
        ordered_params = [
            param
            for param in list(gamma_params) + list(beta_params)
            if param in circuit.parameters
        ]
        num_params = len(ordered_params)
        if num_params == 0:
            raise ValueError("电路不包含可训练参数，无法执行优化")

        def energy_fn(theta: np.ndarray) -> float:
            assignment = dict(zip(ordered_params, theta))
            bound = circuit.assign_parameters(assignment)
            state = Statevector.from_label("0" * circuit.num_qubits)
            state = state.evolve(bound)
            return -_quantum_expectation(graph, state)

        res = minimize(
            energy_fn,
            x0=np.zeros(num_params),
            method="COBYLA",
            options={"maxiter": self.maxiter},
        )
        assignment = dict(zip(ordered_params, res.x))
        return assignment, -res.fun

    def evaluate(
        self,
        graph: nx.Graph,
        circuit,
        beta_params,
        gamma_params,
        gate_sequence: Sequence[str],
        p: int,
    ) -> SearchResult:
        norm_graph = _normalise_graph(graph)
        assignment, energy = self._optimise_parameters(
            circuit, norm_graph, list(beta_params), list(gamma_params)
        )
        classical_opt = _classical_maxcut(norm_graph)
        approx_ratio = energy / classical_opt if classical_opt > 0 else 0.0
        return SearchResult(
            gate_sequence=tuple(gate_sequence),
            p=p,
            energy=energy,
            approximation_ratio=approx_ratio,
            circuit=circuit,
            param_assignment=assignment,
        )

