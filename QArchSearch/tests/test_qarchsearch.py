import networkx as nx
import pytest

qiskit = pytest.importorskip("qiskit")  # noqa: F401

from qarchsearch.foundation import SearchConfig
from qarchsearch.search_algorithm import QArchSearch


def small_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return graph


def test_search_runs(tmp_path):
    config = SearchConfig(
        gate_alphabet=["rx", "h"],
        p_max=1,
        k_max=1,
        optimizer_maxiter=5,
        results_dir=tmp_path / "results",
    )
    search = QArchSearch(config=config)
    result = search.search(small_graph())
    assert result.energy >= 0
    assert 0 <= result.approximation_ratio <= 1

