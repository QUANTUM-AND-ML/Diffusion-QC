from __future__ import annotations

import logging
from typing import Iterable, Sequence, Tuple

import networkx as nx

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit import ParameterVector
except Exception as exc:  # pragma: no cover - 便于在无 qiskit 环境提示
    raise RuntimeError(
        "未检测到 Qiskit，请先安装 `pip install qiskit` 后再运行 QArchSearch"
    ) from exc

logger = logging.getLogger(__name__)


def _normalise_graph(graph: nx.Graph) -> Tuple[nx.Graph, dict[int, int]]:
    mapping = {node: idx for idx, node in enumerate(sorted(graph.nodes()))}
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)
    return relabeled, mapping


def _apply_single_qubit_gate(
    circuit: QuantumCircuit,
    gate: str,
    param: Parameter | None,
    qubit: int,
) -> None:
    gate = gate.lower()
    if gate == "rx":
        circuit.rx(2 * param if param is not None else 0.0, qubit)
    elif gate == "ry":
        circuit.ry(2 * param if param is not None else 0.0, qubit)
    elif gate == "rz":
        circuit.rz(2 * param if param is not None else 0.0, qubit)
    elif gate == "h":
        circuit.h(qubit)
    elif gate == "x":
        circuit.x(qubit)
    elif gate == "y":
        circuit.y(qubit)
    elif gate == "z":
        circuit.z(qubit)
    elif gate == "p":
        circuit.p(param if param is not None else 0.0, qubit)
    else:
        raise ValueError(f"不支持的单比特门：{gate}")


def build_mixer_layer(
    graph: nx.Graph,
    gate_sequence: Sequence[str],
    beta_param: Parameter,
) -> QuantumCircuit:
    """根据 gate 序列构建单层 mixer。"""

    num_qubits = graph.number_of_nodes()
    mixer = QuantumCircuit(num_qubits, name="mixer")
    for gate in gate_sequence:
        g = gate.lower()
        if g in {"rx", "ry", "rz", "h", "x", "y", "z", "p"}:
            for q in range(num_qubits):
                _apply_single_qubit_gate(mixer, g, beta_param, q)
        elif g in {"cx", "cz"}:
            for u, v in graph.edges():
                if g == "cx":
                    mixer.cx(u, v)
                else:
                    mixer.cz(u, v)
        else:
            raise ValueError(f"未实现 gate：{g}")
    return mixer


def _cost_layer(circuit: QuantumCircuit, graph: nx.Graph, gamma_param: Parameter) -> None:
    for u, v in graph.edges():
        circuit.cx(u, v)
        circuit.rz(2 * gamma_param, v)
        circuit.cx(u, v)


def build_qaoa_circuit(
    graph: nx.Graph,
    gate_sequence: Sequence[str],
    p: int,
) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """构建参数化 QAOA 电路。"""

    if p < 1:
        raise ValueError("p 必须 ≥ 1")
    if graph.number_of_nodes() == 0:
        raise ValueError("图不能为空")

    relabeled_graph, _ = _normalise_graph(graph)
    num_qubits = relabeled_graph.number_of_nodes()

    circuit = QuantumCircuit(num_qubits, name=f"qaoa_p{p}")
    for qubit in range(num_qubits):
        circuit.h(qubit)

    gamma_params = ParameterVector("gamma", p)
    beta_params = ParameterVector("beta", p)

    for layer in range(p):
        _cost_layer(circuit, relabeled_graph, gamma_params[layer])
        mixer = build_mixer_layer(relabeled_graph, gate_sequence, beta_params[layer])
        circuit.compose(mixer, inplace=True)

    return circuit, beta_params, gamma_params

