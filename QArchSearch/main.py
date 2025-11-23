from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import networkx as nx

from qarchsearch.foundation import initialize_environment
from qarchsearch.search_algorithm import QArchSearch

logger = logging.getLogger(__name__)


def _load_graph(graph_path: Path | None) -> nx.Graph:
    if graph_path is None:
        logger.warning("未提供图文件，使用 4 节点环形图示例")
        return nx.cycle_graph(4)
    if not graph_path.exists():
        raise FileNotFoundError(f"找不到图文件：{graph_path}")
    if graph_path.suffix in {".edgelist", ".txt"}:
        return nx.read_edgelist(graph_path, nodetype=int)
    if graph_path.suffix == ".json":
        data = json.loads(graph_path.read_text(encoding="utf-8"))
        edges = data.get("edges", data)
        graph = nx.Graph()
        graph.add_edges_from(edges)
        return graph
    raise ValueError("目前仅支持 .edgelist/.txt/.json 图格式")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QArchSearch 命令行入口")
    parser.add_argument("--graph", type=Path, help="图文件路径（edge list / json）")
    parser.add_argument("--p-max", type=int, default=2, help="最大 QAOA 深度")
    parser.add_argument("--k-max", type=int, default=2, help="最大 gate 序列长度")
    parser.add_argument(
        "--gates",
        type=str,
        default="rx,ry,h,cx",
        help="逗号分隔 gate alphabet",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="并行进程数（<=1 表示串行）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="可选 json/yaml 配置文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = initialize_environment(
        config_path=str(args.config) if args.config else None,
        overrides={
            "p_max": args.p_max,
            "k_max": args.k_max,
            "gate_alphabet": [g.strip() for g in args.gates.split(",") if g.strip()],
            "parallel_processes": args.parallel,
        },
    )
    graph = _load_graph(args.graph)

    search = QArchSearch(config=config)
    result = search.search(graph)
    print(
        f"最佳 mixer: gates={result.gate_sequence}, p={result.p}, "
        f"energy={result.energy:.4f}, approx_ratio={result.approximation_ratio:.4f}"
    )


if __name__ == "__main__":
    main()

