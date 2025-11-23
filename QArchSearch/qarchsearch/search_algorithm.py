from __future__ import annotations

import csv
import json
import logging
from typing import Iterable, List, Sequence

import networkx as nx

from qiskit.qasm2 import dumps as qasm2_dumps

from .evaluator import Evaluator, SearchResult
from .foundation import SearchConfig
from .parallel_utils import run_parallel
from .predictor import Predictor
from .qbuilder import build_qaoa_circuit

logger = logging.getLogger(__name__)


class QArchSearch:
    """QArchSearch 主流程实现。"""

    def __init__(
        self,
        config: SearchConfig,
        predictor: Predictor | None = None,
        evaluator: Evaluator | None = None,
    ):
        self.config = config
        self.predictor = predictor or Predictor(
            config.gate_alphabet, config.random_seed or 42
        )
        self.evaluator = evaluator or Evaluator(
            maxiter=config.optimizer_maxiter, tol=config.optimizer_tol
        )
        self.results_dir = config.results_dir

    def _evaluate_candidate(
        self,
        graph: nx.Graph,
        gate_sequence: Sequence[str],
        p: int,
    ) -> SearchResult:
        circuit, beta_params, gamma_params = build_qaoa_circuit(
            graph, gate_sequence, p
        )
        return self.evaluator.evaluate(
            graph=graph,
            circuit=circuit,
            beta_params=beta_params,
            gamma_params=gamma_params,
            gate_sequence=gate_sequence,
            p=p,
        )

    def _select_best(self, current_best: SearchResult | None, candidates: Iterable[SearchResult]) -> SearchResult | None:
        best = current_best
        for result in candidates:
            if best is None or result.energy > best.energy:
                best = result
        return best

    def search(self, graph: nx.Graph) -> SearchResult:
        combos = list(self.predictor.generate(self.config.k_max))
        if not combos:
            raise RuntimeError("Predictor 未生成任何 gate 组合")

        best_result: SearchResult | None = None
        log_rows: List[List[str]] = [["p", "gate_sequence", "energy", "approx_ratio"]]

        for p in range(1, self.config.p_max + 1):
            logger.info("搜索深度 p=%s，共 %s 个候选", p, len(combos))
            tasks = [(graph, combo, p) for combo in combos]
            if self.config.parallel_processes and self.config.parallel_processes > 1:
                logger.info("启用 %s 进程并行评估", self.config.parallel_processes)
                candidates = run_parallel(
                    self._evaluate_candidate, tasks, processes=self.config.parallel_processes
                )
            else:
                candidates = [self._evaluate_candidate(*task) for task in tasks]

            for cand in candidates:
                log_rows.append(
                    [
                        str(p),
                        " ".join(cand.gate_sequence),
                        f"{cand.energy:.4f}",
                        f"{cand.approximation_ratio:.4f}",
                    ]
                )

            best_result = self._select_best(best_result, candidates)
            logger.info(
                "当前最佳：p=%s, gates=%s, energy=%.4f, ratio=%.4f",
                best_result.p if best_result else "-",
                best_result.gate_sequence if best_result else "-",
                best_result.energy if best_result else float("nan"),
                best_result.approximation_ratio if best_result else float("nan"),
            )

        if best_result is None:
            raise RuntimeError("未获得有效的搜索结果")

        self._persist_results(best_result, log_rows)
        return best_result

    def _persist_results(self, best: SearchResult, log_rows: List[List[str]]) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        bound = best.circuit.assign_parameters(best.param_assignment)
        qasm_text = qasm2_dumps(bound)
        (self.results_dir / "best_circuit.qasm").write_text(
            qasm_text, encoding="utf-8"
        )
        metrics_path = self.results_dir / "metrics.json"
        metrics = {
            "energy": best.energy,
            "approximation_ratio": best.approximation_ratio,
            "p": best.p,
            "gate_sequence": list(best.gate_sequence),
            "parameters": {str(k): v for k, v in best.param_assignment.items()},
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        log_path = self.results_dir / "search_log.csv"
        with log_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerows(log_rows)

