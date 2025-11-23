"""
QArchSearch 包初始化。

该包提供基于论文《QArchSearch : A Scalable Quantum Architecture Search Package》
的模块化实现，包括：

- `foundation`: 环境配置与日志初始化
- `predictor`: 候选 mixer 结构生成
- `qbuilder`: QAOA 电路构建
- `evaluator`: 量子电路评估与优化
- `search_algorithm`: 主搜索逻辑
- `parallel_utils`: 并行工具

所有模块都以纯 Python 形式实现，默认依赖 Qiskit 与 NetworkX。
"""

from .foundation import SearchConfig, initialize_environment
from .predictor import Predictor
from .qbuilder import build_mixer_layer, build_qaoa_circuit
from .evaluator import Evaluator, SearchResult
from .search_algorithm import QArchSearch

__all__ = [
    "SearchConfig",
    "initialize_environment",
    "Predictor",
    "build_mixer_layer",
    "build_qaoa_circuit",
    "Evaluator",
    "SearchResult",
    "QArchSearch",
]

