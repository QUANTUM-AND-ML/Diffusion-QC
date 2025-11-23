from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


GateSequence = Tuple[str, ...]


@dataclass
class Predictor:
    """最简版 QArchSearch Predictor。

    当前实现采用随机 / 穷举方式生成 gate 组合，作为论文中 Random Search
    的可复现基线。后续可在此替换为 DNN 预测模型。
    """

    gate_alphabet: Sequence[str]
    random_seed: int | None = 42

    def __post_init__(self) -> None:
        if self.random_seed is not None:
            random.seed(self.random_seed)

    def enumerate_combinations(self, k_max: int) -> List[GateSequence]:
        """枚举 1..k_max 所有 gate 组合（允许重复）。"""

        combos: List[GateSequence] = []
        for k in range(1, k_max + 1):
            combos.extend(itertools.product(self.gate_alphabet, repeat=k))
        return combos

    def sample_combinations(self, k_max: int, sample_size: int) -> List[GateSequence]:
        """随机采样 gate 组合，以控制搜索空间。"""

        universe = self.enumerate_combinations(k_max)
        if len(universe) <= sample_size:
            return universe
        return random.sample(universe, sample_size)

    def generate(self, k_max: int, limit: int | None = None) -> Iterable[GateSequence]:
        """供搜索算法使用的统一接口。"""

        combos = self.enumerate_combinations(k_max)
        if limit:
            combos = combos[:limit]
        return combos

