from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml 为可选依赖
    yaml = None


DEFAULT_GATE_ALPHABET = ["rx", "ry", "rz", "h", "cx", "cz"]


@dataclass
class SearchConfig:
    """QArchSearch 全局配置。"""

    gate_alphabet: Sequence[str] = field(
        default_factory=lambda: DEFAULT_GATE_ALPHABET.copy()
    )
    p_max: int = 2
    k_max: int = 2
    optimizer_maxiter: int = 200
    optimizer_tol: float = 1e-3
    parallel_processes: int = 0
    use_shared_parameters: bool = True
    results_dir: Path = Path("results")
    random_seed: Optional[int] = 42

    def validate(self) -> None:
        if self.p_max < 1:
            raise ValueError("p_max 必须 ≥ 1")
        if self.k_max < 1:
            raise ValueError("k_max 必须 ≥ 1")
        if not self.gate_alphabet:
            raise ValueError("gate_alphabet 不能为空")
        for gate in self.gate_alphabet:
            if not isinstance(gate, str):
                raise TypeError("gate_alphabet 应仅包含字符串 gate 名称")
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)


def _load_from_file(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("缺少 PyYAML 依赖，无法解析 YAML 配置")
        return yaml.safe_load(text) or {}
    if config_path.suffix == ".json":
        return json.loads(text)
    raise ValueError("仅支持 json / yaml 配置文件")


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> SearchConfig:
    """加载 SearchConfig。

    Args:
        config_path: 可选配置文件路径
        overrides: 需要覆盖的字段
    """

    data: Dict[str, Any] = {}
    if config_path:
        data.update(_load_from_file(Path(config_path)))
    if overrides:
        data.update(overrides)
    config = SearchConfig(**data)
    config.validate()
    return config


def initialize_environment(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> SearchConfig:
    """初始化日志与配置。"""

    log_level = os.environ.get("QARCHSEARCH_LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )
    logging.getLogger("qarchsearch").info("初始化 QArchSearch 环境")
    return load_config(config_path, overrides)

