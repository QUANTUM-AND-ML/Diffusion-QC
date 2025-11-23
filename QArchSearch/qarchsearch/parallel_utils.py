from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Iterable, Sequence, Tuple, TypeVar

T = TypeVar("T")


def run_parallel(
    func: Callable[..., T],
    tasks: Sequence[Tuple],
    processes: int | None = None,
) -> Iterable[T]:
    """使用 multiprocessing 运行任务。

    当 `processes <= 1` 或任务数为 0 时退化为串行执行，便于在本地环境调试。
    """

    if not tasks:
        return []

    if processes is None:
        processes = min(len(tasks), max(1, mp.cpu_count() - 1))

    if processes <= 1:
        return [func(*task) for task in tasks]

    with mp.Pool(processes=processes) as pool:
        return pool.starmap(func, tasks)

