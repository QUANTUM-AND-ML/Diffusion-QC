# QArchSearch: 可扩展的量子架构搜索包

基于论文《QArchSearch : A Scalable Quantum Architecture Search Package》实现的量子架构搜索系统，用于在给定门字母表下自动搜索最优的 QAOA Mixer Circuit。本项目采用模块化设计，支持串行与并行搜索，可扩展到大规模量子电路模拟。

## 目录

- [项目概述](#项目概述)
- [环境要求](#环境要求)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [API 文档](#api-文档)
  - [配置模块 (foundation)](#配置模块-foundation)
  - [预测模块 (predictor)](#预测模块-predictor)
  - [电路构建模块 (qbuilder)](#电路构建模块-qbuilder)
  - [评估模块 (evaluator)](#评估模块-evaluator)
  - [搜索算法 (search_algorithm)](#搜索算法-search_algorithm)
  - [并行工具 (parallel_utils)](#并行工具-parallel_utils)
- [使用示例](#使用示例)
- [测试](#测试)
- [输出文件说明](#输出文件说明)
- [性能优化建议](#性能优化建议)

## 项目概述

QArchSearch 是一个用于量子架构搜索的 Python 包，主要解决以下问题：

1. **自动化搜索**：在给定门字母表（Gate Alphabet）下，自动搜索最优的 QAOA Mixer Circuit 结构
2. **性能评估**：通过 COBYLA 优化器训练电路参数，计算能量值与近似比（Approximation Ratio）
3. **可扩展性**：支持多进程并行搜索，可处理大规模图与深度电路

当前实现采用**随机搜索**（Random Search）作为基线算法，符合论文中的实验设置。后续可扩展为基于深度神经网络的智能搜索。

## 环境要求

- **Python**: 3.9 或更高版本
- **操作系统**: Windows / Linux / macOS
- **依赖包**: 见 `requirements.txt`

## 安装指南

### 1. 克隆或下载项目

```bash
cd /path/to/fuxian2
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python -c "import qarchsearch; print('QArchSearch 安装成功')"
```

## 快速开始

### 基本用法

使用示例图运行搜索：

```bash
python main.py --graph data/sample_graph.edgelist --p-max 2 --k-max 2 --gates rx,ry,h,cx
```

### 命令行参数

- `--graph`: 输入图文件路径（edgelist 格式）
- `--p-max`: QAOA 最大深度（默认：2）
- `--k-max`: 门组合最大长度（默认：2）
- `--gates`: 门字母表，逗号分隔（默认：rx,ry,rz,h,cx,cz）
- `--parallel`: 并行进程数，0 表示串行（默认：0）
- `--maxiter`: COBYLA 优化器最大迭代次数（默认：200）
- `--results-dir`: 结果输出目录（默认：results）

### 输出结果

运行完成后，在 `results/` 目录下生成：

- `best_circuit.qasm`: 最佳电路的 OpenQASM 2.0 格式
- `metrics.json`: 性能指标与优化参数
- `search_log.csv`: 所有候选电路的搜索记录

## 项目结构

```
fuxian2/
├── qarchsearch/              # 核心包
│   ├── __init__.py           # 包初始化与导出
│   ├── foundation.py         # 配置管理与环境初始化
│   ├── predictor.py          # 门组合生成器（随机搜索）
│   ├── qbuilder.py           # QAOA 电路构建
│   ├── evaluator.py          # 电路评估与参数优化
│   ├── search_algorithm.py   # 主搜索算法
│   └── parallel_utils.py     # 并行执行工具
├── tests/                    # 单元测试
│   └── test_qarchsearch.py
├── data/                     # 示例图数据
│   ├── sample_graph.edgelist
│   └── test_graph_10nodes.edgelist
├── results/                  # 搜索结果输出（自动创建）
├── main.py                   # CLI 入口程序
├── requirements.txt          # 依赖列表
└── README.md                 # 本文档
```

## API 文档

### 配置模块 (foundation)

#### `SearchConfig`

```python
class SearchConfig:
    """
    存储 QArchSearch 全局配置的数据类。

    详细描述：
        该配置类使用 dataclass 实现，包含搜索算法的所有可调参数。
        支持从 JSON/YAML 文件加载，或通过字典覆盖默认值。
        所有参数在初始化后通过 `validate()` 方法进行合法性检查。

    Attributes:
        gate_alphabet (Sequence[str]): 门字母表，例如 ['rx', 'ry', 'h', 'cx']。
        p_max (int): QAOA 最大深度，必须 ≥ 1，默认值为 2。
        k_max (int): 门组合最大长度，必须 ≥ 1，默认值为 2。
        optimizer_maxiter (int): COBYLA 优化器最大迭代次数，默认值为 200。
        optimizer_tol (float): 优化器收敛容差，默认值为 1e-3。
        parallel_processes (int): 并行进程数，0 表示串行执行，默认值为 0。
        use_shared_parameters (bool): 是否在 mixer 层共享参数，默认值为 True。
        results_dir (Path): 结果输出目录路径，默认值为 Path("results")。
        random_seed (Optional[int]): 随机数种子，用于可复现性，默认值为 42。

    Methods:
        validate: 验证配置参数的合法性。
    """
```

#### `initialize_environment(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> SearchConfig`

```
初始化日志系统并加载配置。

详细描述：
    设置 Python logging 模块的格式与级别，从环境变量 QARCHSEARCH_LOGLEVEL 读取日志级别（默认 INFO）。
    支持从 JSON/YAML 文件加载配置，并通过 overrides 字典覆盖特定字段。
    配置加载后自动调用 validate() 进行验证。

Args:
    config_path (Optional[str]): 配置文件路径，支持 .json 或 .yml/.yaml 格式。如果为 None，则使用默认配置。
    overrides (Optional[Dict[str, Any]]): 需要覆盖的配置字段字典，例如 {'p_max': 3, 'k_max': 4}。

Returns:
    SearchConfig: 初始化完成的配置对象。

Raises:
    FileNotFoundError: 当指定的配置文件不存在时抛出。
    RuntimeError: 当配置文件为 YAML 格式但未安装 PyYAML 时抛出。
    ValueError: 当配置文件格式不支持或配置参数不合法时抛出。

Example:
    >>> config = initialize_environment()
    >>> print(config.p_max)
    2
    >>> config = initialize_environment(overrides={'p_max': 3})
    >>> print(config.p_max)
    3
```

#### `load_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> SearchConfig`

```
从文件或字典加载 SearchConfig 对象。

详细描述：
    优先从配置文件加载，然后应用 overrides 中的覆盖值。
    支持 JSON 和 YAML 两种格式，YAML 需要安装 PyYAML 包。

Args:
    config_path (Optional[str]): 配置文件路径，如果为 None 则跳过文件加载。
    overrides (Optional[Dict[str, Any]]): 覆盖字段字典，键名需与 SearchConfig 字段名一致。

Returns:
    SearchConfig: 加载并验证后的配置对象。

Raises:
    FileNotFoundError: 配置文件不存在。
    RuntimeError: YAML 文件需要 PyYAML 但未安装。
    ValueError: 配置文件格式不支持或参数验证失败。
```

### 预测模块 (predictor)

#### `Predictor`

```python
class Predictor:
    """
    生成候选门组合序列的预测器。

    详细描述：
        当前实现采用随机搜索策略，通过枚举或采样方式生成所有可能的门组合。
        后续可扩展为基于深度神经网络的智能预测器，根据历史性能数据推荐候选结构。
        所有生成的组合以元组形式返回，便于哈希与去重。

    Attributes:
        gate_alphabet (Sequence[str]): 可用的门类型列表，例如 ['rx', 'ry', 'h', 'cx']。
        random_seed (int | None): 随机数种子，用于可复现的随机采样，默认值为 42。

    Methods:
        enumerate_combinations: 枚举所有可能的门组合。
        sample_combinations: 随机采样指定数量的门组合。
        generate: 统一的生成接口，供搜索算法调用。
    """
```

#### `enumerate_combinations(k_max: int) -> List[GateSequence]`

```
枚举从长度 1 到 k_max 的所有门组合（允许重复）。

详细描述：
        使用 itertools.product 生成笛卡尔积，对于每个长度 k (1 ≤ k ≤ k_max)，
        生成所有可能的 k 元组组合。时间复杂度为 O(|A|^k_max)，其中 |A| 为门字母表大小。

Args:
    k_max (int): 最大组合长度，必须 ≥ 1。

Returns:
    List[GateSequence]: 所有可能的门组合列表，每个元素为元组，例如 [('rx',), ('ry',), ('rx', 'ry'), ...]。

Raises:
    无显式异常，但 k_max 过大可能导致内存溢出。

Example:
    >>> pred = Predictor(['rx', 'ry'], random_seed=42)
    >>> combos = pred.enumerate_combinations(2)
    >>> len(combos)
    6  # 2^1 + 2^2 = 2 + 4 = 6
    >>> combos[0]
    ('rx',)
```

#### `sample_combinations(k_max: int, sample_size: int) -> List[GateSequence]`

```
随机采样指定数量的门组合，用于控制搜索空间大小。

详细描述：
        当枚举的组合总数超过 sample_size 时，使用 random.sample 进行无重复采样。
        如果总数小于等于 sample_size，则返回全部组合。采样结果受 random_seed 控制。

Args:
    k_max (int): 最大组合长度，必须 ≥ 1。
    sample_size (int): 期望采样的组合数量，必须 ≥ 1。

Returns:
    List[GateSequence]: 采样得到的门组合列表，长度不超过 sample_size。

Raises:
    无显式异常，但 sample_size 为 0 或负数时可能产生意外行为。

Example:
    >>> pred = Predictor(['rx', 'ry', 'h'], random_seed=42)
    >>> samples = pred.sample_combinations(3, 5)
    >>> len(samples) <= 5
    True
```

#### `generate(k_max: int, limit: int | None = None) -> Iterable[GateSequence]`

```
为搜索算法提供统一的生成接口。

详细描述：
        生成所有可能的门组合，如果指定 limit，则仅返回前 limit 个组合。
        该方法返回可迭代对象，支持延迟计算，适合处理大规模搜索空间。

Args:
    k_max (int): 最大组合长度，必须 ≥ 1。
    limit (int | None): 可选的数量限制，如果为 None 则返回全部组合。

Returns:
    Iterable[GateSequence]: 门组合的可迭代对象，每个元素为元组。

Raises:
    无显式异常。

Example:
    >>> pred = Predictor(['rx', 'ry'])
    >>> for combo in pred.generate(2, limit=3):
    ...     print(combo)
    ('rx',)
    ('ry',)
    ('rx', 'rx')
```

### 电路构建模块 (qbuilder)

#### `build_mixer_layer(graph: nx.Graph, gate_sequence: Sequence[str], beta_param: Parameter) -> QuantumCircuit`

```
根据门序列构建单层 QAOA Mixer 电路。

详细描述：
        遍历 gate_sequence 中的每个门，根据门类型应用相应的量子操作：
        - 单比特门（rx, ry, rz, h, x, y, z, p）：应用到所有量子比特
        - 双比特门（cx, cz）：应用到图的所有边上
        旋转门的参数为 2 * beta_param，符合 QAOA 标准形式。

Args:
    graph (nx.Graph): 输入图对象，节点编号需从 0 开始连续。
    gate_sequence (Sequence[str]): 门序列，例如 ['rx', 'ry', 'cx']。
    beta_param (Parameter): Qiskit Parameter 对象，用于参数化旋转门。

Returns:
    QuantumCircuit: 构建完成的 mixer 层电路，电路名称为 "mixer"。

Raises:
    ValueError: 当遇到不支持的门类型时抛出。

Example:
    >>> from qiskit.circuit import Parameter
    >>> import networkx as nx
    >>> G = nx.cycle_graph(3)
    >>> beta = Parameter('beta')
    >>> mixer = build_mixer_layer(G, ['rx', 'h'], beta)
    >>> print(mixer.num_qubits)
    3
```

#### `build_qaoa_circuit(graph: nx.Graph, gate_sequence: Sequence[str], p: int) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]`

```
构建完整的参数化 QAOA 电路。

详细描述：
        按照 QAOA 标准结构构建电路：
        1. 初始化：对所有量子比特应用 Hadamard 门，制备 |+⟩^⊗n 态
        2. 重复 p 层：每层包含 Cost Layer（基于图的边）和 Mixer Layer（基于 gate_sequence）
        Cost Layer 使用 gamma 参数，Mixer Layer 使用 beta 参数。
        图的节点会被重新编号为 0 到 n-1，确保与量子比特索引一致。

Args:
    graph (nx.Graph): 输入图对象，任意节点标签将被归一化为连续整数。
    gate_sequence (Sequence[str]): Mixer 层的门序列。
    p (int): QAOA 深度，必须 ≥ 1。

Returns:
    Tuple[QuantumCircuit, ParameterVector, ParameterVector]: 
        - 第一个元素：完整的 QAOA 电路
        - 第二个元素：beta 参数向量（长度为 p）
        - 第三个元素：gamma 参数向量（长度为 p）

Raises:
    ValueError: 当 p < 1 或图为空时抛出。

Example:
    >>> import networkx as nx
    >>> G = nx.cycle_graph(4)
    >>> circuit, beta, gamma = build_qaoa_circuit(G, ['rx', 'ry'], p=2)
    >>> print(circuit.num_qubits)
    4
    >>> print(len(beta), len(gamma))
    2 2
```

### 评估模块 (evaluator)

#### `Evaluator`

```python
class Evaluator:
    """
    负责训练与评估 QAOA 电路的性能。

    详细描述：
        使用 COBYLA 优化器训练电路参数，通过 Statevector 模拟计算期望能量。
        评估指标包括能量值（期望 cut 大小）和近似比（相对于经典最优解）。
        优化过程最小化负能量，因此最终能量为 -res.fun。

    Attributes:
        maxiter (int): COBYLA 优化器最大迭代次数，默认值为 200。
        tol (float): 优化器收敛容差，默认值为 1e-3。

    Methods:
        evaluate: 评估给定电路的性能。
        _optimise_parameters: 内部方法，执行参数优化。
    """
```

#### `evaluate(graph: nx.Graph, circuit: QuantumCircuit, beta_params: ParameterVector, gamma_params: ParameterVector, gate_sequence: Sequence[str], p: int) -> SearchResult`

```
评估给定 QAOA 电路的性能。

详细描述：
        执行完整的评估流程：
        1. 归一化图节点编号
        2. 使用 COBYLA 优化器训练参数（最多 maxiter 次迭代）
        3. 计算优化后的期望能量
        4. 通过穷举法计算经典最优解
        5. 计算近似比 = 能量 / 经典最优解
        返回包含所有评估信息的 SearchResult 对象。

Args:
    graph (nx.Graph): 输入图对象。
    circuit (QuantumCircuit): 待评估的参数化 QAOA 电路。
    beta_params (ParameterVector): Mixer 层的参数向量。
    gamma_params (ParameterVector): Cost 层的参数向量。
    gate_sequence (Sequence[str]): 门序列，用于记录。
    p (int): QAOA 深度，用于记录。

Returns:
    SearchResult: 包含以下字段的结果对象：
        - gate_sequence: 门序列元组
        - p: QAOA 深度
        - energy: 优化后的期望能量值（期望 cut 大小）
        - approximation_ratio: 近似比，范围 [0, 1]
        - circuit: 原始参数化电路
        - param_assignment: 优化后的参数赋值字典

Raises:
    ValueError: 当电路不包含可训练参数时抛出。

Example:
    >>> from qarchsearch import build_qaoa_circuit, Evaluator
    >>> import networkx as nx
    >>> G = nx.cycle_graph(3)
    >>> circuit, beta, gamma = build_qaoa_circuit(G, ['rx'], p=1)
    >>> evaluator = Evaluator(maxiter=50)
    >>> result = evaluator.evaluate(G, circuit, beta, gamma, ['rx'], 1)
    >>> print(f"Energy: {result.energy:.4f}, Ratio: {result.approximation_ratio:.4f}")
    Energy: 2.0000, Ratio: 1.0000
```

#### `SearchResult`

```python
@dataclass
class SearchResult:
    """
    存储单次搜索评估的结果。

    详细描述：
        包含电路结构、性能指标和优化参数等完整信息。
        所有字段在评估完成后填充，可用于后续分析和结果持久化。

    Attributes:
        gate_sequence (Tuple[str, ...]): 门序列元组，例如 ('rx', 'ry', 'cx')。
        p (int): QAOA 深度。
        energy (float): 优化后的期望能量值（期望 cut 大小）。
        approximation_ratio (float): 近似比，范围 [0, 1]，1.0 表示达到最优。
        circuit (Any): Qiskit QuantumCircuit 对象（参数化或已绑定参数）。
        param_assignment (Dict[Any, float]): 优化后的参数赋值字典，键为 Parameter 对象。
    """
```

### 搜索算法 (search_algorithm)

#### `QArchSearch`

```python
class QArchSearch:
    """
    实现 QArchSearch 主搜索流程。

    详细描述：
        按照 Algorithm 1 的流程执行搜索：
        1. 对每个深度 p (1 到 p_max)，枚举所有门组合
        2. 为每个组合构建 QAOA 电路并评估性能
        3. 选择当前深度下的最佳电路
        4. 比较不同深度的最佳结果，返回全局最优
        支持串行和并行两种执行模式，并行模式使用 multiprocessing 加速。

    Attributes:
        config (SearchConfig): 搜索配置对象。
        predictor (Predictor): 门组合生成器。
        evaluator (Evaluator): 电路评估器。
        results_dir (Path): 结果输出目录。

    Methods:
        search: 执行完整的搜索流程。
        _evaluate_candidate: 评估单个候选电路。
        _select_best: 从候选列表中选择最佳结果。
        _persist_results: 持久化搜索结果到文件。
    """
```

#### `search(graph: nx.Graph) -> SearchResult`

```
执行完整的量子架构搜索流程。

详细描述：
        按照 Algorithm 1 实现搜索主循环：
        1. 生成所有可能的门组合（通过 predictor）
        2. 对每个深度 p (1 到 p_max)：
           a. 为每个门组合构建并评估 QAOA 电路
           b. 记录所有候选的性能到日志
           c. 选择当前深度的最佳电路
        3. 比较所有深度的最佳结果，返回全局最优
        4. 将最佳结果持久化到 results_dir 目录

Args:
    graph (nx.Graph): 输入图对象，任意节点标签将被归一化。

Returns:
    SearchResult: 全局最佳搜索结果，包含最优的门序列、深度、能量和近似比。

Raises:
    RuntimeError: 当 predictor 未生成任何组合或搜索未获得有效结果时抛出。

Example:
    >>> from qarchsearch import QArchSearch, SearchConfig
    >>> import networkx as nx
    >>> config = SearchConfig(p_max=2, k_max=2, gate_alphabet=['rx', 'ry'])
    >>> search = QArchSearch(config)
    >>> G = nx.cycle_graph(4)
    >>> result = search.search(G)
    >>> print(f"Best: {result.gate_sequence}, Energy: {result.energy:.4f}")
    Best: ('rx', 'ry'), Energy: 4.0000
```

#### `_evaluate_candidate(graph: nx.Graph, gate_sequence: Sequence[str], p: int) -> SearchResult`

```
评估单个候选电路配置。

详细描述：
        内部方法，用于评估给定的 (graph, gate_sequence, p) 三元组。
        首先构建 QAOA 电路，然后调用 evaluator 进行性能评估。

Args:
    graph (nx.Graph): 输入图对象。
    gate_sequence (Sequence[str]): 门序列。
    p (int): QAOA 深度。

Returns:
    SearchResult: 评估结果对象。

Raises:
    继承自 build_qaoa_circuit 和 evaluator.evaluate 的异常。
```

#### `_select_best(current_best: SearchResult | None, candidates: Iterable[SearchResult]) -> SearchResult | None`

```
从候选列表中选择能量最高的结果。

详细描述：
        比较所有候选的能量值，返回能量最大的 SearchResult。
        如果 current_best 不为 None，则与候选列表中的结果进行比较。

Args:
    current_best (SearchResult | None): 当前最佳结果，如果为 None 则从 candidates 中选择。
    candidates (Iterable[SearchResult]): 候选结果列表。

Returns:
    SearchResult | None: 最佳结果，如果 candidates 为空则返回 current_best。
```

### 并行工具 (parallel_utils)

#### `run_parallel(func: Callable[..., T], tasks: Sequence[Tuple], processes: int | None = None) -> Iterable[T]`

```
使用 multiprocessing 并行执行任务列表。

详细描述：
        根据 processes 参数决定执行模式：
        - processes <= 1 或 tasks 为空：串行执行
        - processes > 1：使用 multiprocessing.Pool 并行执行
        当 processes 为 None 时，自动设置为 min(len(tasks), cpu_count() - 1)。
        使用 starmap 方法，支持多参数函数。

Args:
    func (Callable[..., T]): 要执行的函数，必须可序列化（pickle）。
    tasks (Sequence[Tuple]): 任务列表，每个元素为函数参数的元组。
    processes (int | None): 并行进程数，None 表示自动选择。

Returns:
    Iterable[T]: 执行结果列表，顺序与 tasks 一致。

Raises:
    无显式异常，但函数执行过程中的异常会传播。

Example:
    >>> def add(a, b):
    ...     return a + b
    >>> tasks = [(1, 2), (3, 4), (5, 6)]
    >>> results = list(run_parallel(add, tasks, processes=2))
    >>> print(results)
    [3, 7, 11]
```

## 使用示例

### 示例 1: 基本搜索

```python
from qarchsearch import QArchSearch, SearchConfig, initialize_environment
import networkx as nx

# 初始化环境
config = initialize_environment(overrides={
    'p_max': 2,
    'k_max': 2,
    'gate_alphabet': ['rx', 'ry', 'h', 'cx']
})

# 创建搜索对象
search = QArchSearch(config)

# 加载图
G = nx.read_edgelist('data/sample_graph.edgelist', nodetype=int)

# 执行搜索
result = search.search(G)

# 输出结果
print(f"最佳门序列: {result.gate_sequence}")
print(f"最佳深度: {result.p}")
print(f"能量值: {result.energy:.4f}")
print(f"近似比: {result.approximation_ratio:.4f}")
```

### 示例 2: 并行搜索

```python
from qarchsearch import QArchSearch, SearchConfig
import networkx as nx

# 配置并行搜索
config = SearchConfig(
    p_max=3,
    k_max=3,
    gate_alphabet=['rx', 'ry', 'rz', 'h', 'cx'],
    parallel_processes=4  # 使用 4 个进程
)

search = QArchSearch(config)
G = nx.erdos_renyi_graph(10, 0.3, seed=42)
result = search.search(G)
```

### 示例 3: 自定义评估器

```python
from qarchsearch import QArchSearch, SearchConfig, Evaluator
import networkx as nx

# 创建自定义评估器（增加迭代次数）
evaluator = Evaluator(maxiter=500, tol=1e-4)

config = SearchConfig(p_max=2, k_max=2)
search = QArchSearch(config, evaluator=evaluator)

G = nx.cycle_graph(5)
result = search.search(G)
```

## 测试

运行单元测试：

```bash
pytest
```

运行测试并显示覆盖率：

```bash
pytest --cov=qarchsearch --cov-report=html
```

## 输出文件说明

### `best_circuit.qasm`

OpenQASM 2.0 格式的量子电路，所有参数已绑定为优化后的数值。可直接导入 Qiskit 或其他量子计算框架。

### `metrics.json`

包含以下字段的 JSON 文件：

```json
{
  "energy": 12.0,
  "approximation_ratio": 0.9231,
  "p": 3,
  "gate_sequence": ["rx", "ry", "cx"],
  "parameters": {
    "gamma[0]": 0.123,
    "gamma[1]": 0.456,
    "beta[0]": 0.789,
    ...
  }
}
```

### `search_log.csv`

CSV 格式的搜索日志，包含所有候选电路的评估结果：

| p | gate_sequence | energy | approx_ratio |
|---|---------------|--------|--------------|
| 1 | rx | 8.5000 | 0.6538 |
| 1 | ry | 8.5000 | 0.6538 |
| ... | ... | ... | ... |



## 项目归档

本项目已归档至 [QUANTUM-AND-ML/Diffusion-QC](https://github.com/QUANTUM-AND-ML/Diffusion-QC) 仓库的 `QArchSearch` 目录。

### 归档信息

- **仓库地址**: https://github.com/QUANTUM-AND-ML/Diffusion-QC
- **项目路径**: `/QArchSearch`
- **复现论文**: QArchSearch : A Scalable Quantum Architecture Search Package (SC-W 2023)
- **实现版本**: 基于随机搜索的基线实现




