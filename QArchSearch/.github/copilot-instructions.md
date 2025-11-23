# Copilot Instructions for QArchSearch (Diffusion-QC)

## Project Overview
- **QArchSearch** is a modular Python package for automated quantum architecture search, focused on finding optimal QAOA mixer circuits for a given gate alphabet and graph.
- The codebase is organized for extensibility, supporting both serial and parallel search, and is designed for large-scale quantum circuit modeling.

## Key Components & Data Flow
- **qarchsearch/**: Core package. Major modules:
  - `foundation.py`: Configuration management (`SearchConfig`), environment setup (`initialize_environment`).
  - `predictor.py`: Generates candidate gate sequences (random search baseline, extensible to ML-based predictors).
  - `qbuilder.py`: Builds parameterized QAOA circuits from gate sequences and graphs.
  - `evaluator.py`: Trains and evaluates circuits using COBYLA optimizer, computes energy and approximation ratio.
  - `search_algorithm.py`: Orchestrates the main search loop (Algorithm 1), supports serial and multiprocessing execution.
  - `parallel_utils.py`: Utilities for parallel execution (multiprocessing).
- **main.py**: CLI entry point. Parses arguments, loads graph, runs search, writes results.
- **results/**: Output directory (auto-created). Stores `best_circuit.qasm`, `metrics.json`, `search_log.csv`.
- **tests/**: Unit tests (pytest compatible).

## Developer Workflows
- **Install dependencies**: `pip install -r requirements.txt`
- **Run search**: `python main.py --graph data/sample_graph.edgelist --p-max 2 --k-max 2 --gates rx,ry,h,cx`
- **Test**: `pytest` or `pytest --cov=qarchsearch --cov-report=html`
- **Custom config**: Use `initialize_environment` or pass overrides to `SearchConfig`.
- **Parallel search**: Set `--parallel N` or `parallel_processes` in config for multiprocessing.

## Project-Specific Patterns
- **Configuration**: All search parameters are managed via `SearchConfig` (dataclass). Supports JSON/YAML config files and runtime overrides.
- **Gate sequence generation**: Use `Predictor` for candidate enumeration or sampling. Extendable for ML-based search.
- **Circuit construction**: Always use `build_qaoa_circuit` for QAOA circuit creation; ensures node relabeling and parameter consistency.
- **Evaluation**: Use `Evaluator.evaluate` for circuit assessment; returns a `SearchResult` dataclass with all metrics.
- **Results**: All outputs are written to `results/` in standard formats (QASM, JSON, CSV).

## Integration & Extensibility
- **Add new search strategies**: Implement new predictors in `predictor.py` and update `QArchSearch` logic.
- **Custom evaluators**: Pass a custom `Evaluator` instance to `QArchSearch` for advanced optimization or metrics.
- **Data**: Input graphs must be in edgelist format. Node labels are normalized internally.

## Examples
- See `README.md` for full code and CLI usage examples.
- Example: Parallel search with custom gates:
  ```python
  config = SearchConfig(p_max=3, k_max=3, gate_alphabet=['rx','ry','rz','h','cx'], parallel_processes=4)
  search = QArchSearch(config)
  G = nx.erdos_renyi_graph(10, 0.3, seed=42)
  result = search.search(G)
  ```

## Conventions
- All core logic is in `qarchsearch/`.
- Only use public APIs (`QArchSearch`, `SearchConfig`, `Predictor`, `Evaluator`, `build_qaoa_circuit`).
- Prefer config-driven workflows over hardcoding parameters.
- All results and logs are reproducible via config and random seed.

---
For more, see `README.md` and code docstrings. When in doubt, follow the patterns in `main.py` and `qarchsearch/` modules.
