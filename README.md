![PyPI](https://img.shields.io/pypi/v/evobench)
![PyPI - License](https://img.shields.io/pypi/l/evobench)
![PyPI - Downloads](https://img.shields.io/pypi/dm/evobench)
[![codecov](https://codecov.io/gh/piotr-rarus/evobench/branch/master/graph/badge.svg?token=D2M7V412G0)](https://codecov.io/gh/piotr-rarus/evobench)

# Evobench

Evobench is a collection of benchmark problems dedicated for optimization problems (both synthetic and practical). Please note that Python isn't still best tool for solving optimization problems, as loops are still slow. This might change in a next couple of years. Our main intention is to provide easily accessible package for PoC, research or teaching purposes.

## Getting started

```sh
pip install evobench
```

```py
import evobench


trap = evobench.discrete.Trap(blocks=[4, 4, 4])

population = trap.initialize_population(population_size=1e3)
trap.evaluate_population(population)
```

Fitness evaluation produces side effect of defining solution's fitness.

```py
print(population.solutions[0].fitness)
```

You can also evaluate single solution.

```py
fitness = trap.evaluate_solution(population.solutions[0])
```

Every time you evaluate undefined solution we increment `ffe` counter.
Solution is not evaluated again, if it didn't change.
You can access it through a `benchmark` instance.

```py
print(trap.ffe)
```

## Overview

This package exposes following problems.

### Practical

- TSP
- cVRP

### Discrete

- Trap
- Step Trap
- Bimodal
- Step Bimodal
- HIFF
- Ising Spin Glass

### Continuous

- Trap
- Step Trap
- Multimodal
- Step Multimodal
- Sawtooth

## Compound Benchmark

Creating your own compound benchmarks is really easy.
You just need to define your sub-benchmarks and pass them as a list. All other fuctions work just the same as with the normal `Benchmark`.

```py
from evobench import CompoundBenchmark, continuous, discrete


benchmark = CompoundBenchmark(
    benchmarks=[
        discrete.Trap(blocks=[5, 2, 4]),
        continuous.Trap(blocks=[3, 6, 4])
    ],
    use_shuffle=True,
    multiprocessing=True,
    verbose=1
)

population = benchmark.initialize_population(population_size=1000)
benchmark.evaluate_population(population)
```

## Ising Spin Glass

To instantiate _ISG_ you need to pass specific problem configuration.

```py
from evobench.discrete import IsingSpinGlass


isg = IsingSpinGlass('IsingSpinGlass_pm_16_0')
```

You can find 5,000 instances at `evobench\discrete\isg\data` folder. Instances vary in length and complexity.

## How to implement your own benchmark

Inherit `Benchmark` class from `evobench.benchmark`. Then implement:

- `def _evaluate_solution(self, solution: Solution) -> float`
- `def random_solutions(self, population_size: int) -> List[Solution]`

### Partially separable

You need to inherit `Separable` class from `evobench.separable`.
Then just implement:

- `def evaluate_block(self, block: np.ndarray) -> int`.
  
Best follow `evobench.discrete.trap` implementation.

## Linkage quality

Linkage quality metrics are located at `evobench.linkage.metrics`.
Available metrics:

- Mean Reciprocal Ranking @K
- Mean Average Precision @K
- NDCG @K $B
- Fill Quality

## Coming soon

We'll be adding more problems in the near future. If you're looking for any particular problem, please mail us or open an issue.
We're working on linkage quality metrics. Once they're published, we'll be incorporating them to this package.
