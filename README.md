# Evobench

Evobench is a collection of benchmark problems dedicated for model-based large scale optimization.

## Overview

This package contains following problems.

### Discrete

- trap
- step trap
- bimodal
- step bimodal
- HIFF
- Ising Spin Glass

### Continuous

- trap
- step trap
- multimodal
- step multimodal
- sawtooth

### Compound

You can create your own benchmark made of ther benchmarks.

## Getting started

```sh
pip install evobench
```

```py
import evobench


trap = evobench.discrete.Trap(blocks=[4, 4, 4])

population = trap.initialize_population(population_size=1000)
fitness = trap.evaluate_population(population)
```

You can also evaluate single solution.

```py
fitness = trap.evaluate_solution(population.solutions[0])
```

Every time you're evaluating solutions we increment _ffe_ counter.
Solution is not evaluated again, if it didn't change.
You can access it through `benchmark` instance.

```py
print(trap.ffe)
```

## Ising Spin Glass

To instantiate _ISG_ you need to pass specific problem configuration.

```py
from evobench.discrete import IsingSpinGlass


isg = IsingSpinGlass('IsingSpinGlass_pm_16_0')
```

You can find 5,000 instances at `evobench\discrete\isg\data` folder. Instances vary in length and complexity.

## Compound Benchmark

Creating your own compound benchmarks is really easy.
You just need to define your sub-benchmarks and pass them as a list. All other fuctions work just the same as with the normal `Benchmark`.

```py
from evobench import CompoundBenchmark
from evobench import continuous, discrete


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

## How to implement your own function

### Fully separable

You need to inherit `Separable` class from `evobench.separable`.
Then just implement `def evaluate_block(self, block: np.ndarray) -> int` method. Best follow `evobench.discrete.trap` implementation.

### Other

Inherit `Benchmark` class from `evobench.benchmark`. Then implement `def _evaluate_solution(self, solution: Solution) -> float` method.

## Linkage quality

Linkage quality metrics are located at `evobench.linkage.metrics`.
Available metrics:
    - fill quality

## Coming soon

We'll be adding more problems in the near future. If you're looking for any particular problem, please mail us or open an issue.
We're working on linkage quality metrics. Once they're published, we'll be incorporating them to this package.
