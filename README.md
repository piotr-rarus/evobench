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
- multimodal
- step multimodal
- sawtooth

## Getting started

```sh
pip install evobench
```

```py
import evobench


trap = evobench.discrete.Trap(block_size=5, repetitions=3)
initialization = evobench.discrete.initialization.Uniform(population_size=4e3)

population = initialization.initialize_population(trap.genome_size)
fitness = trap.evaluate_population(population)
```

You can also evaluate single solution.

```py
fitness = trap.evaluate_solution(population.solutions[0])
```

Every time you're evaluating solutions we increment _ffe_ counter.
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

## How to implement your own function

### Fully separable

You need to inherit `Separable` class from `evobench.separable`.
Then just implement `def evaluate_block(self, block: np.ndarray) -> int` method. Best follow `evobench.discrete.trap` implementation.

### Other

Inherit `Benchmark` class from `evobench.benchmark`. Then implement `def _evaluate_solution(self, solution: Solution) -> float` method.

## Coming soon

We'll be adding more problems in near future. If you're looking for any particular problem, please mail us or open an issue.

We're thinking about interactive visualizations, so you can sample the space and check how it looks. It's easier than digging through definitions.

We're working on linkage quality metrics. Once they're published, we'll incorporate them to this package.
