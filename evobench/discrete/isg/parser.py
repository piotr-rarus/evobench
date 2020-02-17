from .config import Config
from .spin import Spin
from pathlib import Path
import numpy as np


def load(path: Path) -> Config:

    with path.open() as file:
        lines = file.readlines()

        global_optimum, best_solution = lines[0].split(' ')
        global_optimum = float(global_optimum.strip())

        best_solution = best_solution.strip()
        best_solution = [float(gene) for gene in best_solution]
        best_solution = np.array(best_solution)

        spin_configs = []

        for line in lines[2:]:
            a_index, b_index, factor = line.split(' ')

            a_index = int(a_index)
            b_index = int(b_index)
            factor = int(factor)

            spin_config = Spin(
                a_index,
                b_index,
                factor
            )

            spin_configs.append(spin_config)

        config = Config(
            path.name,
            global_optimum,
            best_solution,
            spin_configs
        )

        return config
