from pytest import fixture

from evobench.util import check_samples

from ..max_sat import MaxSat

__BLOCK_SIZE = 3
__REPETITIONS = 2
__K_CLAUSES = 2
__RANDOM_SEED = 23423

__SAMPLES = [
    ([1, 1, 1, 1, 1, 1], 4),
    ([0, 0, 0, 0, 0, 0], 3),
    ([1, 0, 0, 0, 0, 0], 2)
]


@fixture
def max_sat() -> MaxSat:
    return MaxSat(
        __K_CLAUSES,
        __BLOCK_SIZE,
        __REPETITIONS,
        random_seed=__RANDOM_SEED
    )


def test_clauses(max_sat: MaxSat):
    clauses = max_sat.clauses

    assert isinstance(clauses, list)
    assert all(isinstance(clause, list) for clause in clauses)


def test_samples(max_sat: MaxSat):
    check_samples(__SAMPLES, max_sat)
