from pytest import fixture

from evobench.discrete.step_bimodal import StepBimodal
from evobench.util import check_samples


__SAMPLES = [
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3),
    ([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 2),
    ([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], 1),
]


@fixture
def step_bimodal() -> StepBimodal:
    return StepBimodal(
        blocks=[11],
        step_size=2
    )


def test_samples(step_bimodal: StepBimodal):
    check_samples(__SAMPLES, step_bimodal)


def test_global_opt(step_bimodal: StepBimodal):
    assert isinstance(step_bimodal.global_opt, float)
    assert step_bimodal.global_opt == 3
