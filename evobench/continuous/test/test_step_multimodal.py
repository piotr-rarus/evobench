from pytest import fixture

from evobench.util import check_samples

from ..step_multimodal import StepMultimodal

__BLOCK_SIZE = 5
__REPETITIONS = 2

__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
]


@fixture
def step_multimodal() -> StepMultimodal:
    return StepMultimodal(
        block_size=__BLOCK_SIZE,
        repetitions=__REPETITIONS,
        step_size=0.1,
    )


def test_samples(step_multimodal: StepMultimodal):
    check_samples(__SAMPLES, step_multimodal)
