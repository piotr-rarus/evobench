from pytest import fixture

from ..step_multimodal import StepMultimodal

__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
]


@fixture
def step_multimodal() -> StepMultimodal:
    return StepMultimodal(blocks=[5, 5], step_size=0.1)


def test_samples(step_multimodal: StepMultimodal, helpers):
    helpers.check_samples(__SAMPLES, step_multimodal)
