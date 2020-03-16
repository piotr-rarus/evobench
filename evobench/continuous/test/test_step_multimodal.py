from pytest import fixture

from evobench.util import check_samples

from ..step_multimodal import StepMultimodal


__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
]


@fixture
def step_multimodal() -> StepMultimodal:
    return StepMultimodal(blocks=[5, 5], step_size=0.1)


def test_samples(step_multimodal: StepMultimodal):
    check_samples(__SAMPLES, step_multimodal)


def test_global_opt(step_multimodal: StepMultimodal):
    assert isinstance(step_multimodal.global_opt, float)
    assert step_multimodal.global_opt == 2
