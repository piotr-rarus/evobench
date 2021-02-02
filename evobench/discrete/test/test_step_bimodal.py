from pytest import fixture

from evobench.discrete.step_bimodal import StepBimodal


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


def test_samples(step_bimodal: StepBimodal, helpers):
    helpers.check_samples(__SAMPLES, step_bimodal)


def test_as_dict(step_bimodal: StepBimodal):
    as_dict = step_bimodal.as_dict
    assert isinstance(as_dict, dict)
