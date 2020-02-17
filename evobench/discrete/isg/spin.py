from dataclasses import dataclass


@dataclass(frozen=True)
class Spin:
    a_index: int
    b_index: int
    factor: int
