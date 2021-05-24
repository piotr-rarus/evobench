from abc import ABC, abstractproperty

from evobench.linkage.dsm import DependencyStructureMatrix


class DependencyStructureMatrixMixin(ABC):

    @abstractproperty
    def dsm(self) -> DependencyStructureMatrix:
        pass
