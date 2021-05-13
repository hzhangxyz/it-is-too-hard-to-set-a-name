#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
from __future__ import annotations
from typing import Tuple
from multimethod import multimethod
import TAT
from .abstract_lattice import AbstractLattice
from . import simple_update_lattice
from . import sampling_gradient_lattice

__all__ = ["ExactLattice"]

CTensor: type = TAT.No.Z.Tensor
Tensor: type = TAT.No.D.Tensor


class ExactLattice(AbstractLattice):
    __slots__ = ["vector"]

    @multimethod
    def __init__(self, M: int, N: int, *, d: int) -> None:
        super().__init__(M, N, d=d)
        self.vector: Tensor = self._initialize_vector()

    @multimethod
    def __init__(self, other: ExactLattice) -> None:
        super().__init__(other)
        self.vector: Tensor = other.vector

    @multimethod
    def __init__(self, other: simple_update_lattice.SimpleUpdateLattice) -> None:
        super().__init__(other)

        self.vector: Tensor = Tensor(1)
        for i in range(self.M):
            for j in range(self.N):
                to_contract = other[i, j]
                to_contract = other.try_multiple(to_contract, i, j, "D")
                to_contract = other.try_multiple(to_contract, i, j, "R")
                self.vector = self.vector.contract(to_contract.edge_rename({"D": f"D-{j}", "P": f"P-{i}-{j}"}), {("R", "L"), (f"D-{j}", "U")})
                # print("Singularity:", self.vector.norm_max())
                self.vector /= self.vector.norm_max()

    @multimethod
    def __init__(self, other: sampling_gradient_lattice.SamplingGradientLattice) -> None:
        super().__init__(other)

        self.vector: Tensor = Tensor(1)
        for i in range(self.M):
            for j in range(self.N):
                to_contract = other[i, j]
                self.vector = self.vector.contract(to_contract.edge_rename({"D": f"D-{j}", "P": f"P-{i}-{j}"}), {("R", "L"), (f"D-{j}", "U")})
                # print("Singularity:", self.vector.norm_max())
                self.vector /= self.vector.norm_max()

    def _initialize_vector(self) -> Tensor:
        name_list = [f"P-{i}-{j}" for i in range(self.M) for j in range(self.N)]
        dimension_list = [self.dimension_physics for _ in range(self.M) for _ in range(self.N)]
        vector = Tensor(name_list, dimension_list).randn()
        vector /= vector.norm_max()
        return vector

    def update(self, time: int = 1, approximate_energy: float = -0.5, print_energy: bool = False) -> float:
        total_approximate_energy: float = abs(approximate_energy) * self.M * self.N
        energy: float = 0
        for _ in range(time):
            temporary_vector: Tensor = self.vector.same_shape().zero()
            for positions, value in self.hamiltonian.items():
                temporary_vector += self.vector.contract_all_edge(value.edge_rename({f"I{t}": f"P-{i}-{j}" for t, [i, j] in enumerate(positions)})) \
                    .edge_rename({f"O{t}": f"P-{i}-{j}" for t, [i, j] in enumerate(positions)})
            self.vector *= total_approximate_energy
            self.vector -= temporary_vector
            # v <- a v - H v = (a - H) v => E = a - v'/v
            norm_max: float = float(self.vector.norm_max())
            energy = total_approximate_energy - norm_max
            self.vector /= norm_max
            if print_energy:
                print(energy / (self.M * self.N))
        return energy / (self.M * self.N)

    def denominator(self) -> float:
        return float(self.vector.contract_all_edge(self.vector))

    @multimethod
    def observe(self, positions: Tuple[Tuple[int, int], ...], observer: Tensor, calculate_denominator: bool = True) -> float:
        numerator: Tensor = self.vector.contract_all_edge(observer.edge_rename({f"I{t}": f"P-{i}-{j}" for t, [i, j] in enumerate(positions)
                                                                               })).edge_rename({f"O{t}": f"P-{i}-{j}" for t, [i, j] in enumerate(positions)}).contract_all_edge(self.vector)
        if calculate_denominator:
            return float(numerator) / self.denominator()
        else:
            return float(numerator)

    @multimethod
    def observe(self, positions: Tuple[Tuple[int, int], ...], observer: CTensor, calculate_denominator: bool = True) -> float:
        complex_vector: CTensor = self.vector.to(complex)
        numerator: Tensor = complex_vector.contract_all_edge(observer.edge_rename({f"I{t}": f"P-{i}-{j}" for t, [i, j] in enumerate(positions)
                                                                                  })).edge_rename({f"O{t}": f"P-{i}-{j}" for t, [i, j] in enumerate(positions)}).contract_all_edge(complex_vector)
        if calculate_denominator:
            return complex(numerator).real / self.denominator()
        else:
            return complex(numerator).real

    def observe_energy(self) -> float:
        energy = 0
        for positions, observer in self.hamiltonian.items():
            energy += self.observe(positions, observer, False)
        return energy / self.denominator() / (self.M * self.N)
