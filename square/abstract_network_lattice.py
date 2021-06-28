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
from typing import List, Tuple
from multimethod import multimethod
import TAT
from .abstract_lattice import AbstractLattice

__all__ = ["AbstractNetworkLattice"]

CTensor: type = TAT.No.Z.Tensor
Tensor: type = TAT.No.D.Tensor


class AbstractNetworkLattice(AbstractLattice):
    __slots__ = ["dimension_virtual", "_lattice"]

    @multimethod
    def __init__(self, M: int, N: int, *, D: int, d: int) -> None:
        super().__init__(M, N, d=d)
        self.dimension_virtual: int = D

        self._lattice: List[List[Tensor]] = [[self._initialize_tensor_in_network(i, j) for j in range(self.N)] for i in range(self.M)]

    @multimethod
    def __init__(self, other: AbstractNetworkLattice) -> None:
        super().__init__(other)

        self.dimension_virtual: int = other.dimension_virtual
        self._lattice: List[List[Tensor]] = [[other[i, j] for j in range(self.N)] for i in range(self.M)]

    def __getitem__(self, position: Tuple[int, int]) -> Tensor:
        return self._lattice[position[0]][position[1]]

    def __setitem__(self, position: Tuple[int, int], value: Tensor) -> None:
        self._lattice[position[0]][position[1]] = value

    def _initialize_tensor_in_network(self, i: int, j: int) -> Tensor:
        name_list = ["P"]
        dimension_list = [self.dimension_physics]
        if i != 0:
            name_list.append("U")
            dimension_list.append(self.dimension_virtual)
        if j != 0:
            name_list.append("L")
            dimension_list.append(self.dimension_virtual)
        if i != self.M - 1:
            name_list.append("D")
            dimension_list.append(self.dimension_virtual)
        if j != self.N - 1:
            name_list.append("R")
            dimension_list.append(self.dimension_virtual)
        return Tensor(name_list, dimension_list).randn()
