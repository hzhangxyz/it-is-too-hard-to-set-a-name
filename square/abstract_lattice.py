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
from typing import Dict, Tuple
from multimethod import multimethod
import TAT

__all__ = ["AbstractLattice"]

CTensor: type = TAT.No.Z.Tensor
Tensor: type = TAT.No.D.Tensor


class AbstractLattice:
    __slots__ = ["M", "N", "dimension_physics", "hamiltonian"]

    @multimethod
    def __init__(self, M: int, N: int, *, d: int) -> None:
        self.M: int = M
        self.N: int = N
        self.dimension_physics: int = d
        self.hamiltonian: Dict[Tuple[Tuple[int, int], ...], Tensor] = {}
        # 可以是任意体哈密顿量, 哈密顿量的name使用I0, I1, ..., O0, O1, ...

    @multimethod
    def __init__(self, other: AbstractLattice) -> None:
        self.M: int = other.M
        self.N: int = other.N
        self.dimension_physics: int = other.dimension_physics
        self.hamiltonian: Dict[Tuple[Tuple[int, int], ...], Tensor] = other.hamiltonian.copy()

    @staticmethod
    def _check_hamiltonian_name(tensor: Tensor, body: int) -> None:
        if {f"{i}" for i in tensor.name} != {f"{i}{j}" for i in ["I", "O"] for j in range(body)}:
            raise ValueError("Wrong hamiltonian name")

    @property
    def single_site_hamiltonian(self) -> None:
        raise RuntimeError("Getting hamiltonian is not allowed")

    @single_site_hamiltonian.setter
    def single_site_hamiltonian(self, value: Tensor) -> None:
        self._check_hamiltonian_name(value, 1)
        for i in range(self.M):
            for j in range(self.N):
                if ((i, j),) in self.hamiltonian:
                    raise ValueError("Hamiltonian term have already set")
                else:
                    self.hamiltonian[((i, j),)] = value

    @property
    def vertical_bond_hamiltonian(self) -> None:
        raise RuntimeError("Getting hamiltonian is not allowed")

    @vertical_bond_hamiltonian.setter
    def vertical_bond_hamiltonian(self, value: Tensor) -> None:
        self._check_hamiltonian_name(value, 2)
        for i in range(self.M - 1):
            for j in range(self.N):
                if ((i, j), (i + 1, j)) in self.hamiltonian:
                    raise ValueError("Hamiltonian term have already set")
                else:
                    self.hamiltonian[(i, j), (i + 1, j)] = value

    @property
    def horizontal_bond_hamiltonian(self) -> None:
        raise RuntimeError("Getting hamiltonian is not allowed")

    @horizontal_bond_hamiltonian.setter
    def horizontal_bond_hamiltonian(self, value: Tensor) -> None:
        self._check_hamiltonian_name(value, 2)
        for i in range(self.M):
            for j in range(self.N - 1):
                if ((i, j), (i, j + 1)) in self.hamiltonian:
                    raise ValueError("Hamiltonian term have already set")
                else:
                    self.hamiltonian[(i, j), (i, j + 1)] = value
