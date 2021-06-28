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
import TAT
from .auxiliaries import SquareAuxiliariesSystem
from .exact_lattice import ExactLattice
from .simple_update_lattice import SimpleUpdateLattice
from .sampling_gradient_lattice import SamplingGradientLattice

__all__ = ["SquareAuxiliariesSystem", "ExactLattice", "SimpleUpdateLattice", "SamplingGradientLattice", "CTensor", "Tensor", "Sx", "Sy", "Sz", "SxSx", "SySy", "SzSz", "SS"]

CTensor: type = TAT.No.Z.Tensor
Tensor: type = TAT.No.D.Tensor

Sx: Tensor = Tensor(["I0", "O0"], [2, 2])
Sx.block[{}] = [[0, 0.5], [0.5, 0]]
Sy: CTensor = CTensor(["I0", "O0"], [2, 2])
Sy.block[{}] = [[0, -0.5j], [0.5j, 0]]
Sz: Tensor = Tensor(["I0", "O0"], [2, 2])
Sz.block[{}] = [[0.5, 0], [0, -0.5]]

SxSx: Tensor = Sx.edge_rename({"I0": "I1", "O0": "O1"}).contract_all_edge(Sx).to(float)
SySy: Tensor = Sy.edge_rename({"I0": "I1", "O0": "O1"}).contract_all_edge(Sy).to(float)
SzSz: Tensor = Sz.edge_rename({"I0": "I1", "O0": "O1"}).contract_all_edge(Sz).to(float)

SS: Tensor = SxSx + SySy + SzSz
