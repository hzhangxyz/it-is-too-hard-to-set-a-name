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
from typing import Dict, Optional, Tuple
from multimethod import multimethod
import TAT
from .abstract_network_lattice import AbstractNetworkLattice
from .auxiliaries import SquareAuxiliariesSystem
from . import sampling_gradient_lattice

__all__ = ["SimpleUpdateLattice"]

CTensor: type = TAT.No.Z.Tensor
Tensor: type = TAT.No.D.Tensor

clear_line = "\u001b[2K"


class SimpleUpdateLattice(AbstractNetworkLattice):
    __slots__ = ["environment", "auxiliary"]

    @multimethod
    def __init__(self, M: int, N: int, *, D: int, d: int) -> None:
        super().__init__(M, N, D=D, d=d)

        self.environment: Dict[Tuple[str, int, int], Tensor] = {}
        # 第一个str可以是"D"或者"R"

        self.auxiliary: Optional[SquareAuxiliariesSystem] = None

    @multimethod
    def __init__(self, other: SimpleUpdateLattice) -> None:
        super().__init__(other)

        self.environment: Dict[Tuple[str, int, int], Tensor] = other.environment.copy()

        self.auxiliary: Optional[SquareAuxiliariesSystem] = None
        if other.auxiliary is not None:
            self.auxiliary = SquareAuxiliariesSystem(other.auxiliary)

    @multimethod
    def __init__(self, other: sampling_gradient_lattice.SamplingGradientLattice) -> None:
        super().__init__(other)

        self.environment: Dict[Tuple[str, int, int], Tensor] = {}

        self.auxiliary: Optional[SquareAuxiliariesSystem] = None

    def _construct_environment(self) -> None:
        for i in range(self.M):
            for j in range(self.N - 1):
                self.environment["R", i, j] = Tensor([",SVD_U", ",SVD_V"], [self.dimension_virtual, self.dimension_virtual]).identity({(",SVD_U", ",SVD_V")})

        for i in range(self.M - 1):
            for j in range(self.N):
                self.environment["D", i, j] = Tensor([",SVD_U", ",SVD_V"], [self.dimension_virtual, self.dimension_virtual]).identity({(",SVD_U", ",SVD_V")})

    def simple_update(self, time: int, delta_t: float, new_dimension: int = 0) -> None:
        if new_dimension != 0:
            self.dimension_virtual = new_dimension
        updater: Dict[Tuple[Tuple[int, int], ...], Tensor] = {}
        for positions, term in self.hamiltonian.items():
            site_number: int = len(positions)
            updater[positions] = (-delta_t * term).exponential({(f"I{i}", f"O{i}") for i in range(site_number)}, 8)
        for t in range(time):
            print(f"Simple updating, total_step={time}, delta_t={delta_t}, dimension={self.dimension_virtual}, step={t}", end="\r")
            for positions, term in updater.items():
                self._single_term_simple_update(positions, term)
            # 老版本python不支持直接对dict_items直接做reversed
            for positions, term in reversed(list(updater.items())):
                self._single_term_simple_update(positions, term)
        print(f"{clear_line}Simple update done, total_step={time}, delta_t={delta_t}, dimension={self.dimension_virtual}")

    def _single_term_simple_update(self, positions: Tuple[Tuple[int, int], ...], updater: Tensor) -> None:
        if len(positions) == 1:
            return self._single_term_simple_update_single_site(positions[0], updater)
        if len(positions) == 2:
            position_1, position_2 = positions
            if position_1[0] == position_2[0]:
                if position_1[1] == position_2[1] + 1:
                    return self._single_term_simple_update_double_site_nearest_horizontal(position_2, updater)
                if position_2[1] == position_1[1] + 1:
                    return self._single_term_simple_update_double_site_nearest_horizontal(position_1, updater)
            if position_1[1] == position_2[1]:
                if position_1[0] == position_2[0] + 1:
                    return self._single_term_simple_update_double_site_nearest_vertical(position_2, updater)
                if position_2[0] == position_1[0] + 1:
                    return self._single_term_simple_update_double_site_nearest_vertical(position_1, updater)
        raise NotImplementedError("Unsupported simple update style")

    def _single_term_simple_update_double_site_nearest_horizontal(self, position: Tuple[int, int], updater: Tensor) -> None:
        i, j = position
        left = self[i, j]
        left = self.try_multiple(left, i, j, "L")
        left = self.try_multiple(left, i, j, "U")
        left = self.try_multiple(left, i, j, "D")
        left = self.try_multiple(left, i, j, "R")
        right = self[i, j + 1]
        right = self.try_multiple(right, i, j + 1, "U")
        right = self.try_multiple(right, i, j + 1, "D")
        right = self.try_multiple(right, i, j + 1, "R")
        left_q, left_r = left.qr("r", {"P", "R"}, "R", "L")
        right_q, right_r = right.qr("r", {"P", "L"}, "L", "R")
        u, s, v = left_r.edge_rename({"P": "P0"}) \
            .contract(right_r.edge_rename({"P": "P1"}), {("R", "L")}) \
            .contract(updater, {("P0", "I0"), ("P1", "I1")}) \
            .svd({"L", "O0"}, "R", "L", self.dimension_virtual)
        s /= s.norm_max()
        self.environment["R", i, j] = s
        u = u.contract(left_q, {("L", "R")}).edge_rename({"O0": "P"})
        u = self.try_multiple(u, i, j, "L", True)
        u = self.try_multiple(u, i, j, "U", True)
        u = self.try_multiple(u, i, j, "D", True)
        u /= u.norm_max()
        self[i, j] = u
        v = v.contract(right_q, {("R", "L")}).edge_rename({"O1": "P"})
        v = self.try_multiple(v, i, j + 1, "U", True)
        v = self.try_multiple(v, i, j + 1, "D", True)
        v = self.try_multiple(v, i, j + 1, "R", True)
        v /= v.norm_max()
        self[i, j + 1] = v

    def _single_term_simple_update_double_site_nearest_vertical(self, position: Tuple[int, int], updater: Tensor) -> None:
        i, j = position
        up = self[i, j]
        up = self.try_multiple(up, i, j, "L")
        up = self.try_multiple(up, i, j, "U")
        up = self.try_multiple(up, i, j, "D")
        up = self.try_multiple(up, i, j, "R")
        down = self[i + 1, j]
        down = self.try_multiple(down, i + 1, j, "L")
        down = self.try_multiple(down, i + 1, j, "D")
        down = self.try_multiple(down, i + 1, j, "R")
        up_q, up_r = up.qr("r", {"P", "D"}, "D", "U")
        down_q, down_r = down.qr("r", {"P", "U"}, "U", "D")
        u, s, v = up_r.edge_rename({"P": "P0"}) \
            .contract(down_r.edge_rename({"P": "P1"}), {("D", "U")}) \
            .contract(updater, {("P0", "I0"), ("P1", "I1")}) \
            .svd({"U", "O0"}, "D", "U", self.dimension_virtual)
        s /= s.norm_max()
        self.environment["D", i, j] = s
        u = u.contract(up_q, {("U", "D")}).edge_rename({"O0": "P"})
        u = self.try_multiple(u, i, j, "L", True)
        u = self.try_multiple(u, i, j, "U", True)
        u = self.try_multiple(u, i, j, "R", True)
        u /= u.norm_max()
        self[i, j] = u
        v = v.contract(down_q, {("D", "U")}).edge_rename({"O1": "P"})
        v = self.try_multiple(v, i + 1, j, "L", True)
        v = self.try_multiple(v, i + 1, j, "D", True)
        v = self.try_multiple(v, i + 1, j, "R", True)
        v /= v.norm_max()
        self[i + 1, j] = v

    def _single_term_simple_update_single_site(self, position: Tuple[int, int], updater: Tensor) -> None:
        i, j = position
        self[i, j] = self[i, j].contract(updater, {("P", "I0")}).edge_rename({"O0": "P"})

    def try_multiple(self, tensor: Tensor, i: int, j: int, direction: str, division: bool = False) -> Tensor:
        if direction == "L":
            if ("R", i, j - 1) in self.environment:
                return tensor.multiple(self.environment["R", i, j - 1], "L", "v", division)
        if direction == "U":
            if ("D", i - 1, j) in self.environment:
                return tensor.multiple(self.environment["D", i - 1, j], "U", "v", division)
        if direction == "D":
            if ("D", i, j) in self.environment:
                return tensor.multiple(self.environment["D", i, j], "D", "u", division)
        if direction == "R":
            if ("R", i, j) in self.environment:
                return tensor.multiple(self.environment["R", i, j], "R", "u", division)
        return tensor

    def initialize_auxiliary(self, Dc: int) -> None:
        self.auxiliary = SquareAuxiliariesSystem(self.M, self.N, Dc)
        for i in range(self.M):
            for j in range(self.N):
                this_site = self[i, j]
                this_site = self.try_multiple(this_site, i, j, "D")
                this_site = self.try_multiple(this_site, i, j, "R")
                double_site = this_site.edge_rename({"L": "L1", "R": "R1", "U": "U1", "D": "D1"}).contract(this_site.edge_rename({"L": "L2", "R": "R2", "U": "U2", "D": "D2"}), {("P", "P")})
                merge_map = {}
                if i != 0:
                    merge_map["U"] = ["U1", "U2"]
                if j != 0:
                    merge_map["L"] = ["L1", "L2"]
                if i != self.M - 1:
                    merge_map["D"] = ["D1", "D2"]
                if j != self.N - 1:
                    merge_map["R"] = ["R1", "R2"]
                self.auxiliary[i, j] = double_site.merge_edge(merge_map)

    def denominator(self) -> float:
        if self.auxiliary == None:
            raise ValueError("Need to initialize auxiliary before call auxiliary observe")
        return float(self.auxiliary[None])

    @multimethod
    def observe(self, positions: Tuple[Tuple[int, int], ...], observer: Tensor, calculate_denominator: bool = True) -> float:
        if self.auxiliary == None:
            raise ValueError("Need to initialize auxiliary before call auxiliary observe")
        numerator_with_hole: Tensor = self.auxiliary[positions]
        single_side: Tensor = Tensor(1)
        all_remained = set()
        for index, position in enumerate(positions):
            this_site = self[position]
            this_site = self.try_multiple(this_site, position[0], position[1], "D")
            this_site = self.try_multiple(this_site, position[0], position[1], "R")
            this_site = this_site.edge_rename({f"{direction}": f"{direction}{index}" for direction in "LRUDP"})
            contract_set = set()
            x1, y1 = position
            if x1 != 0:
                all_remained.add(f"U{index}")
            if y1 != 0:
                all_remained.add(f"L{index}")
            if x1 != self.M - 1:
                all_remained.add(f"D{index}")
            if y1 != self.N - 1:
                all_remained.add(f"R{index}")
            for other_index in range(index):
                other_position = positions[other_index]
                x2, y2 = other_position
                if x2 == x1 and y2 + 1 == y1:
                    # -X
                    contract_set.add((f"R{other_index}", f"L{index}"))
                    all_remained.remove(f"L{index}")
                    all_remained.remove(f"R{other_index}")
                elif x2 == x1 and y2 - 1 == y1:
                    # X-
                    contract_set.add((f"L{other_index}", f"R{index}"))
                    all_remained.remove(f"R{index}")
                    all_remained.remove(f"L{other_index}")
                elif x2 + 1 == x1 and y2 == y1:
                    # |
                    # X
                    contract_set.add((f"D{other_index}", f"U{index}"))
                    all_remained.remove(f"U{index}")
                    all_remained.remove(f"D{other_index}")
                elif x2 - 1 == x1 and y2 == y1:
                    # X
                    # |
                    contract_set.add((f"U{other_index}", f"D{index}"))
                    all_remained.remove(f"D{index}")
                    all_remained.remove(f"U{other_index}")
            single_side = single_side.contract(this_site, contract_set)
        numerator_holes: Tensor = single_side.edge_rename({f"{name}": f"{name}.1" for name in all_remained}) \
            .contract(observer, {(f"P{index}", f"I{index}") for index in range(len(positions))}) \
            .contract(single_side.edge_rename({f"{name}": f"{name}.2" for name in all_remained}), \
                    {(f"O{index}", f"P{index}") for index in range(len(positions))}) \
            .merge_edge({f"{name}": [f"{name}.1", f"{name}.2"] for name in all_remained})
        numerator: Tensor = numerator_holes.contract_all_edge(numerator_with_hole)
        if calculate_denominator:
            return float(numerator) / self.denominator()
        else:
            return float(numerator)

    def observe_energy(self) -> float:
        energy = 0
        for positions, observer in self.hamiltonian.items():
            energy += self.observe(positions, observer, False)
        return energy / self.denominator() / (self.M * self.N)
