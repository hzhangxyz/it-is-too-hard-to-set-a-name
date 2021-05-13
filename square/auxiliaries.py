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
from typing import Dict, List, Optional, Tuple
from multimethod import multimethod
import TAT

__all__ = ["SquareAuxiliariesSystem"]

CTensor: type = TAT.No.Z.Tensor
Tensor: type = TAT.No.D.Tensor


class SquareAuxiliariesSystem:
    __slots__ = ["_M", "_N", "_dimension_cut", "_lattice", "_auxiliaries"]

    @multimethod
    def __init__(self, M: int, N: int, Dc: int) -> None:
        self._M: int = M
        self._N: int = N
        self._dimension_cut: int = Dc
        self._lattice: List[List[Optional[Tensor]]] = [[None for _ in range(self._N)] for _ in range(self._M)]
        self._auxiliaries: Dict[Tuple[str, int, int], Tensor] = {}

    @multimethod
    def __init__(self, other: SquareAuxiliariesSystem) -> None:
        self._M: int = other._M
        self._N: int = other._N
        self._dimension_cut: int = other._dimension_cut
        self._lattice: List[List[Optional[Tensor]]] = [[other._lattice[i][j] for j in range(self._N)] for i in range(self._M)]
        self._auxiliaries: Dict[Tuple[str, int, int], Tensor] = other._auxiliaries.copy()

    def __iadd__(self, other: SquareAuxiliariesSystem) -> SquareAuxiliariesSystem:
        if self._M != other._M:
            raise ValueError("Different M when combining two SquareAuxiliariesSystem")
        if self._N != other._N:
            raise ValueError("Different N when combining two SquareAuxiliariesSystem")
        if self._dimension_cut != other._dimension_cut:
            raise ValueError("Different dimension cut when combining two SquareAuxiliariesSystem")
        for i in range(self._M):
            for j in range(self._N):
                if self._lattice[i][j] is not None and other._lattice[i][j] is not None:
                    # 这里不比较是否值一样, 只要不是同一个id, 就会报错
                    if self._lattice[i][j] is not other._lattice[i][j]:
                        raise ValueError("Overlap tensor in lattice when combining two SquareAuxiliariesSystem")
        self._auxiliaries |= other._auxiliaries
        return self

    def __add__(self, other: SquareAuxiliariesSystem) -> SquareAuxiliariesSystem:
        result = SquareAuxiliariesSystem(self)
        result += other
        return result

    def __delitem__(self, position: Tuple[int, int]) -> None:
        i, j = position
        self._refresh_line("right", j)
        self._refresh_line("left", j)
        self._refresh_line("down", i)
        self._refresh_line("up", i)
        for t in range(self._M):
            if t < i:
                self._try_to_delete_auxiliaries("down-to-up-3", t, j)
                self._try_to_delete_auxiliaries("down-to-up-3-1", t, j)
            elif t > i:
                self._try_to_delete_auxiliaries("up-to-down-3", t, j)
                self._try_to_delete_auxiliaries("up-to-down-3-1", t, j)
            else:
                self._try_to_delete_auxiliaries("down-to-up-3", t, j)
                self._try_to_delete_auxiliaries("up-to-down-3", t, j)
                self._try_to_delete_auxiliaries("down-to-up-3-1", t, j)
                self._try_to_delete_auxiliaries("up-to-down-3-1", t, j)
        for t in range(self._N):
            if t < j:
                self._try_to_delete_auxiliaries("right-to-left-3", i, t)
                self._try_to_delete_auxiliaries("right-to-left-3-1", i, t)
            elif t > j:
                self._try_to_delete_auxiliaries("left-to-right-3", i, t)
                self._try_to_delete_auxiliaries("left-to-right-3-1", i, t)
            else:
                self._try_to_delete_auxiliaries("right-to-left-3", i, t)
                self._try_to_delete_auxiliaries("left-to-right-3", i, t)
                self._try_to_delete_auxiliaries("right-to-left-3-1", i, t)
                self._try_to_delete_auxiliaries("left-to-right-3-1", i, t)
        self._lattice[i][j] = None

    def __setitem__(self, position: Tuple[int, int], value: Tensor) -> None:
        self.__delitem__(position)
        i, j = position
        self._lattice[i][j] = value

    def _refresh_line(self, kind: str, index: int) -> None:
        if kind == "right":
            if index != self._N:
                flag = False
                for i in range(self._M):
                    flag = self._try_to_delete_auxiliaries("left-to-right", i, index)
                    self._try_to_delete_auxiliaries("up-to-down-3", i, index + 1)
                    self._try_to_delete_auxiliaries("down-to-up-3", i, index + 1)
                    self._try_to_delete_auxiliaries("up-to-down-3-1", i, index + 1)
                    self._try_to_delete_auxiliaries("down-to-up-3-1", i, index + 1)
                if flag:
                    self._refresh_line(kind, index + 1)
        elif kind == "left":
            if index != -1:
                flag = False
                for i in range(self._M):
                    flag = self._try_to_delete_auxiliaries("right-to-left", i, index)
                    self._try_to_delete_auxiliaries("up-to-down-3", i, index - 1)
                    self._try_to_delete_auxiliaries("down-to-up-3", i, index - 1)
                    self._try_to_delete_auxiliaries("up-to-down-3-1", i, index - 1)
                    self._try_to_delete_auxiliaries("down-to-up-3-1", i, index - 1)
                if flag:
                    self._refresh_line(kind, index - 1)
        elif kind == "down":
            if index != self._M:
                flag = False
                for j in range(self._N):
                    flag = self._try_to_delete_auxiliaries("up-to-down", index, j)
                    self._try_to_delete_auxiliaries("left-to-right-3", index + 1, j)
                    self._try_to_delete_auxiliaries("right-to-left-3", index + 1, j)
                    self._try_to_delete_auxiliaries("left-to-right-3-1", index + 1, j)
                    self._try_to_delete_auxiliaries("right-to-left-3-1", index + 1, j)
                if flag:
                    self._refresh_line(kind, index + 1)
        elif kind == "up":
            if index != -1:
                flag = False
                for j in range(self._N):
                    flag = self._try_to_delete_auxiliaries("down-to-up", index, j)
                    self._try_to_delete_auxiliaries("left-to-right-3", index - 1, j)
                    self._try_to_delete_auxiliaries("right-to-left-3", index - 1, j)
                    self._try_to_delete_auxiliaries("left-to-right-3-1", index - 1, j)
                    self._try_to_delete_auxiliaries("right-to-left-3-1", index - 1, j)
                if flag:
                    self._refresh_line(kind, index + 1)
        else:
            raise ValueError("Wrong Type in Refresh Line")

    def _try_to_delete_auxiliaries(self, index: str, i: int, j: int) -> bool:
        if (index, i, j) in self._auxiliaries:
            del self._auxiliaries[index, i, j]
            return True
        else:
            return False

    # TODO Lazy style?
    def _get_auxiliaries(self, kind: str, i: int, j: int) -> Tensor:
        if (kind, i, j) not in self._auxiliaries:
            if kind == "up-to-down":
                if i == -1:
                    for t in range(self._N):
                        self._auxiliaries[kind, i, t] = Tensor(1)
                elif -1 < i < self._M:
                    line_1 = [self._get_auxiliaries(kind, i - 1, t) for t in range(self._N)]
                    line_2 = [self._lattice[i][t] for t in range(self._N)]
                    result = self._two_line_to_one_line(["U", "D", "L", "R"], line_1, line_2, self._dimension_cut)
                    for t in range(self._N):
                        self._auxiliaries[kind, i, t] = result[t]
                else:
                    raise ValueError("Wrong Auxiliaries Position")
            elif kind == "down-to-up":
                if i == self._M:
                    for t in range(self._N):
                        self._auxiliaries[kind, i, t] = Tensor(1)
                elif -1 < i < self._M:
                    line_1 = [self._get_auxiliaries(kind, i + 1, t) for t in range(self._N)]
                    line_2 = [self._lattice[i][t] for t in range(self._N)]
                    result = self._two_line_to_one_line(["D", "U", "L", "R"], line_1, line_2, self._dimension_cut)
                    for t in range(self._N):
                        self._auxiliaries[kind, i, t] = result[t]
                else:
                    raise ValueError("Wrong Auxiliaries Position")
            elif kind == "left-to-right":
                if j == -1:
                    for t in range(self._M):
                        self._auxiliaries[kind, t, j] = Tensor(1)
                elif -1 < j < self._N:
                    line_1 = [self._get_auxiliaries(kind, t, j - 1) for t in range(self._M)]
                    line_2 = [self._lattice[t][j] for t in range(self._M)]
                    result = self._two_line_to_one_line(["L", "R", "U", "D"], line_1, line_2, self._dimension_cut)
                    for t in range(self._M):
                        self._auxiliaries[kind, t, j] = result[t]
                else:
                    raise ValueError("Wrong Auxiliaries Position")
            elif kind == "right-to-left":
                if j == self._N:
                    for t in range(self._M):
                        self._auxiliaries[kind, t, j] = Tensor(1)
                elif -1 < j < self._N:
                    line_1 = [self._get_auxiliaries(kind, t, j + 1) for t in range(self._M)]
                    line_2 = [self._lattice[t][j] for t in range(self._M)]
                    result = self._two_line_to_one_line(["R", "L", "U", "D"], line_1, line_2, self._dimension_cut)
                    for t in range(self._M):
                        self._auxiliaries[kind, t, j] = result[t]
                else:
                    raise ValueError("Wrong Auxiliaries Position")
            elif kind == "up-to-down-3-1":
                if i == -1:
                    self._auxiliaries[kind, i, j] = Tensor(1)
                elif -1 < i < self._M:
                    """
                       D2 D3
                       |  |
                    D1R-
                    |
                    """
                    self._auxiliaries[kind, i, j] = self._get_auxiliaries("up-to-down-3", i - 1, j) \
                        .contract(self._get_auxiliaries("left-to-right", i, j - 1), {("D1", "U")}).edge_rename({"D": "D1"})
                else:
                    raise ValueError("Wrong Auxiliaries Position In Three Line Type")
            elif kind == "up-to-down-3":
                if i == -1:
                    self._auxiliaries[kind, i, j] = Tensor(1)
                elif -1 < i < self._M:
                    """
                    D1 D2 D3
                    |  |  |
                    """
                    self._auxiliaries[kind, i, j] = self._get_auxiliaries("up-to-down-3-1", i, j) \
                        .contract(self._lattice[i][j], {("D2", "U"), ("R", "L")}).edge_rename({"D": "D2"}) \
                        .contract(self._get_auxiliaries("right-to-left", i, j + 1), {("D3", "U"), ("R", "L")}).edge_rename({"D": "D3"})
                else:
                    raise ValueError("Wrong Auxiliaries Position In Three Line Type")
            elif kind == "down-to-up-3-1":
                if i == self._M:
                    self._auxiliaries[kind, i, j] = Tensor(1)
                elif -1 < i < self._M:
                    """
                          |
                        -LU3
                    |  |
                    U1 U2
                    """
                    self._auxiliaries[kind, i, j] = self._get_auxiliaries("down-to-up-3", i + 1, j) \
                        .contract(self._get_auxiliaries("right-to-left", i, j + 1), {("U3", "D")}).edge_rename({"U": "U3"})
                else:
                    raise ValueError("Wrong Auxiliaries Position In Three Line Type")
            elif kind == "down-to-up-3":
                if i == self._M:
                    self._auxiliaries[kind, i, j] = Tensor(1)
                elif -1 < i < self._M:
                    """
                    |  |  |
                    U1 U2 U3
                    """
                    self._auxiliaries[kind, i, j] = self._get_auxiliaries("down-to-up-3-1", i, j) \
                        .contract(self._lattice[i][j], {("U2", "D"), ("L", "R")}).edge_rename({"U": "U2"}) \
                        .contract(self._get_auxiliaries("left-to-right", i, j - 1), {("U1", "D"), ("L", "R")}).edge_rename({"U": "U1"})
                else:
                    raise ValueError("Wrong Auxiliaries Position In Three Line Type")
            elif kind == "left-to-right-3-1":
                if j == -1:
                    self._auxiliaries[kind, i, j] = Tensor(1)
                elif -1 < j < self._N:
                    """
                         DR1 -
                    R2 - |
                    R3 -
                    """
                    self._auxiliaries[kind, i, j] = self._get_auxiliaries("left-to-right-3", i, j - 1) \
                        .contract(self._get_auxiliaries("up-to-down", i - 1, j), {("R1", "L")}).edge_rename({"R": "R1"})
                else:
                    raise ValueError("Wrong Auxiliaries Position In Three Line Type")
            elif kind == "left-to-right-3":
                if j == -1:
                    self._auxiliaries[kind, i, j] = Tensor(1)
                elif -1 < j < self._N:
                    """
                    R1 -
                    R2 -
                    R3 -
                    """
                    self._auxiliaries[kind, i, j] = self._get_auxiliaries("left-to-right-3-1", i, j) \
                        .contract(self._lattice[i][j], {("R2", "L"), ("D", "U")}).edge_rename({"R": "R2"}) \
                        .contract(self._get_auxiliaries("down-to-up", i + 1, j), {("R3", "L"), ("D", "U")}).edge_rename({"R": "R3"})
                else:
                    raise ValueError("Wrong Auxiliaries Position In Three Line Type")
            elif kind == "right-to-left-3-1":
                if j == self._N:
                    self._auxiliaries[kind, i, j] = Tensor(1)
                elif -1 < j < self._N:
                    """
                          - L1
                        | - L2
                    - L3U
                    """
                    self._auxiliaries[kind, i, j] = self._get_auxiliaries("right-to-left-3", i, j + 1) \
                        .contract(self._get_auxiliaries("down-to-up", i + 1, j), {("L3", "R")}).edge_rename({"L": "L3"})
                else:
                    raise ValueError("Wrong Auxiliaries Position In Three Line Type")
            elif kind == "right-to-left-3":
                if j == self._N:
                    self._auxiliaries[kind, i, j] = Tensor(1)
                elif -1 < j < self._N:
                    """
                    - L1
                    - L2
                    - L3
                    """
                    self._auxiliaries[kind, i, j] = self._get_auxiliaries("right-to-left-3-1", i, j) \
                        .contract(self._lattice[i][j], {("L2", "R"), ("U", "D")}).edge_rename({"L": "L2"}) \
                        .contract(self._get_auxiliaries("up-to-down", i - 1, j), {("L1", "R"), ("U", "D")}).edge_rename({"L": "L1"})
                else:
                    raise ValueError("Wrong Auxiliaries Position In Three Line Type")
            else:
                raise ValueError("Wrong Auxiliaries Kind")
        return self._auxiliaries[kind, i, j]

    @multimethod
    def __getitem__(self, _: None) -> Tensor:
        return self._get_auxiliaries("left-to-right-3", self._M - 1, self._N - 1)

    @multimethod
    def __getitem__(self, positions: Tuple[Tuple[int, int], ...]) -> Tensor:
        if len(positions) == 0:
            return self._get_auxiliaries("left-to-right-3", self._M - 1, self._N - 1)
        if len(positions) == 1:
            i, j = positions[0]
            return self._get_auxiliaries("left-to-right-3-1", i, j) \
                .contract(self._get_auxiliaries("right-to-left-3-1", i, j ), {("R1", "L1"), ("R3", "L3")}) \
                .edge_rename({"R2": "L0", "L2": "R0", "U": "D0", "D": "U0"})
        if len(positions) == 2:
            x1, y1 = positions[0]
            x2, y2 = positions[1]
            if x1 == x2:
                if y1 + 1 == y2:
                    return self._get_auxiliaries("left-to-right-3-1", x1, y1) \
                        .contract(self._get_auxiliaries("down-to-up", x1 + 1, y1).edge_rename({"R": "R3"}), {("R3", "L")}) \
                        .edge_rename({"D": "U0", "U": "D0"}) \
                        .contract(self._get_auxiliaries("up-to-down", x2 - 1, y2).edge_rename({"R": "R1"}), {("R1", "L")}) \
                        .edge_rename({"D": "U1"}) \
                        .contract(self._get_auxiliaries("right-to-left-3-1", x2, y2), {("R1", "L1"), ("R3", "L3")}) \
                        .edge_rename({"R2": "L0", "L2": "R1", "U": "D1"})
                if y2 + 1 == y1:
                    return self._get_auxiliaries("left-to-right-3-1", x2, y2) \
                        .contract(self._get_auxiliaries("down-to-up", x2 + 1, y2).edge_rename({"R": "R3"}), {("R3", "L")}) \
                        .edge_rename({"D": "U1", "U": "D1"}) \
                        .contract(self._get_auxiliaries("up-to-down", x1 - 1, y1).edge_rename({"R": "R1"}), {("R1", "L")}) \
                        .edge_rename({"D": "U0"}) \
                        .contract(self._get_auxiliaries("right-to-left-3-1", x1, y1), {("R1", "L1"), ("R3", "L3")}) \
                        .edge_rename({"R2": "L1", "L2": "R0", "U": "D0"})
            if y1 == y2:
                if x1 + 1 == x2:
                    return self._get_auxiliaries("up-to-down-3-1", x1, y1) \
                        .contract(self._get_auxiliaries("right-to-left", x1, y1 + 1).edge_rename({"D": "D3"}), {("D3", "U")}) \
                        .edge_rename({"R": "L0", "L": "R0"}) \
                        .contract(self._get_auxiliaries("left-to-right", x2, y2 - 1).edge_rename({"D": "D1"}), {("D1", "U")}) \
                        .edge_rename({"R": "L1"}) \
                        .contract(self._get_auxiliaries("down-to-up-3-1", x2, y2), {("D1", "U1"), ("D3", "U3")}) \
                        .edge_rename({"D2": "U0", "U2": "D1", "L": "R1"})
                if x2 + 1 == x1:
                    return self._get_auxiliaries("up-to-down-3-1", x2, y2) \
                        .contract(self._get_auxiliaries("right-to-left", x2, y2 + 1).edge_rename({"D": "D3"}), {("D3", "U")}) \
                        .edge_rename({"R": "L1", "L": "R1"}) \
                        .contract(self._get_auxiliaries("left-to-right", x1, y1 - 1).edge_rename({"D": "D1"}), {("D1", "U")}) \
                        .edge_rename({"R": "L0"}) \
                        .contract(self._get_auxiliaries("down-to-up-3-1", x1, y1), {("D1", "U1"), ("D3", "U3")}) \
                        .edge_rename({"D2": "U1", "U2": "D0", "L": "R0"})
        raise NotImplementedError("Unsupported getitem style")

    @multimethod
    def __getitem__(self, replacement: Dict[Tuple[int, int], Tensor]) -> Tensor:
        positions = [position for position in replacement]
        new_tensors = [replacement[position] for position in positions]
        if len(replacement) == 0:
            return self._get_auxiliaries("left-to-right-3", self._M - 1, self._N - 1)
        if len(replacement) == 1:
            i, j = positions[0]
            new_tensor = new_tensors[0]
            return self._get_auxiliaries("left-to-right-3-1", i, j) \
                .contract(new_tensor.edge_rename({"R": "R2"}), {("R2", "L"), ("D", "U")}) \
                .contract(self._get_auxiliaries("right-to-left-3-1", i, j), {("R1", "L1"), ("R2", "L2"), ("R3", "L3"), ("D", "U")})
        if len(replacement) == 2:
            x1, y1 = positions[0]
            x2, y2 = positions[1]
            new_tensor_1 = new_tensors[0]
            new_tensor_2 = new_tensors[1]
            if x1 == x2:
                if y1 + 1 == y2:
                    return self._get_auxiliaries("left-to-right-3-1", x1, y1) \
                        .contract(new_tensor_1.edge_rename({"R": "R2"}), {("R2", "L"), ("D", "U")}) \
                        .contract(self._get_auxiliaries("down-to-up", x1 + 1, y1).edge_rename({"R": "R3"}), {("R3", "L"), ("D", "U")}) \
                        .contract(self._get_auxiliaries("up-to-down", x2 - 1, y2).edge_rename({"R": "R1"}), {("R1", "L")}) \
                        .contract(new_tensor_2.edge_rename({"R": "R2"}), {("R2", "L"), ("D", "U")}) \
                        .contract(self._get_auxiliaries("right-to-left-3-1", x2, y2), {("R1", "L1"), ("R2", "L2"), ("R3", "L3"), ("D", "U")})
                if y2 + 1 == y1:
                    return self._get_auxiliaries("left-to-right-3-1", x2, y2) \
                        .contract(new_tensor_2.edge_rename({"R": "R2"}), {("R2", "L"), ("D", "U")}) \
                        .contract(self._get_auxiliaries("down-to-up", x2 + 1, y2).edge_rename({"R": "R3"}), {("R3", "L"), ("D", "U")}) \
                        .contract(self._get_auxiliaries("up-to-down", x1 - 1, y1).edge_rename({"R": "R1"}), {("R1", "L")}) \
                        .contract(new_tensor_1.edge_rename({"R": "R2"}), {("R2", "L"), ("D", "U")}) \
                        .contract(self._get_auxiliaries("right-to-left-3-1", x1, y1), {("R1", "L1"), ("R2", "L2"), ("R3", "L3"), ("D", "U")})
            if y1 == y2:
                if x1 + 1 == x2:
                    return self._get_auxiliaries("up-to-down-3-1", x1, y1) \
                        .contract(new_tensor_1.edge_rename({"D": "D2"}), {("D2", "U"), ("R", "L")}) \
                        .contract(self._get_auxiliaries("right-to-left", x1, y1 + 1).edge_rename({"D": "D3"}), {("D3", "U"), ("R", "L")}) \
                        .contract(self._get_auxiliaries("left-to-right", x2, y2 - 1).edge_rename({"D": "D1"}), {("D1", "U")}) \
                        .contract(new_tensor_2.edge_rename({"D": "D2"}), {("D2", "U"), ("R", "L")}) \
                        .contract(self._get_auxiliaries("down-to-up-3-1", x2, y2), {("D1", "U1"), ("D2", "U2"), ("D3", "U3"), ("R", "L")})
                if x2 + 1 == x1:
                    return self._get_auxiliaries("up-to-down-3-1", x2, y2) \
                        .contract(new_tensor_2.edge_rename({"D": "D2"}), {("D2", "U"), ("R", "L")}) \
                        .contract(self._get_auxiliaries("right-to-left", x2, y2 + 1).edge_rename({"D": "D3"}), {("D3", "U"), ("R", "L")}) \
                        .contract(self._get_auxiliaries("left-to-right", x1, y1 - 1).edge_rename({"D": "D1"}), {("D1", "U")}) \
                        .contract(new_tensor_1.edge_rename({"D": "D2"}), {("D2", "U"), ("R", "L")}) \
                        .contract(self._get_auxiliaries("down-to-up-3-1", x1, y1), {("D1", "U1"), ("D2", "U2"), ("D3", "U3"), ("R", "L")})
        raise NotImplementedError("Unsupported getitem style")

    @staticmethod
    def _two_line_to_one_line(udlr_name: List[str], line_1: List[Tensor], line_2: List[Tensor], cut: int) -> List[Tensor]:
        [up, down, left, right] = udlr_name
        up1 = up + "1"
        up2 = up + "2"
        down1 = down + "1"
        down2 = down + "2"
        left1 = left + "1"
        left2 = left + "2"
        right1 = right + "1"
        right2 = right + "2"

        length = len(line_1)
        if len(line_1) != len(line_2):
            raise ValueError("Different Length in Two Line to One Line")
        double_line = []
        for i in range(length):
            double_line.append(line_1[i].edge_rename({left: left1, right: right1}).contract(line_2[i].edge_rename({left: left2, right: right2}), {(down, up)}))

        for i in range(length - 1):
            # 虽然实际上是range(length - 2), 但是多计算一个以免角标merge的麻烦
            q, r = double_line[i].qr("R", {right1, right2}, right, left)
            double_line[i] = q
            double_line[i + 1] = double_line[i + 1].contract(r, {(left1, right1), (left2, right2)})

        for i in reversed(range(length - 1)):
            # 可能上下都有
            [u, s, v] = double_line[i].edge_rename({up: up1, down: down1}) \
                .contract(double_line[i + 1].edge_rename({up: up2, down: down2}), {(right, left)}) \
                .svd({left, up1, down1}, right, left, cut)
            double_line[i + 1] = v.edge_rename({up2: up, down2: down})
            double_line[i] = u.multiple(s, right, "u").edge_rename({up1: up, down1: down})

        return double_line
