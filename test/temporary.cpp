/**
 * Copyright (C) 2019-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <TAT/TAT.hpp>

int main() {
   using Tensor = TAT::Tensor<double, TAT::U1Symmetry>;
   auto A = Tensor({"t", "s0", "s1"}, {{-1}, {0, 1}, {0, 1}}).zero();
   A.core->storage[0] = 1;
   A.core->storage[1] = -1;
   std::cout << A << "\n";
   auto B = A.conjugate();
   std::cout << B << "\n";
   std::cout << A.contract_all_edge(B) << "\n";
}
