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

#include "run_test.hpp"
void run_test() {
   std::cout
         << TAT::Tensor<double, TAT::FermiSymmetry>{{"Left", "Right", "Up"}, {{{0, 1}, {1, 2}}, {{-1, 1}, {-2, 3}, {0, 2}}, {{0, 3}, {1, 1}}}, true}
                  .range(2)
         << "\n";
   std::cout
         << TAT::Tensor<
                  double,
                  TAT::FermiU1Symmetry>{{"Left", "Right", "Up"}, {{{{0, 0}, 1}, {{1, 1}, 2}}, {{{-1, -1}, 1}, {{-2, 0}, 3}, {{0, 0}, 2}}, {{{0, 0}, 3}, {{1, -1}, 1}}}, true}
                  .range(2)
         << "\n";
   auto a = TAT::Tensor<double, TAT::FermiU1Symmetry>{
         {"Left", "Right", "Up"},
         {{{{0, 0}, 1}, {{1, 1}, 2}}, {{{-1, -1}, 1}, {{-2, 0}, 3}, {{0, 0}, 2}}, {{{0, 0}, 3}, {{1, -1}, 1}}},
         true};
   std::cout
         << TAT::Tensor<
                  double,
                  TAT::FermiU1Symmetry>{{"Left", "Right", "Up"}, {{{{0, 0}, 1}, {{1, 1}, 2}}, {{{-1, -1}, 1}, {{-2, 0}, 3}, {{0, 0}, 2}}, {{{0, 0}, 3}, {{1, -1}, 1}}}, true}
                  .range(2)
         << "\n";
   std::cout << TAT::Tensor<double, TAT::FermiU1Symmetry>{1234}.at({}) << "\n";
   std::cout
         << TAT::Tensor<
                  double,
                  TAT::FermiU1Symmetry>{{"Left", "Right", "Up"}, {{{{0, 0}, 1}, {{1, 1}, 2}}, {{{-1, -1}, 1}, {{-2, 0}, 3}, {{0, 0}, 2}}, {{{0, 0}, 3}, {{1, -1}, 1}}}, true}
                  .range(2)
                  .at({{"Left", {{1, 1}, 1}}, {"Up", {{1, -1}, 0}}, {"Right", {{-2, 0}, 0}}})
         << "\n";
}
