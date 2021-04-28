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
   const auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {2, 3}}.set([]() {
      static double i = -1;
      return i += 1;
   });
   auto b = a.merge_edge({{"Merged", {"Left", "Right"}}});
   auto c = a.merge_edge({{"Merged", {"Right", "Left"}}});
   auto d = c.split_edge({{"Merged", {{"1", 3}, {"2", 2}}}});
   std::cout << a << "\n";
   std::cout << b << "\n";
   std::cout << c << "\n";
   std::cout << d << "\n";
   auto e =
         TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>{
               {"Left", "Right", "Up"},
               {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
               .set([]() {
                  static double i = 0;
                  return i += 1;
               });
   std::cout << e << "\n";
   auto f = e.merge_edge({{"Merged", {"Left", "Up"}}});
   std::cout << f << "\n";
   auto g = f.split_edge({{"Merged", {{"Left", {{{-1, 3}, {0, 1}, {1, 2}}}}, {"Up", {{{-1, 2}, {0, 3}, {1, 1}}}}}}});
   std::cout << g << "\n";
   auto h = g.transpose({"Left", "Right", "Up"});
   std::cout << h << "\n";
}
