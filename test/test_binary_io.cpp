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
   std::stringstream ss;
   auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right", "Up"}, {2, 3, 4}}.range();
   ss < a;
   auto b = TAT::Tensor<double, TAT::NoSymmetry>();
   ss > b;
   std::cout << a << "\n";
   std::cout << b << "\n";
   auto c =
         TAT::Tensor<double, TAT::U1Symmetry>{
               {"Left", "Right", "Up"},
               {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
               .range(2);
   ss < c;
   auto d = TAT::Tensor<double, TAT::U1Symmetry>();
   ss > d;
   std::cout << c << "\n";
   std::cout << d << "\n";
   auto e = TAT::Tensor<std::complex<int>, TAT::NoSymmetry>{{"Up", "Left", "Right"}, {1, 2, 3}}.set([]() {
      static int i = 0;
      static int arr[6] = {0x12345, 0x23456, 0x34567, 0x45678, 0x56789, 0x6789a};
      return arr[i++];
   });
   ss < e;
   auto f = TAT::Tensor<std::complex<int>, TAT::NoSymmetry>();
   ss > f;
   std::cout << e << "\n";
   std::cout << f << "\n";
   auto g =
         TAT::Tensor<std::complex<double>, TAT::U1Symmetry>{
               {"Left", "Right", "Up"},
               {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
               .range(2);
   ss < g;
   auto h = TAT::Tensor<std::complex<double>, TAT::U1Symmetry>();
   ss > h;
   std::cout << g << "\n";
   std::cout << h << "\n";
   auto i =
         TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>{
               {"Left", "Right", "Up"},
               {{{-2, 3}, {0, 1}, {-1, 2}}, {{0, 2}, {1, 3}}, {{0, 3}, {1, 1}}},
               true}
               .range(2);
   ss < i;
   auto j = TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>();
   ss > j;
   std::cout << i << "\n";
   std::cout << j << "\n";
}
