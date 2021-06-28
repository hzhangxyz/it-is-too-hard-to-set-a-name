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
   auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {2, 3}}.range();
   std::cout << a << "\n";
   std::cout << a.transpose({"Right", "Left"}) << "\n";
   auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right", "Up"}, {2, 3, 4}}.range();
   std::cout << b << "\n";
   std::cout << b.transpose({"Right", "Up", "Left"}) << "\n";
   auto c =
         TAT::Tensor<std::complex<double>, TAT::U1Symmetry>{
               {"Left", "Right", "Up"},
               {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
               .range(1);
   std::cout << c << "\n";
   auto ct = c.transpose({"Right", "Up", "Left"});
   std ::cout << ct << "\n";
   std::cout << c.const_at({{"Left", {-1, 0}}, {"Right", {1, 2}}, {"Up", {0, 0}}}) << "\n";
   std::cout << ct.const_at({{"Left", {-1, 0}}, {"Right", {1, 2}}, {"Up", {0, 0}}}) << "\n";
#define t_edge(...) \
   { {__VA_ARGS__}, true }
#define f_edge(...) \
   { {__VA_ARGS__}, false }
   auto d =
         TAT::Tensor<double, TAT::FermiSymmetry>{
               {"Left", "Right", "Up"},
               {t_edge({-1, 3}, {0, 1}, {1, 2}), t_edge({-1, 1}, {0, 2}, {1, 3}), t_edge({-1, 2}, {0, 3}, {1, 1})}}
               .range(1);
   std::cout << d << "\n";
   auto dt = d.transpose({"Right", "Up", "Left"});
   std::cout << dt << "\n";
   auto e = TAT::Tensor<double, TAT::NoSymmetry>{{"Down", "Up", "Left", "Right"}, {2, 3, 4, 5}}.range(1);
   std::cout << e << "\n";
   auto et = e.transpose({"Left", "Down", "Right", "Up"});
   std::cout << et << "\n";
   std::cout << e.const_at({{"Down", 1}, {"Up", 1}, {"Left", 2}, {"Right", 2}}) << "\n";
   std::cout << et.const_at({{"Down", 1}, {"Up", 1}, {"Left", 2}, {"Right", 2}}) << "\n";
   auto f = TAT::Tensor<double, TAT::NoSymmetry>{{"l1", "l2", "l3"}, {2, 3, 4}}.range();
   std::cout << f << "\n";
   std::cout << f.transpose({"l1", "l2", "l3"}) << "\n";
   std::cout << f.transpose({"l1", "l3", "l2"}) << "\n";
   std::cout << f.transpose({"l2", "l1", "l3"}) << "\n";
   std::cout << f.transpose({"l2", "l3", "l1"}) << "\n";
   std::cout << f.transpose({"l3", "l1", "l2"}) << "\n";
   std::cout << f.transpose({"l3", "l2", "l1"}) << "\n";
}
