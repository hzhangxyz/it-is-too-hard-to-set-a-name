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
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>({"A", "B", "C", "D", "E"}, {2, 3, 2, 3, 4}).range().trace({{"A", "C"}, {"B", "D"}}) << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>({"A", "B", "C"}, {2, 2, 3}).range().trace({{"A", "B"}}) << "\n";
   auto a = TAT::Tensor<double, TAT::NoSymmetry>({"A", "B", "C"}, {4, 3, 5}).range();
   auto b = TAT::Tensor<double, TAT::NoSymmetry>({"D", "E", "F"}, {5, 4, 6}).range();
   std::cout << a.contract(b, {{"A", "E"}, {"C", "D"}}) - a.contract(b, {}).trace({{"A", "E"}, {"C", "D"}});

   do {
      auto c =
            TAT::Tensor<double, TAT::FermiSymmetry>{{"A", "B", "C"}, {{{-1, 1}, {0, 1}, {-2, 1}}, {{0, 1}, {1, 2}}, {{0, 2}, {1, 2}}}, true}.range();
      auto d =
            TAT::Tensor<double, TAT::FermiSymmetry>{{"E", "F", "G"}, {{{0, 2}, {1, 1}}, {{-2, 1}, {-1, 1}, {0, 2}}, {{0, 1}, {-1, 2}}}, true}.range();
      std::cout << c << "\n";
      std::cout << d << "\n";
      std::cout << TAT::Tensor<double, TAT::FermiSymmetry>::contract(c, d, {{"B", "G"}}) << "\n";
      std::cout << TAT::Tensor<double, TAT::FermiSymmetry>::contract(c.transpose({"A", "C", "B"}), d.transpose({"G", "E", "F"}), {{"B", "G"}})
                << "\n";
      std::cout << TAT::Tensor<double, TAT::FermiSymmetry>::contract(c, d, {}).trace({{"B", "G"}}) << "\n";
      std::cout
            << TAT::Tensor<double, TAT::FermiSymmetry>::contract(c.transpose({"A", "C", "B"}), d.transpose({"G", "E", "F"}), {}).trace({{"B", "G"}})
            << "\n";
   } while (false);
   do {
      auto c =
            TAT::Tensor<double, TAT::FermiSymmetry>{{"A", "C", "D"}, {{{-1, 1}, {0, 1}, {-2, 1}}, {{0, 2}, {1, 2}}, {{-2, 2}, {-1, 1}, {0, 2}}}, true}
                  .range();
      auto d =
            TAT::Tensor<double, TAT::FermiSymmetry>{{"E", "F", "H"}, {{{0, 2}, {1, 1}}, {{-2, 1}, {-1, 1}, {0, 2}}, {{0, 2}, {1, 1}, {2, 2}}}, true}
                  .range();
      std::cout << c << "\n";
      std::cout << d << "\n";
      std::cout << TAT::Tensor<double, TAT::FermiSymmetry>::contract(c, d, {{"D", "H"}}) << "\n";
      std::cout << TAT::Tensor<double, TAT::FermiSymmetry>::contract(c.transpose({"A", "C", "D"}), d.transpose({"H", "E", "F"}), {{"D", "H"}})
                << "\n";
      std::cout << TAT::Tensor<double, TAT::FermiSymmetry>::contract(c, d, {}).trace({{"D", "H"}}) << "\n";
      std::cout
            << TAT::Tensor<double, TAT::FermiSymmetry>::contract(c.transpose({"A", "C", "D"}), d.transpose({"H", "E", "F"}), {}).trace({{"D", "H"}})
            << "\n";
   } while (false);
   do {
      auto c =
            TAT::Tensor<double, TAT::U1Symmetry>{
                  {"A", "B", "C", "D"},
                  {{{-1, 1}, {0, 1}, {-2, 1}}, {{0, 1}, {1, 2}}, {{0, 2}, {1, 2}}, {{-2, 2}, {-1, 1}, {0, 2}}},
                  true}
                  .range();
      auto d =
            TAT::Tensor<double, TAT::U1Symmetry>{
                  {"E", "F", "G", "H"},
                  {{{0, 2}, {1, 1}}, {{-2, 1}, {-1, 1}, {0, 2}}, {{0, 1}, {-1, 2}}, {{0, 2}, {1, 1}, {2, 2}}},
                  true}
                  .range();
      std::cout << TAT::Tensor<double, TAT::U1Symmetry>::contract(c, d, {{"B", "G"}, {"D", "H"}}) << "\n";
      std::cout << TAT::Tensor<double, TAT::U1Symmetry>::contract(
                         c.transpose({"A", "C", "B", "D"}),
                         d.transpose({"G", "H", "E", "F"}),
                         {{"B", "G"}, {"D", "H"}})
                << "\n";
      std::cout << TAT::Tensor<double, TAT::U1Symmetry>::contract(c, d, {}).trace({{"B", "G"}}).trace({{"D", "H"}}) << "\n";
      // TODO U1这里也有问题
      std::cout << TAT::Tensor<double, TAT::U1Symmetry>::contract(c, d, {}).trace({{"B", "G"}, {"D", "H"}}) << "\n";
      std::cout << TAT::Tensor<double, TAT::U1Symmetry>::contract(c.transpose({"A", "C", "B", "D"}), d.transpose({"G", "H", "E", "F"}), {})
                         .trace({{"B", "G"}, {"D", "H"}})
                << "\n";
   } while (false);
   do {
      auto c =
            TAT::Tensor<double, TAT::FermiSymmetry>{
                  {"A", "B", "C", "D"},
                  {{{-1, 1}, {0, 1}, {-2, 1}}, {{0, 1}, {1, 2}}, {{0, 2}, {1, 2}}, {{-2, 2}, {-1, 1}, {0, 2}}},
                  true}
                  .range();
      auto d =
            TAT::Tensor<double, TAT::FermiSymmetry>{
                  {"E", "F", "G", "H"},
                  {{{0, 2}, {1, 1}}, {{-2, 1}, {-1, 1}, {0, 2}}, {{0, 1}, {-1, 2}}, {{0, 2}, {1, 1}, {2, 2}}},
                  true}
                  .range();
      std::cout << TAT::Tensor<double, TAT::FermiSymmetry>::contract(c, d, {{"B", "G"}, {"D", "H"}}) << "\n";
      std::cout << TAT::Tensor<double, TAT::FermiSymmetry>::contract(
                         c.transpose({"A", "C", "B", "D"}),
                         d.transpose({"G", "H", "E", "F"}),
                         {{"B", "G"}, {"D", "H"}})
                << "\n";
      std::cout << TAT::Tensor<double, TAT::FermiSymmetry>::contract(c, d, {}).trace({{"B", "G"}}).trace({{"D", "H"}}) << "\n";
      std::cout << TAT::Tensor<double, TAT::FermiSymmetry>::contract(c, d, {}).trace({{"B", "G"}, {"D", "H"}}) << "\n";
      std::cout << TAT::Tensor<double, TAT::FermiSymmetry>::contract(c.transpose({"A", "C", "B", "D"}), d.transpose({"G", "H", "E", "F"}), {})
                         .trace({{"B", "G"}, {"D", "H"}})
                << "\n";
      // TODO 这里问题有点大
   } while (false);
}
