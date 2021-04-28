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
   std::cout << TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range() << "\n";
   std::cout << TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{"Left", "Right"}, {0, 3}} << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{}, {}}.set([]() {
      return 10;
   }) << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{}, {}}
                      .set([]() {
                         return 10;
                      })
                      .at({})
             << "\n";
   std::cout << TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range().at({{"Right", 2}, {"Left", 1}}) << "\n";
}
