/**
 * Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#include <sstream>

#include "run_test.hpp"

void run_test() {
   using namespace TAT;
   Tensor<> a;
   Tensor<double, U1Symmetry> b;
   char c, d;
   std::stringstream(
         "{names:[L.*&^eft,R--..ight],edges:[3,4],blocks:[0,1,2,3,4,5,6,7,8,9,10,11]}?{names:[A,B,C,D],edges:[{-2:1,-1:1,0:1},{0:1,1:2},{0:2,1:2"
         "},{-2:2,-1:1,0:2}],blocks:{[-2,1,1,0]:[0,1,2,3,4,5,6,7],[-1,0,1,0]:[8,9,10,11],[-1,1,0,0]:[12,13,14,15,16,17,18,19],[-1,1,1,-1]:[20,21"
         ",22,23],[0,0,0,0]:[24,25,26,27],[0,0,1,-1]:[28,29],[0,1,0,-1]:[30,31,32,33],[0,1,1,-2]:[34,35,36,37,38,39,40,41]}}*") >>
         a >> c >> b >> d;
   std::cout << a << std::endl;
   std::cout << b << std::endl;
   std::cout << c << d << std::endl;
}
