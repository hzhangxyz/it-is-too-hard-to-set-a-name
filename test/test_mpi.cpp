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

#ifndef TAT_USE_MPI
#error testing mpi but mpi not enabled
#endif
#include <TAT/TAT.hpp>

#include "run_test.hpp"

using Tensor = TAT::Tensor<double, TAT::NoSymmetry>;

void run_test() {
   auto input = Tensor(TAT::mpi.rank);
   auto result = TAT::mpi.reduce(input, TAT::mpi.size / 2, [](auto a, auto b) {
      return a + b;
   });
   TAT::mpi.out_one(TAT::mpi.size / 2) << result << "\n";
   result = TAT::mpi.broadcast(result, TAT::mpi.size / 2);
   TAT::mpi.barrier();
   TAT::mpi.out_rank() << result << "\n";
}
