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

#include "PyTAT.hpp"

namespace TAT {
   void set_random(py::module_& tat_m) {
      auto random_m = tat_m.def_submodule("random", "random for TAT");
      random_m.def("seed", &set_random_seed, "Set Random Seed");
      random_m.def(
            "uniform_int",
            [](int min, int max) {
               return [distribution = std::uniform_int_distribution<int>(min, max)]() mutable {
                  return distribution(random_engine);
               };
            },
            py::arg("min") = 0,
            py::arg("max") = 1,
            "Get random uniform integer");
      random_m.def(
            "uniform_real",
            [](double min, double max) {
               return [distribution = std::uniform_real_distribution<double>(min, max)]() mutable {
                  return distribution(random_engine);
               };
            },
            py::arg("min") = 0,
            py::arg("max") = 1,
            "Get random uniform real");
      random_m.def(
            "normal",
            [](double mean, double stddev) {
               return [distribution = std::normal_distribution<double>(mean, stddev)]() mutable {
                  return distribution(random_engine);
               };
            },
            py::arg("mean") = 0,
            py::arg("stddev") = 1,
            "Get random normal real");
   }
} // namespace TAT
