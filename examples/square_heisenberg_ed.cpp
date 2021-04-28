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
#include <cstdio>
#include <fire.hpp>
#include <random>

using Tensor = TAT::Tensor<double>;

auto Sz = TAT::Tensor<std::complex<double>>({"I", "O"}, {2, 2}).set([]() {
   static int i = 0;
   static std::complex<double> data[4] = {1, 0, 0, -1};
   return data[i++] / 2.;
});
auto Sx = TAT::Tensor<std::complex<double>>({"I", "O"}, {2, 2}).set([]() {
   static int i = 0;
   static std::complex<double> data[4] = {0, 1, 1, 0};
   return data[i++] / 2.;
});
auto Sy = TAT::Tensor<std::complex<double>>({"I", "O"}, {2, 2}).set([]() {
   static int i = 0;
   static std::complex<double> data[4] = {0, {0, 1}, {0, -1}, 0};
   return data[i++] / 2.;
});
auto SzSz = Sz.edge_rename({{"I", "I1"}, {"O", "O1"}})
                  .contract_all_edge(Sz.edge_rename({{"I", "I2"}, {"O", "O2"}}))
                  .transpose({"I1", "I2", "O1", "O2"})
                  .to<double>();
auto SxSx = Sx.edge_rename({{"I", "I1"}, {"O", "O1"}})
                  .contract_all_edge(Sx.edge_rename({{"I", "I2"}, {"O", "O2"}}))
                  .transpose({"I1", "I2", "O1", "O2"})
                  .to<double>();
auto SySy = Sy.edge_rename({{"I", "I1"}, {"O", "O1"}})
                  .contract_all_edge(Sy.edge_rename({{"I", "I2"}, {"O", "O2"}}))
                  .transpose({"I1", "I2", "O1", "O2"})
                  .to<double>();
auto SS = SxSx + SySy + SzSz;

auto random_engine = std::default_random_engine(std::random_device()());

struct SpinLattice {
   Tensor state_vector;
   std::vector<Tensor> bonds;
   double energy;
   double approximate_energy;

   SpinLattice(const std::vector<std::string>& node_names, double approximate_energy = 0) : approximate_energy(std::abs(approximate_energy)) {
      auto edge_to_initial = std::vector<int>(node_names.size(), 2);
      auto dist = std::normal_distribution<double>(0, 1);
      state_vector = Tensor(node_names, edge_to_initial).set([&]() {
         return dist(random_engine);
      });
   }

   void set_bond(const std::string& n1, const std::string& n2, const Tensor& matrix) {
      bonds.push_back(matrix.edge_rename({{"I1", n1}, {"I2", n2}, {"O1", "_" + n1}, {"O2", "_" + n2}}));
   }

   void update() {
      auto norm_max = double(state_vector.norm<-1>());
      energy = approximate_energy - norm_max;
      state_vector /= norm_max;
      auto state_vector_temporary = state_vector.same_shape().zero();
      for (const auto& bond : bonds) {
         const auto& name = bond.names;
         auto this_term = state_vector.contract_all_edge(bond).edge_rename({{name[2], name[0]}, {name[3], name[1]}});
         state_vector_temporary += this_term;
      }
      state_vector *= approximate_energy;
      state_vector -= state_vector_temporary;
   }
};

struct SquareSpinLattice : SpinLattice {
   int n1;
   int n2;

   static auto get_node_names(int n1, int n2) {
      auto result = std::vector<std::string>();
      for (auto i = 0; i < n1; i++) {
         for (auto j = 0; j < n2; j++) {
            result.push_back(std::to_string(i) + "." + std::to_string(j));
         }
      }
      return result;
   }

   SquareSpinLattice(int n1, int n2, double approximate_energy = 0) : SpinLattice(get_node_names(n1, n2), approximate_energy), n1(n1), n2(n2) {}

   void set_bond(const std::tuple<int, int>& p1, const std::tuple<int, int>& p2, const Tensor& matrix) {
      std::string n1 = std::to_string(std::get<0>(p1)) + "." + std::to_string(std::get<1>(p1));
      std::string n2 = std::to_string(std::get<0>(p2)) + "." + std::to_string(std::get<1>(p2));
      SpinLattice::set_bond(n1, n2, matrix);
   }
};

int fired_main(
      int n1 = fire::arg({"-M", "system size"}, 4),
      int n2 = fire::arg({"-N", "system size"}, 4),
      int step = fire::arg({"-S", "step to run"}, 1000),
      bool print_energy = fire::arg({"-P"}),
      int seed = fire::arg({"-R", "random seed"}, 0)) {
   std::ios::sync_with_stdio(false);
   random_engine.seed(seed);
   auto lattice = SquareSpinLattice(n1, n2, 1);
   for (auto i = 0; i < n1 - 1; i++) {
      for (auto j = 0; j < n2; j++) {
         lattice.set_bond({i, j}, {i + 1, j}, SS);
      }
   }
   for (auto i = 0; i < n1; i++) {
      for (auto j = 0; j < n2 - 1; j++) {
         lattice.set_bond({i, j}, {i, j + 1}, SS);
      }
   }
   for (auto t = 0; t < step; t++) {
      lattice.update();
      if (print_energy) {
         std::printf("%.20f\n", lattice.energy / (n1 * n2));
      }
   }
   return 0;
}

FIRE(fired_main)
