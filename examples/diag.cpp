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

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include <TAT/TAT.hpp>

using Tensor = TAT::Tensor<double, TAT::NoSymmetry>;

struct lattice {
   Tensor state;
   std::vector<std::pair<int, int>> link;
   Tensor hamiltonian;

   lattice& create_vector(int n) & {
      auto name = std::vector<TAT::DefaultName>();
      auto edge = std::vector<TAT::Edge<TAT::NoSymmetry>>();
      for (auto i = 0; i < n; i++) {
         name.emplace_back(std::to_string(i));
         edge.emplace_back(2);
      }
      state = Tensor(std::move(name), std::move(edge));
      return *this;
   }
   lattice create_vector(int n) && {
      return std::move(create_vector(n));
   }

   lattice& set_link(std::vector<std::pair<int, int>>&& l) & {
      link = std::move(l);
      return *this;
   }
   lattice set_link(std::vector<std::pair<int, int>>&& l) && {
      return std::move(set_link(std::move(l)));
   }

   lattice& set_h(const std::vector<double>& h) & {
      hamiltonian = Tensor({"I0", "I1", "O0", "O1"}, {2, 2, 2, 2}).set([&h]() {
         static int i = 0;
         return h[i++];
      });
      return *this;
   }
   lattice set_h(const std::vector<double>& h) && {
      return std::move(set_h(h));
   }

   template<typename G>
   lattice& set_state(G&& g) & {
      state.set(std::forward<G>(g));
      return *this;
   }
   template<typename G>
   lattice set_state(G&& g) && {
      state.set(std::forward<G>(g));
      return std::move(*this);
   }

   void update(double delta_t, int n) {
      static const auto identity = Tensor(hamiltonian.names, hamiltonian.core->edges).set([]() {
         // clang-format off
         static const std::vector<double> id = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1};
         // clang-format on
         static int i = 0;
         return id[i++];
      });
      auto name = state.names;
      const auto updater = identity - delta_t * hamiltonian;
      for (auto t = 0; t < n; t++) {
         for (const auto& [i, j] : link) {
            state = Tensor::contract(state, updater, {{std::to_string(i), "I0"}, {std::to_string(j), "I1"}})
                          .edge_rename({{"O0", std::to_string(i)}, {"O1", std::to_string(j)}});
            state = state / state.norm<-1>();
         }
      }
      state.transform([](double i) {
         if (std::abs(i) > 1e-10) {
            return i;
         } else {
            return double(0);
         }
      });
      std::cout << energy() << '\t' << state << "\n";
   }

   double energy() {
      const auto psi_psi = state.contract_all_edge();
      auto H_psi = state.same_shape().zero();
      for (const auto& [i, j] : link) {
         H_psi += Tensor::contract(state, hamiltonian, {{std::to_string(i), "I0"}, {std::to_string(j), "I1"}})
                        .edge_rename({{"O0", std::to_string(i)}, {"O1", std::to_string(j)}});
      }
      const auto psi_H_psi = state.contract_all_edge(H_psi);
      return double(psi_H_psi) / double(psi_psi);
   }
};

int main(int argc, char** argv) {
   std::stringstream out;
   auto cout_buf = std::cout.rdbuf();
   if (argc != 1) {
      std::cout.rdbuf(out.rdbuf());
   }

   std::mt19937 engine(0);
   std::uniform_real_distribution<double> dis(-1, 1);
   auto gen = [&]() {
      return dis(engine);
   };
   auto l = lattice()
                  .create_vector(4)
                  .set_link({{0, 1}, {1, 2}, {2, 3}})
                  // clang-format off
                  .set_h({1/4., 0,     0,     0,
                          0,    -1/4., 2/4.,  0,
                          0,    2/4.,  -1/4., 0,
                          0,    0,     0,     1/4.})
                  // clang-format on
                  .set_state(gen);
   l.update(0.5, 100);
   l.update(0.2, 100);
   l.update(0.1, 100);
   l.update(0.05, 100);
   l.update(0.02, 100);
   l.update(0.01, 100);
   l.update(0.005, 100);

   if (argc != 1) {
      std::cout.rdbuf(cout_buf);
      std::ifstream fout(argv[1]);
      std::string sout((std::istreambuf_iterator<char>(fout)), std::istreambuf_iterator<char>());
      return sout != out.str();
   }
   return 0;
}
