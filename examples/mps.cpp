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

#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>

#include <TAT/TAT.hpp>

using Tensor = TAT::Tensor<double, TAT::NoSymmetry>;

struct MPS {
   std::vector<Tensor> chain;
   Tensor hamiltonian;
   unsigned int dimension;

   template<typename G>
   MPS(int n, unsigned int d, G&& g, const std::vector<double>& h) : dimension(d) {
      for (int i = 0; i < n; i++) {
         if (i == 0) {
            chain.push_back(Tensor({"Phy", "Right"}, {2, d}).set(g));
         } else if (i == n - 1) {
            chain.push_back(Tensor({"Phy", "Left"}, {2, d}).set(g));
         } else {
            chain.push_back(Tensor({"Phy", "Left", "Right"}, {2, d, d}).set(g));
         }
      }
      hamiltonian = Tensor({"I0", "I1", "O0", "O1"}, {2, 2, 2, 2}).set([&h]() {
         static int i = 0;
         return h[i++];
      });
   }

   void update(int time, double delta_t, int step) {
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
      const auto updater = identity - delta_t * hamiltonian;
      std::cout << "step = -1\nEnergy = " << energy() << '\n';
      show();
      for (int i = 0; i < time; i++) {
         update_once(updater);
         if (i % step == step - 1) {
            std::cout << "step = " << i << "\nEnergy = " << energy() << '\n';
            show();
         }
      }
   }

   void update_once(const Tensor& updater) {
      for (unsigned i = 0; i < chain.size() - 1; i++) {
         // i and i+1
         auto AB = Tensor::contract(chain[i].edge_rename({{"Phy", "PhyA"}}), chain[i + 1].edge_rename({{"Phy", "PhyB"}}), {{"Right", "Left"}});
         auto ABH = Tensor::contract(AB, updater, {{"PhyA", "I0"}, {"PhyB", "I1"}});
         auto [u, s, v] = ABH.svd({"Left", "O0"}, "Right", "Left", dimension);
         chain[i] = std::move(u).edge_rename({{"O0", "Phy"}});
         chain[i + 1] = std::move(v.multiple(s, "Left", 'v')).edge_rename({{"O1", "Phy"}});
         chain[i] = chain[i] / chain[i].norm<-1>();
         chain[i + 1] = chain[i + 1] / chain[i + 1].norm<-1>();
      }
      for (unsigned i = chain.size() - 1; i-- > 0;) {
         // i+1 and i
         auto AB = Tensor::contract(chain[i + 1].edge_rename({{"Phy", "PhyA"}}), chain[i].edge_rename({{"Phy", "PhyB"}}), {{"Left", "Right"}});
         auto ABH = Tensor::contract(AB, updater, {{"PhyA", "I0"}, {"PhyB", "I1"}});
         auto [u, s, v] = ABH.svd({"Right", "O0"}, "Left", "Right", dimension);
         chain[i + 1] = std::move(u).edge_rename({{"O0", "Phy"}});
         chain[i] = std::move(v.multiple(s, "Right", 'v')).edge_rename({{"O1", "Phy"}});
         chain[i + 1] = chain[i + 1] / chain[i + 1].norm<-1>();
         chain[i] = chain[i] / chain[i].norm<-1>();
      }
   }

   double energy() const {
      auto left_pool = std::map<int, Tensor>();
      auto right_pool = std::map<int, Tensor>();
      std::function<Tensor(int)> get_left = [&](int i) {
         auto found = left_pool.find(i);
         if (found != left_pool.end()) {
            return found->second;
         }
         if (i == -1) {
            return Tensor(1);
         }
         return get_left(i - 1)
               .contract(chain[i], {{"Right1", "Left"}})
               .edge_rename({{"Right", "Right1"}})
               .contract(chain[i], {{"Right2", "Left"}, {"Phy", "Phy"}})
               .edge_rename({{"Right", "Right2"}});
      };
      std::function<Tensor(int)> get_right = [&](int i) {
         auto found = right_pool.find(i);
         if (found != right_pool.end()) {
            return found->second;
         }
         if (i == int(chain.size())) {
            return Tensor(1);
         }
         return get_right(i + 1)
               .contract(chain[i], {{"Left1", "Right"}})
               .edge_rename({{"Left", "Left1"}})
               .contract(chain[i], {{"Left2", "Right"}, {"Phy", "Phy"}})
               .edge_rename({{"Left", "Left2"}});
      };
      double energy = 0;
      for (unsigned i = 0; i < chain.size() - 1; i++) {
         energy += double(get_left(i - 1)
                                .contract(chain[i], {{"Right1", "Left"}})
                                .edge_rename({{"Right", "Right1"}, {"Phy", "PhyA"}})
                                .contract(chain[i + 1], {{"Right1", "Left"}})
                                .edge_rename({{"Right", "Right1"}, {"Phy", "PhyB"}})
                                .contract(hamiltonian, {{"PhyA", "I0"}, {"PhyB", "I1"}})
                                .contract(chain[i], {{"Right2", "Left"}, {"O0", "Phy"}})
                                .edge_rename({{"Right", "Right2"}})
                                .contract(chain[i + 1], {{"Right2", "Left"}, {"O1", "Phy"}})
                                .edge_rename({{"Right", "Right2"}})
                                .contract(get_right(i + 2), {{"Right1", "Left1"}, {"Right2", "Left2"}}));
      }
      energy /= double(get_right(0));
      return energy / chain.size();
   }

   void show() const {
      for (const auto& i : chain) {
         std::cout << i << '\n';
      }
   }
};

int main(int argc, char** argv) {
   std::stringstream out;
   auto cout_buf = std::cout.rdbuf();
   if (argc != 6) {
      std::cout.rdbuf(out.rdbuf());
   }

   std::mt19937 engine(0);
   std::uniform_real_distribution<double> dis(-1, 1);
   auto gen = [&]() {
      return dis(engine);
   };
   auto mps = MPS(std::atoi(argv[1]), std::atoi(argv[2]), gen, {1 / 4., 0, 0, 0, 0, -1 / 4., 2 / 4., 0, 0, 2 / 4., -1 / 4., 0, 0, 0, 0, 1 / 4.});
   mps.update(std::atoi(argv[3]), std::atof(argv[4]), std::atoi(argv[5]));

   if (argc != 6) {
      std::cout.rdbuf(cout_buf);
      std::ifstream fout(argv[6]);
      std::string sout((std::istreambuf_iterator<char>(fout)), std::istreambuf_iterator<char>());
      return sout != out.str();
   }
   return 0;
}
