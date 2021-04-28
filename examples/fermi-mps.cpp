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

#include <functional>
#include <random>

#include <TAT/TAT.hpp>

using Tensor = TAT::Tensor<double, TAT::FermiSymmetry>;

// 无自旋
// H = t c_i^+ c_{i+1} + h.c.

struct MPS {
   std::vector<Tensor> chain;
   Tensor hamiltonian;
   int dimension;

   template<typename G>
   MPS(int n, int d, G&& g) : dimension(d) {
      for (int i = 0; i < n; i++) {
         if (i == 0) {
            chain.push_back(Tensor({"Total", "Phy", "Right"}, {{-1}, {0, 1}, {1, 0}}, true).set(g));
            //}
            // if (i == 1) {
            //   chain.push_back(Tensor({"Left", "Phy", "Right"}, {{{-2, 1}, {-1, 1}}, {{0, 1}, {1, 1}}, {{2, 1}, {1, 2}, {0, 1}}}, true).set(g));
            //} else if (i == 2) {
            //   chain.push_back(Tensor({"Left", "Phy", "Right"}, {{{-2, 1}, {-1, 2}, {0, 1}}, {{0, 1}, {1, 1}}, {{1, 1}, {0, 1}}}, true).set(g));
         } else {
            chain.push_back(Tensor({"Left", "Phy"}, {{-1, 0}, {0, 1}}, true).set(g));
         }
      }
      hamiltonian = Tensor({"I0", "I1", "O0", "O1"}, {{{0, 1}, {-1, 1}}, {{0, 1}, {-1, 1}}, {{0, 1}, {1, 1}}, {{0, 1}, {1, 1}}}, true).zero();
      hamiltonian.block({{"I0", 0}, {"O0", 1}, {"I1", -1}, {"O1", 0}})[0] = 1;
      hamiltonian.block({{"I1", 0}, {"O1", 1}, {"I0", -1}, {"O0", 0}})[0] = 1;
   }

   void update(int time, double delta_t, int step) {
      static const auto identity = [&]() {
         auto res = hamiltonian.same_shape().zero();
         for (TAT::U1 i = 0; i < 2; i++) {
            for (TAT::U1 j = 0; j < 2; j++) {
               res.block({{"I0", -i}, {"O0", i}, {"I1", -j}, {"O1", j}})[0] = 1;
            }
         }
         return res;
      }();
      const auto updater = identity - delta_t * hamiltonian;

      show();
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
      auto conjugate_chain = std::vector<Tensor>();
      for (const auto& tensor : chain) {
         conjugate_chain.push_back(tensor.conjugate());
      }
      conjugate_chain[0] = conjugate_chain[0].edge_rename({{"Total", "Total_Conjugate"}});

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
               .contract(conjugate_chain[i], {{"Right2", "Left"}, {"Phy", "Phy"}})
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
               .contract(conjugate_chain[i], {{"Left2", "Right"}, {"Phy", "Phy"}})
               .edge_rename({{"Left", "Left2"}});
      };

      std::cout << "\nSTART\n";
      for (const auto& i : chain) {
         std::cout << i << "\n";
      }
      for (const auto& i : conjugate_chain) {
         std::cout << i << "\n";
      }
      std::cout << "CONTRACT\n";
      std::cout << get_left(-1) << "\n";
      std::cout << get_left(0) << "\n";
      std::cout << get_left(1) << "\n";
      std::cout << get_right(2) << "\n";
      std::cout << get_right(1) << "\n";
      std::cout << get_right(0) << "\n";
      std::cout << get_left(0).contract(get_right(1), {{"Right1", "Left1"}, {"Right2", "Left2"}}) << "\n";
      std::cout << "END\n\n";

      double energy = 0;
      for (unsigned i = 0; i < chain.size() - 1; i++) {
         energy += double(get_left(i - 1)
                                .contract(chain[i], {{"Right1", "Left"}})
                                .edge_rename({{"Right", "Right1"}, {"Phy", "PhyA"}})
                                .contract(chain[i + 1], {{"Right1", "Left"}})
                                .edge_rename({{"Right", "Right1"}, {"Phy", "PhyB"}})
                                .contract(hamiltonian, {{"PhyA", "I0"}, {"PhyB", "I1"}})
                                .contract(conjugate_chain[i], {{"Right2", "Left"}, {"O0", "Phy"}})
                                .edge_rename({{"Right", "Right2"}})
                                .contract(conjugate_chain[i], {{"Right2", "Left"}, {"O1", "Phy"}})
                                .edge_rename({{"Right", "Right2"}})
                                .contract(get_right(i + 2), {{"Right1", "Left1"}, {"Right2", "Left2"}}));
      }
      energy /= double(get_right(0));
      return energy / chain.size();
      // TODO: problem, write u1 mps first
   }

   void show() const {
      for (const auto& i : chain) {
         std::cout << i << '\n';
      }
   }
};

int main(int argc, char** argv) {
   std::mt19937 engine(0);
   std::uniform_real_distribution<double> dis(-1, 1);
   auto gen = [&]() {
      return dis(engine);
   };
   auto mps = MPS(2, 2, gen);
   // mps.update(std::atoi(argv[3]), std::atof(argv[4]), std::atoi(argv[5]));
   mps.update(10, 0.1, 1);
   return 0;
}
