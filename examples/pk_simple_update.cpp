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

#define TAT_USE_SIMPLE_NAME
#define TAT_USE_EASY_CONVERSION
#include <TAT/TAT.hpp>
#include <random>

using Tensor = TAT::Tensor<>;
using Singular = TAT::Singular<>;
enum struct Direction
{
   Right,
   Down
};

auto random_engine = std::default_random_engine(std::random_device()());
auto distribution = std::normal_distribution<double>(0, 1);

auto hamiltonian = Tensor({"I0", "I1", "O0", "O1"}, {2, 2, 2, 2}).set([]() {
   static int i = 0;
   static double data[] = {1. / 4, 0, 0, 0, 0, -1. / 4, 2. / 4, 0, 0, 2. / 4, -1. / 4, 0, 0, 0, 0, 1. / 4};
   return data[i++];
});
auto identity = Tensor({"I0", "I1", "O0", "O1"}, {2, 2, 2, 2}).set([]() {
   static int i = 0;
   static double data[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
   return data[i++];
});

struct TwoDimensionHeisenberg {
   int L1;
   int L2;
   int D;
   std::map<std::tuple<int, int>, Tensor> lattice;
   std::map<std::tuple<int, int, Direction>, Singular> environment;

   TwoDimensionHeisenberg(int L1, int L2, int D) : L1(L1), L2(L2), D(D) {
      for (auto i = 0; i < L1; i++) {
         for (auto j = 0; j < L2; j++) {
            lattice[{i, j}] = _create_tensor(i, j);
         }
      }
   }

   void simple_update(int step, double delta_t) {
      auto updater = identity - delta_t * hamiltonian;
      for (auto i = 0; i < step; i++) {
         _single_step_simple_update(updater);
         std::cout << i << "\n";
         // show();
      }
   }

   void show() const {
      for (const auto& [position, tensor] : lattice) {
         const auto& [i, j] = position;
         std::cout << "{" << i << ", " << j << "}: " << tensor << "\n";
      }
   }

   Tensor _create_tensor(int l1, int l2) const {
      auto name_list = std::vector<std::string>();
      if (l1 != 0) {
         name_list.push_back("Up");
      }
      if (l2 != 0) {
         name_list.push_back("Left");
      }
      if (l1 != L1 - 1) {
         name_list.push_back("Down");
      }
      if (l2 != L2 - 1) {
         name_list.push_back("Right");
      }
      auto dimension_list = std::vector<TAT::Edge<TAT::NoSymmetry>>(name_list.size(), D);
      name_list.push_back("Phy");
      dimension_list.push_back(2);
      auto result = Tensor(name_list, dimension_list);
      result.set([]() {
         return distribution(random_engine);
      });
      return result;
   }

   void _single_step_simple_update(const Tensor& updater) {
      // LR
      for (auto i = 0; i < L1; i++) {
         for (auto j = 0; j < L2 - 1; j++) {
            _single_term_simple_update(updater, Direction::Right, i, j);
         }
      }
      // UD
      for (auto j = 0; j < L2; j++) {
         for (auto i = 0; i < L1 - 1; i++) {
            _single_term_simple_update(updater, Direction::Down, i, j);
         }
      }
      // DU
      for (auto j = L2; j-- > 0;) {
         for (auto i = L1 - 1; i-- > 0;) {
            _single_term_simple_update(updater, Direction::Down, i, j);
         }
      }
      // RL
      for (auto i = L1; i-- > 0;) {
         for (auto j = L2 - 1; j-- > 0;) {
            _single_term_simple_update(updater, Direction::Right, i, j);
         }
      }
   }

   void _try_multiple(Tensor& tensor, int l1, int l2, const std::string& direction, bool division = false) const {
      if (direction == "Left") {
         if (auto found = environment.find({l1, l2 - 1, Direction::Right}); found != environment.end()) {
            tensor = tensor.multiple(found->second, "Left", 'v', division);
         }
      } else if (direction == "Up") {
         if (auto found = environment.find({l1 - 1, l2, Direction::Down}); found != environment.end()) {
            tensor = tensor.multiple(found->second, "Up", 'v', division);
         }
      } else if (direction == "Down") {
         if (auto found = environment.find({l1, l2, Direction::Down}); found != environment.end()) {
            tensor = tensor.multiple(found->second, "Down", 'u', division);
         }
      } else {
         if (auto found = environment.find({l1, l2, Direction::Right}); found != environment.end()) {
            tensor = tensor.multiple(found->second, "Right", 'u', division);
         }
      }
   }

   void _single_term_simple_update(const Tensor& updater, Direction direction, int l1, int l2) {
      if (direction == Direction::Right) {
         auto left = lattice[{l1, l2}].edge_rename({});
         _try_multiple(left, l1, l2, "Left");
         _try_multiple(left, l1, l2, "Up");
         _try_multiple(left, l1, l2, "Down");
         _try_multiple(left, l1, l2, "Right");
         auto right = lattice[{l1, l2 + 1}].edge_rename({});
         _try_multiple(right, l1, l2 + 1, "Up");
         _try_multiple(right, l1, l2 + 1, "Down");
         _try_multiple(right, l1, l2 + 1, "Right");
         auto [left_q, left_r] = left.qr('r', {"Phy", "Right"}, "Right", "Left");
         auto [right_q, right_r] = right.qr('r', {"Phy", "Left"}, "Left", "Right");
         auto [u, s, v] = left_r.edge_rename({{"Phy", "Phy0"}})
                                .contract(right_r.edge_rename({{"Phy", "Phy1"}}), {{"Right", "Left"}})
                                .contract(updater, {{"Phy0", "I0"}, {"Phy1", "I1"}})
                                .svd({"Left", "O0"}, "Right", "Left", D);
         u /= u.norm<-1>();
         s /= s.norm<-1>();
         v /= v.norm<-1>();
         environment[{l1, l2, Direction::Right}] = std::move(s);
         u = u.contract(left_q, {{"Left", "Right"}}).edge_rename({{"O0", "Phy"}});
         _try_multiple(u, l1, l2, "Left", true);
         _try_multiple(u, l1, l2, "Up", true);
         _try_multiple(u, l1, l2, "Down", true);
         lattice[{l1, l2}] = std::move(u);
         v = v.contract(right_q, {{"Right", "Left"}}).edge_rename({{"O1", "Phy"}});
         _try_multiple(v, l1, l2 + 1, "Up", true);
         _try_multiple(v, l1, l2 + 1, "Down", true);
         _try_multiple(v, l1, l2 + 1, "Right", true);
         lattice[{l1, l2 + 1}] = std::move(v);
      } else {
         auto up = lattice[{l1, l2}].edge_rename({});
         _try_multiple(up, l1, l2, "Left");
         _try_multiple(up, l1, l2, "Up");
         _try_multiple(up, l1, l2, "Down");
         _try_multiple(up, l1, l2, "Right");
         auto down = lattice[{l1 + 1, l2}].edge_rename({});
         _try_multiple(down, l1 + 1, l2, "Left");
         _try_multiple(down, l1 + 1, l2, "Down");
         _try_multiple(down, l1 + 1, l2, "Right");
         auto [up_q, up_r] = up.qr('r', {"Phy", "Down"}, "Down", "Up");
         auto [down_q, down_r] = down.qr('r', {"Phy", "Up"}, "Up", "Down");
         auto [u, s, v] = up_r.edge_rename({{"Phy", "Phy0"}})
                                .contract(down_r.edge_rename({{"Phy", "Phy1"}}), {{"Down", "Up"}})
                                .contract(updater, {{"Phy0", "I0"}, {"Phy1", "I1"}})
                                .svd({"Up", "O0"}, "Down", "Up", D);
         u /= u.norm<-1>();
         s /= s.norm<-1>();
         v /= v.norm<-1>();
         environment[{l1, l2, Direction::Down}] = std::move(s);
         u = u.contract(up_q, {{"Up", "Down"}}).edge_rename({{"O0", "Phy"}});
         _try_multiple(u, l1, l2, "Left", true);
         _try_multiple(u, l1, l2, "Up", true);
         _try_multiple(u, l1, l2, "Right", true);
         lattice[{l1, l2}] = std::move(u);
         v = v.contract(down_q, {{"Down", "Up"}}).edge_rename({{"O1", "Phy"}});
         _try_multiple(v, l1 + 1, l2, "Left", true);
         _try_multiple(v, l1 + 1, l2, "Down", true);
         _try_multiple(v, l1 + 1, l2, "Right", true);
         lattice[{l1 + 1, l2}] = std::move(v);
      }
   }

   auto exact_state() const {
      auto psi = Tensor(1);
      for (const auto& [position, tensor] : lattice) {
         auto const& [i, j] = position;
         auto this_tensor = tensor.edge_rename({});
         _try_multiple(this_tensor, i, j, "Down");
         _try_multiple(this_tensor, i, j, "Right");
         this_tensor = this_tensor.edge_rename(
               {{"Up", "Up" + std::to_string(j)},
                {"Down", "Down" + std::to_string(j)},
                {"Left", "Left" + std::to_string(i)},
                {"Right", "Right" + std::to_string(i)},
                {"Phy", std::to_string(i) + "." + std::to_string(j)}});
         psi = psi.contract(
               this_tensor,
               {{"Down" + std::to_string(j), "Up" + std::to_string(j)},
                {"Up" + std::to_string(j), "Down" + std::to_string(j)},
                {"Left" + std::to_string(i), "Right" + std::to_string(i)},
                {"Right" + std::to_string(i), "Left" + std::to_string(i)}});
      }
      return psi;
   }

   void exact_update() const {
      auto psi = exact_state();
      psi /= psi.norm<-1>();
      double approximate_energy = 1;
      for (auto t = 0; t < 100; t++) {
         auto temporary = psi.same_shape().zero();
         for (auto i = 0; i < L1; i++) {
            for (auto j = 0; j < L2 - 1; j++) {
               auto this_term = psi.contract(
                     hamiltonian.edge_rename(
                           {{"O0", std::to_string(i) + "." + std::to_string(j)}, {"O1", std::to_string(i) + "." + std::to_string(j + 1)}}),
                     {{std::to_string(i) + "." + std::to_string(j), "I0"}, {std::to_string(i) + "." + std::to_string(j + 1), "I1"}});
               temporary += this_term;
            }
         }
         for (auto j = 0; j < L2; j++) {
            for (auto i = 0; i < L1 - 1; i++) {
               auto this_term = psi.contract(
                     hamiltonian.edge_rename(
                           {{"O0", std::to_string(i) + "." + std::to_string(j)}, {"O1", std::to_string(i + 1) + "." + std::to_string(j)}}),
                     {{std::to_string(i) + "." + std::to_string(j), "I0"}, {std::to_string(i + 1) + "." + std::to_string(j), "I1"}});
               temporary += this_term;
            }
         }
         psi *= approximate_energy;
         psi -= temporary;
         auto norm_max = double(psi.norm<-1>());
         double energy = approximate_energy - norm_max;
         psi /= norm_max;
         std::cout << energy / (L1 * L2) << "\n";
      }
   }
};

int main() {
   auto lattice = TwoDimensionHeisenberg(20, 20, 16);
   lattice.simple_update(4, 0.1);
   // auto lattice = TwoDimensionHeisenberg(4, 4, 10);
   // lattice.simple_update(1000, 0.1);
   // lattice.simple_update(1000, 0.01);
   // lattice.exact_update();
}
