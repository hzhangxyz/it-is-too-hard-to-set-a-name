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

#include "tools.hpp"

using Tensor = TAT::Tensor<float, TAT::NoSymmetry>;

auto get_name(std::string alphabet) {
   return [=](int i) {
      return alphabet + std::to_string(i);
   };
}
auto left = get_name("left");
auto right = get_name("right");
auto up = get_name("up");
auto down = get_name("down");

struct PBC {
   int L;
   unsigned int D;
   std::map<int, std::map<int, Tensor>> lattice;
   // std::map<std::set<std::tuple<int, int>>, std::map<TAT::NoSymmetry, std::vector<float>>> environment;

   PBC(int L, unsigned int D, std::function<float()> generator) : L(L), D(D) {
      for (int i = 0; i < L; i++) {
         for (int j = 0; j < L; j++) {
            lattice[i][j] = Tensor({"left", "right", "up", "down"}, {D, D, D, D}).set(generator);
         }
      }
   }

   float get_exact() const {
      auto result = Tensor(1);
      for (int i = 0; i < L; i++) {
         for (int j = 0; j < L; j++) {
            auto contract_names = std::set<std::pair<TAT::DefaultName, TAT::DefaultName>>{{right(i), left(i)}, {down(j), up(j)}};
            if (j == L - 1) {
               contract_names.insert({left(i), right(i)});
            }
            if (i == L - 1) {
               contract_names.insert({up(j), down(j)});
            }
            result = result.contract(
                  lattice.at(i).at(j).edge_rename({{"up", up(j)}, {"down", down(j)}, {"left", left(i)}, {"right", right(i)}}),
                  contract_names);
         }
      }
      return float(result);
   }

   float contract_with_two_line_to_one_line(TAT::Size D_cut) const {
      auto up_to_down_aux = std::map<int, std::map<int, Tensor>>();
      for (int j = 0; j < L; j++) {
         up_to_down_aux[0][j] = lattice.at(0).at(j).copy();
      }
      for (int i = 0; i < L - 1; i++) {
         // std::clog << "Dealing with Line " << i << " and Line " << i + 1 << "\n";
         auto line_1 = std::vector<const Tensor*>();
         auto line_2 = std::vector<const Tensor*>();
         for (int j = 0; j < L; j++) {
            line_1.push_back(&up_to_down_aux.at(i).at(j));
            line_2.push_back(&lattice.at(i + 1).at(j));
         }
         auto new_line = tools::two_line_to_one_line({"up", "down", "left", "right"}, line_1, line_2, D_cut);
         // dealing with last and first
         auto [u, s, v] = Tensor::contract(
                                new_line[L - 1].edge_rename({{"up", "up1"}, {"down", "down1"}}),
                                new_line[0].edge_rename({{"up", "up2"}, {"down", "down2"}}),
                                {{"right_1", "left_1"}, {"right_2", "left_2"}})
                                .svd({"up1", "down1", "left"}, "right", "left", D_cut);
         new_line[0] = v.edge_rename({{"up2", "up"}, {"down2", "down"}});
         new_line[L - 1] = u.multiple(s, "right", 'u').edge_rename({{"up1", "up"}, {"down1", "down"}});
         // std::clog << "new line:\n";
         for (int j = 0; j < L; j++) {
            // std::clog << new_line[j] << "\n";
            up_to_down_aux[i + 1][j] = std::move(new_line[j]);
         }
      }
      // std::clog << "end 2to1\n";
      auto result = Tensor(1);
      for (int j = 0; j < L; j++) {
         std::clog << up_to_down_aux[L - 1][j] << "\n";
         result = result.contract(up_to_down_aux[L - 1][j].trace({{"up", "down"}}), {{"right", "left"}});
         // std::clog << result << "\n";
      }
      // std::cout << result << "\n";
      std::clog << result << "\n";
      std::clog << result.trace({{"left", "right"}}) << "\n";
      // TODO here rank 0 issue
      return float(result.trace({{"left", "right"}}));
   }
};

int main() {
   std::mt19937 engine(0);
   std::uniform_real_distribution<double> dis(-1, 1);
   auto generator = [&]() {
      return dis(engine);
   };
   auto pbc = PBC(3, 2, generator);
   std::cout << "Exact : " << pbc.get_exact() << "\n";
   for (TAT::Size cut = 4; cut <= 30; cut++) {
      std::cout << /*cut << "\t: " <<*/ pbc.contract_with_two_line_to_one_line(cut) << "\n";
      return 0;
   }
   return 0;
}
