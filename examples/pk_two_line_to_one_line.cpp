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
#include <random>
using Tensor = TAT::Tensor<>;

int main() {
   int length = 20;
   unsigned int D = 16;
   unsigned int Dc = 32;

   auto random_engine = std::default_random_engine(std::random_device()());
   auto distribution = std::uniform_real_distribution<double>(0, 1);

   std::vector<Tensor> line_1;
   std::vector<Tensor> line_2;
   for (auto i = 0; i < length; i++) {
      line_1.push_back(Tensor({"Left", "Right", "Down"}, {i == 0 ? 1u : Dc, i == length - 1 ? 1u : Dc, D}).set([&]() {
         return distribution(random_engine);
      }));
      line_2.push_back(Tensor({"Left", "Right", "Down", "Up"}, {i == 0 ? 1u : D, i == length - 1 ? 1u : D, D, D}).set([&]() {
         return distribution(random_engine);
      }));
   }

   for (int _ = 0; _ < 10; _++) {
      std::vector<Tensor> double_line;
      for (auto i = 0; i < length; i++) {
         double_line.push_back(line_1[i]
                                     .edge_rename({{"Left", "Left1"}, {"Right", "Right1"}})
                                     .contract(line_2[i].edge_rename({{"Left", "Left2"}, {"Right", "Right2"}}), {{"Down", "Up"}}));
      }

      for (int i = 0; i < length - 1; i++) {
         // 0&1 .. L-3&L-2, L-2&L-1
         auto [q, r] = double_line[i].qr('r', {"Right1", "Right2"}, "Right", "Left");
         double_line[i] = std::move(q);
         double_line[i + 1] = double_line[i + 1].contract(r, {{"Left1", "Right1"}, {"Left2", "Right2"}});
         double_line[i + 1] /= double_line[i + 1].norm<-1>();
      }

      for (int i = length - 2; i >= 0; i--) {
         // L-2&L-1 .. 0&1
         auto [u, s, v] = double_line[i]
                                .edge_rename({{"Down", "Down1"}})
                                .contract(double_line[i + 1].edge_rename({{"Down", "Down2"}}), {{"Right", "Left"}})
                                .svd({"Left", "Down1"}, "Right", "Left", Dc);
         double_line[i + 1] = v.edge_rename({{"Down2", "Down"}});
         double_line[i] = u.edge_rename({{"Down1", "Down"}}).multiple(s, "Right", 'u');
      }
   }
}
