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

#pragma once
#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <TAT/TAT.hpp>

namespace tools {
   using namespace TAT;

   // default: up to down
   template<typename ScalarType, typename Symmetry, typename Name>
   auto two_line_to_one_line(
         const std::array<Name, 4>& udlr_name,
         std::vector<const Tensor<ScalarType, Symmetry, Name>*> line_1,
         std::vector<const Tensor<ScalarType, Symmetry, Name>*> line_2,
         Size cut) {
      const auto& [up, down, left, right] = udlr_name;
      const Name up1 = static_cast<const std::string&>(up) + "_1";
      const Name up2 = static_cast<const std::string&>(up) + "_2";
      const Name down1 = static_cast<const std::string&>(down) + "_1";
      const Name down2 = static_cast<const std::string&>(down) + "_2";
      const Name left1 = static_cast<const std::string&>(left) + "_1";
      const Name left2 = static_cast<const std::string&>(left) + "_2";
      const Name right1 = static_cast<const std::string&>(right) + "_1";
      const Name right2 = static_cast<const std::string&>(right) + "_2";

      int length = line_1.size();
      if (length != int(line_2.size())) {
         detail::error("Different Length When Do Two Line to One Line");
      }
      // std::clog << "Two Line to One Line Start\n";

      // product
      std::vector<Tensor<ScalarType, Symmetry, Name>> double_line;
      // std::clog << "double line:\n";
      for (auto i = 0; i < length; i++) {
         double_line.push_back(Tensor<ScalarType, Symmetry, Name>::contract(
               line_1[i]->edge_rename({{left, left1}, {right, right1}}),
               line_2[i]->edge_rename({{left, left2}, {right, right2}}),
               {{down, up}}));
         // std::clog << double_line[i] << "\n";
      }

      // left canonicalize
      for (int i = 0; i < length - 1; i++) {
         // lattice: 0 ~ L-1
         // 0 ... L-2
         auto [u, s, v] = double_line[i].svd({right1, right2}, left, right);
         double_line[i] = std::move(v);
         double_line[i + 1] = double_line[i + 1].contract(u, {{left1, right1}, {left2, right2}}).multiple(s, left, 'u');
      }
      // std::clog << "double line:\n";
      // for (int i = 0; i < length; i++) {
      //   std::clog << double_line[i] << "\n";
      //}

      // right svd
      for (int i = length - 2; i >= 0; i--) {
         // L-2 and L-1, ... 0 and 1
         // i and i+1
         // get name of double_line[i]
         auto u_names = std::set<Name>(double_line[i].names.begin(), double_line[i].names.end());
         u_names.erase(right);
         u_names.erase(up);
         u_names.erase(down);
         u_names.insert(up1);
         u_names.insert(down1);
         auto [u, s, v] = Tensor<ScalarType, Symmetry, Name>::contract(
                                double_line[i].edge_rename({{up, up1}, {down, down1}}),
                                double_line[i + 1].edge_rename({{up, up2}, {down, down2}}),
                                {{right, left}})
                                .svd(u_names, right, left, cut);
         double_line[i + 1] = v.edge_rename({{up2, up}, {down2, down}});
         double_line[i] = u.multiple(s, right, 'u').edge_rename({{up1, up}, {down1, down}});
      }

      // std::clog << "Two Line to One Line End\n";
      return double_line;
   }
} // namespace tools
#endif
