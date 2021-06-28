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

#include "run_test.hpp"

void run_test() {
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B"}, {8, 8}}.range() << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B"}, {8, 8}}
                      .range()
                      .edge_rename({{"A", "C"}})
                      .edge_operator(
                            {{"C", {{"D", 4}, {"E", 2}}}, {"B", {{"F", 2}, {"G", 4}}}},
                            {"D", "F"},
                            {{"I", {"D", "F"}}, {"J", {"G", "E"}}},
                            {"J", "I"})
             << "\n";
   std::cout << TAT::Tensor<>{{"A", "B", "C"}, {2, 3, 4}}.range().edge_operator({}, {}, {}, {"B", "C", "A"}) << '\n';
   do {
      auto a =
            TAT::Tensor<double, TAT::U1Symmetry>{
                  {"Left", "Right", "Up", "Down"},
                  {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 4}, {1, 2}}, {{-1, 2}, {0, 3}, {1, 1}}, {{-1, 1}, {0, 3}, {1, 2}}}}
                  .set([]() {
                     static double i = 0;
                     return i += 1;
                  });
      auto b = a.edge_rename({{"Right", "Right1"}}).split_edge({{"Down", {{"Down1", {{{0, 1}, {1, 2}}}}, {"Down2", {{{-1, 1}, {0, 1}}}}}}});
      auto c = b.transpose({"Down1", "Right1", "Up", "Left", "Down2"});
      auto d = c.merge_edge({{"Left", {"Left", "Down2"}}});
      auto total = a.edge_rename({{"Right", "Right1"}})
                         .edge_operator(
                               {{"Down", {{"Down1", {{{0, 1}, {1, 2}}}}, {"Down2", {{{-1, 1}, {0, 1}}}}}}},
                               {},
                               {{"Left", {"Left", "Down2"}}},
                               {"Down1", "Right1", "Up", "Left"});
      std::cout << (total - d).norm<-1>() << "\n";
   } while (false);
   do {
      auto a =
            TAT::Tensor<double, TAT::FermiSymmetry>{
                  {"Left", "Right", "Up", "Down"},
                  {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 4}, {1, 2}}, {{-1, 2}, {0, 3}, {1, 1}}, {{-1, 1}, {0, 3}, {1, 2}}}}
                  .set([]() {
                     static double i = 0;
                     return i += 1;
                  });
      auto b = a.edge_rename({{"Right", "Right1"}}).split_edge({{"Down", {{"Down1", {{{0, 1}, {1, 2}}}}, {"Down2", {{{-1, 1}, {0, 1}}}}}}});
      auto r = b.reverse_edge({"Left"});
      auto c = r.transpose({"Down1", "Right1", "Up", "Left", "Down2"});
      auto d = c.merge_edge({{"Left", {"Left", "Down2"}}});
      auto total = a.edge_rename({{"Right", "Right1"}})
                         .edge_operator(
                               {{"Down", {{"Down1", {{{0, 1}, {1, 2}}}}, {"Down2", {{{-1, 1}, {0, 1}}}}}}},
                               {"Left"},
                               {{"Left", {"Left", "Down2"}}},
                               {"Down1", "Right1", "Up", "Left"});
      std::cout << (total - d).norm<-1>() << "\n";
      std::cout << total << "\n";
   } while (false);
}
