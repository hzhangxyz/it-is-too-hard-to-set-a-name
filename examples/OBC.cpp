/**
 * Copyright (C) 2020-2021  Hao Zhang<zh970205@mail.ustc.edu.cn>
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
            auto names = std::vector<TAT::DefaultName>();
            if (i != 0) {
               names.emplace_back("up");
            }
            if (i != L - 1) {
               names.emplace_back("down");
            }
            if (j != 0) {
               names.emplace_back("left");
            }
            if (j != L - 1) {
               names.emplace_back("right");
            }
            lattice[i][j] = Tensor(names, std::vector<TAT::Edge<TAT::NoSymmetry>>(names.size(), D)).set(generator);
         }
      }
   }

   float get_exact() const {
      auto result = Tensor(1);
      for (int i = 0; i < L; i++) {
         for (int j = 0; j < L; j++) {
            auto contract_names = std::set<std::pair<TAT::DefaultName, TAT::DefaultName>>{{right(i), left(i)}, {down(j), up(j)}};
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
         // std::clog << "new line:\n";
         for (int j = 0; j < L; j++) {
            // std::clog << new_line[j] << "\n";
            up_to_down_aux[i + 1][j] = std::move(new_line[j]);
         }
      }
      auto result = Tensor(1);
      for (int j = 0; j < L; j++) {
         result = result.contract(up_to_down_aux[L - 1][j], {{"right", "left"}});
      }
      // std::cout << result << "\n";
      return float(result);
   }
};

int main(const int argc, char** argv) {
   std::stringstream out;
   auto cout_buf = std::cout.rdbuf();
   if (argc != 1) {
      std::cout.rdbuf(out.rdbuf());
   }
   std::mt19937 engine(0);
   std::uniform_real_distribution<double> dis(-1, 1);
   auto generator = [&]() {
      return dis(engine);
   };
   auto pbc = PBC(4, 4, generator);
   std::cout << std::fixed;
   std::cout << "Exact\t: " << pbc.get_exact() << "\n";
   for (TAT::Size cut = 1; cut <= 30; cut++) {
      std::cout << cut << "\t: " << pbc.contract_with_two_line_to_one_line(cut) << "\n";
   }
   if (argc != 1) {
      std::cout.rdbuf(cout_buf);
      std::ifstream fout(argv[1]);
      std::string sout((std::istreambuf_iterator<char>(fout)), std::istreambuf_iterator<char>());
      return sout != out.str();
   }
   return 0;
}
