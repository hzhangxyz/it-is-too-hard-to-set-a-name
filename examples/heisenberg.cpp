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

#undef TAT_USE_TIMER
#define TAT_USE_SIMPLE_NOSYMMETRY
// #define TAT_USE_SIMPLE_NAME
// #define LAZY_DEBUG

#include <fstream>
#include <square/square.hpp>

enum struct LatticeType
{
   Exact,
   Simple,
   Sample
};

int main(int argc, char** argv) {
   bool real_cin = true;
   std::streambuf* cinbuf = std::cin.rdbuf();
   std::ifstream config_file;
   if (argc != 1) {
      config_file.open(argv[1]);
      if (config_file.is_open()) {
         std::cin.rdbuf(config_file.rdbuf());
         real_cin = false;
      }
   } else {
      config_file.open("INPUT");
      if (config_file.is_open()) {
         std::cin.rdbuf(config_file.rdbuf());
         real_cin = false;
      }
   }
   if (real_cin) {
      TAT::mpi.out_one() << "Input directly from stdin is not support more then one process";
   }
   // lattice
   LatticeType lattice_type = LatticeType::Simple;
   square::ExactLattice<double> exact_lattice;
   square::SimpleUpdateLattice<double> simple_update_lattice;
   square::SamplingGradientLattice<double> sampling_gradient_lattice;
   while (true) {
      if (real_cin) {
         TAT::mpi.out_one() << "> ";
      }
      std::string command;
      std::cin >> command;
      if (command == "exit") {
         TAT::mpi.out_one() << "Exiting\n";
         return 0;
      }
      if (command == "") {
         if (!real_cin) {
            TAT::mpi.out_one() << "End of file, continue reading from stdin\n";
            std::cin.rdbuf(cinbuf);
            real_cin = true;
         }
         continue;
      }
      if (command == "use") {
         std::string new_type;
         std::cin >> new_type;
         if (new_type == "exact") {
            lattice_type = LatticeType::Exact;
         } else if (new_type == "simple") {
            lattice_type = LatticeType::Simple;
         } else if (new_type == "sample") {
            lattice_type = LatticeType::Sample;
         } else {
            TAT::mpi.err_one() << "Unknown type: " << new_type << "\n";
            return -1;
         }
         continue;
      }
      if (command == "seed") {
         std::uint32_t seed;
         std::cin >> seed;
         square::random::seed(seed);
         continue;
      }
      switch (lattice_type) {
         case LatticeType::Exact:
            if (command == "save") {
               std::string file_name;
               std::cin >> file_name;
               std::ofstream(file_name) < TAT::fastname_dataset < exact_lattice;
            } else if (command == "open") {
               std::string file_name;
               std::cin >> file_name;
               std::ifstream(file_name) > TAT::fastname_dataset > exact_lattice;
            } else if (command == "new") {
               int M, N;
               square::Size d;
               std::cin >> M >> N >> d;
               exact_lattice = square::ExactLattice<double>(M, N, d);
               exact_lattice.set_all_horizontal_bond(square::Common<double>::SS());
               exact_lattice.set_all_vertical_bond(square::Common<double>::SS());
            } else if (command == "update") {
               int total_step;
               square::real<double> approximate_energy;
               std::cin >> total_step >> approximate_energy;
               exact_lattice.update(total_step, approximate_energy);
            } else if (command == "energy") {
               TAT::mpi.out_one() << exact_lattice.observe_energy() << "\n";
            } else {
               TAT::mpi.err_one() << "Invalid Command: " << command << "\n";
               return -1;
            }
            break;
         case LatticeType::Simple:
            if (command == "save") {
               std::string file_name;
               std::cin >> file_name;
               std::ofstream(file_name) < TAT::fastname_dataset < simple_update_lattice;
            } else if (command == "open") {
               std::string file_name;
               std::cin >> file_name;
               std::ifstream(file_name) > TAT::fastname_dataset > simple_update_lattice;
            } else if (command == "new") {
               int M, N;
               square::Size D, d;
               std::cin >> M >> N >> D >> d;
               simple_update_lattice = square::SimpleUpdateLattice<double>(M, N, D, d);
               simple_update_lattice.set_all_horizontal_bond(square::Common<double>::SS());
               simple_update_lattice.set_all_vertical_bond(square::Common<double>::SS());
            } else if (command == "update") {
               int total_step;
               square::real<double> delta_t;
               square::Size new_dimension;
               std::cin >> total_step >> delta_t >> new_dimension;
               simple_update_lattice.update(total_step, delta_t, new_dimension);
            } else if (command == "exact") {
               lattice_type = LatticeType::Exact;
               exact_lattice = square::ExactLattice(simple_update_lattice);
            } else if (command == "sample") {
               square::Size Dc;
               std::cin >> Dc;
               lattice_type = LatticeType::Sample;
               sampling_gradient_lattice = square::SamplingGradientLattice<double>(simple_update_lattice, Dc);
            } else {
               std::cerr << "Invalid Command: " << command << "\n";
               return -1;
            }
            break;
         case LatticeType::Sample:
            if (command == "save") {
               std::string file_name;
               std::cin >> file_name;
               std::ofstream(file_name) < TAT::fastname_dataset < sampling_gradient_lattice;
            } else if (command == "open") {
               std::string file_name;
               std::cin >> file_name;
               std::ifstream(file_name) > TAT::fastname_dataset > sampling_gradient_lattice;
            } else if (command == "new") {
               int M, N;
               square::Size D, Dc, d;
               std::cin >> M >> N >> D >> Dc >> d;
               sampling_gradient_lattice = square::SamplingGradientLattice<double>(M, N, D, Dc, d);
               sampling_gradient_lattice.set_all_horizontal_bond(square::Common<double>::SS());
               sampling_gradient_lattice.set_all_vertical_bond(square::Common<double>::SS());
            } else if (command == "ergodic") {
               sampling_gradient_lattice.ergodic({}, true);
            } else if (command == "markov") {
               std::uint64_t total_step;
               std::cin >> total_step;
               sampling_gradient_lattice.markov(total_step, {}, true);
            } else if (command == "gradient") {
               std::uint64_t gradient_step, markov_step;
               square::real<double> step_size;
               std::cin >> gradient_step >> step_size >> markov_step;
               TAT::mpi.out_one() << "Gradient descent start, total_step=" << gradient_step << "\n";
               TAT::mpi.out_one() << "\n\n\n";
               const char* move_up = "\u001b[1A";
               for (std::uint64_t step = 0; step < gradient_step; step++) {
                  TAT::mpi.out_one() << move_up << "\r" << square::clear_line << move_up << "\r" << square::clear_line << move_up << "\r"
                                     << square::clear_line << "Gradient descenting, total_step=" << gradient_step << ", step=" << (step + 1) << "\n";
                  auto [result, variance, gradient] = sampling_gradient_lattice.markov(markov_step, {}, true, true);
                  for (auto i = 0; i < sampling_gradient_lattice.M; i++) {
                     for (auto j = 0; j < sampling_gradient_lattice.N; j++) {
                        sampling_gradient_lattice.lattice[i][j] -= step_size * gradient[i][j];
                        sampling_gradient_lattice.lattice[i][j] /= sampling_gradient_lattice.lattice[i][j].norm<-1>();
                     }
                  }
                  const auto& energy = result.at("Energy");
                  const auto& energy_variance_square = variance.at("Energy");
                  square::real<double> total_energy = 0;
                  for (const auto& [positions, value] : energy) {
                     total_energy += value;
                  }
                  square::real<double> total_energy_variance_square = 0;
                  for (const auto& [positions, value] : energy_variance_square) {
                     total_energy_variance_square += value;
                  }
                  auto site_number = sampling_gradient_lattice.M * sampling_gradient_lattice.N;
                  TAT::mpi.out_one() << "Current Energy is " << total_energy / site_number
                                     << " with sigma=" << std::sqrt(total_energy_variance_square) / site_number;
               }
               TAT::mpi.out_one() << "\n"
                                  << "Gradient descent done, total_step=" << gradient_step << "\n";
            } else if (command == "equilibrate") {
               sampling_gradient_lattice.initialize_spin([](int i, int j) {
                  return (i + j) % 2;
               });
               std::uint64_t total_step;
               std::cin >> total_step;
               sampling_gradient_lattice.equilibrate(total_step);
            } else if (command == "exact") {
               lattice_type = LatticeType::Exact;
               exact_lattice = square::ExactLattice(sampling_gradient_lattice);
            } else if (command == "simple") {
               lattice_type = LatticeType::Simple;
               simple_update_lattice = square::SimpleUpdateLattice<double>(sampling_gradient_lattice);
            } else {
               std::cerr << "Invalid Command: " << command << "\n";
               return -1;
            }
            break;
      }
   }
   return 0;
}
