/**
 * \file sampling_gradient_lattice.hpp
 *
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

#pragma once
#ifndef SQUARE_SAMPLING_GRADIENT_LATTICE_HPP
#define SQUARE_SAMPLING_GRADIENT_LATTICE_HPP

#include "abstract_network_lattice.hpp"
#include "auxiliaries_system.hpp"

namespace square {
   template<typename T>
   struct SpinConfiguration : SquareAuxiliariesSystem<T> {
      const SamplingGradientLattice<T>* owner;
      std::vector<std::vector<int>> configuration;

      using SquareAuxiliariesSystem<T>::M;
      using SquareAuxiliariesSystem<T>::N;
      using SquareAuxiliariesSystem<T>::dimension_cut;
      using SquareAuxiliariesSystem<T>::lattice;
      using SquareAuxiliariesSystem<T>::operator();

      SpinConfiguration() : SquareAuxiliariesSystem<T>(), owner(nullptr), configuration() {}
      SpinConfiguration(const SpinConfiguration<T>&) = default;
      SpinConfiguration(SpinConfiguration<T>&&) = default;
      SpinConfiguration<T>& operator=(const SpinConfiguration<T>&) = default;
      SpinConfiguration<T>& operator=(SpinConfiguration<T>&&) = default;

      SpinConfiguration(const SamplingGradientLattice<T>* owner) :
            SquareAuxiliariesSystem<T>(owner->M, owner->N, owner->dimension_cut), owner(owner) {
         for (auto i = 0; i < M; i++) {
            auto& row = configuration.emplace_back();
            for (auto j = 0; j < N; j++) {
               row.push_back(-1);
            }
         }
      }

      void set(const std::tuple<int, int>& position, int spin) {
         auto [x, y] = position;
         if (configuration[x][y] != spin) {
            if (spin == -1) {
               lattice[x][y]->unset();
               configuration[x][y] = spin;
            } else {
#ifdef LAZY_DEBUG
               std::clog << "Flip at (" << x << ", " << y << ") to " << spin << "\n";
#endif
               lattice[x][y]->set(owner->lattice[x][y].shrink({{"P", spin}}));
               configuration[x][y] = spin;
            }
         }
      }

      auto operator()(const std::map<std::tuple<int, int>, int>& replacement, T ws, char hint = ' ') const {
         auto real_replacement = std::map<std::tuple<int, int>, Tensor<T>>();
         for (auto& [position, spin] : replacement) {
            auto [x, y] = position;
            if (configuration[x][y] != spin) {
               real_replacement[{x, y}] = owner->lattice[x][y].shrink({{"P", spin}});
            }
         }
         if (real_replacement.empty()) {
            return ws;
         } else {
            return operator()(real_replacement, hint);
         }
      }

      auto operator()() const {
         return operator()(std::map<std::tuple<int, int>, Tensor<T>>());
      }
   };

   template<typename T>
   struct SamplingGradientLattice : AbstractNetworkLattice<T> {
      Size dimension_cut;
      SpinConfiguration<T> spin;

      // spin应当只用this初始化, 随后initialize_spin即可
      SamplingGradientLattice() : AbstractNetworkLattice<T>(), dimension_cut(0), spin() {}

      SamplingGradientLattice(const SamplingGradientLattice<T>& other) :
            AbstractNetworkLattice<T>(other), dimension_cut(other.dimension_cut), spin(this) {
         initialize_spin(other.spin.configuration);
      }
      SamplingGradientLattice(SamplingGradientLattice<T>&& other) :
            AbstractNetworkLattice<T>(std::move(other)), dimension_cut(other.dimension_cut), spin(this) {
         initialize_spin(other.spin.configuration);
      }
      SamplingGradientLattice<T>& operator=(const SamplingGradientLattice<T>& other) {
         if (this != &other) {
            new (this) SamplingGradientLattice<T>(other);
         }
         return *this;
      }
      SamplingGradientLattice<T>& operator=(SamplingGradientLattice<T>&& other) {
         if (this != &other) {
            new (this) SamplingGradientLattice<T>(std::move(other));
         }
         return *this;
      }

      SamplingGradientLattice(int M, int N, Size D, Size Dc, Size d) : AbstractNetworkLattice<T>(M, N, D, d), dimension_cut(Dc), spin(this) {}

      explicit SamplingGradientLattice(const SimpleUpdateLattice<T>& other, Size Dc);

      using AbstractNetworkLattice<T>::M;
      using AbstractNetworkLattice<T>::N;
      using AbstractNetworkLattice<T>::dimension_physics;
      using AbstractNetworkLattice<T>::hamiltonians;
      using AbstractNetworkLattice<T>::dimension_virtual;
      using AbstractNetworkLattice<T>::lattice;

      void set_dimension_cut(Size Dc) {
         dimension_cut = Dc;
         auto configuration = std::move(spin.configuration);
         spin = SpinConfiguration<T>(this);
         initialize_spin(configuration);
      }

      void initialize_spin(std::function<int(int, int)> function) {
         for (auto i = 0; i < M; i++) {
            for (auto j = 0; j < N; j++) {
               spin.set({i, j}, function(i, j));
            }
         }
      }

      void initialize_spin(const std::vector<std::vector<int>>& configuration) {
         for (auto i = 0; i < M; i++) {
            for (auto j = 0; j < N; j++) {
               spin.set({i, j}, configuration[i][j]);
            }
         }
      }

      auto markov(
            std::uint64_t total_step,
            std::map<std::string, std::map<std::vector<std::tuple<int, int>>, std::shared_ptr<const Tensor<T>>>> observers,
            bool calculate_energy = false,
            bool calculate_gradient = false) {
         if (calculate_gradient) {
            calculate_energy = true;
         }
         if (calculate_energy) {
            observers["Energy"] = hamiltonians;
         }
         // TODO 如何计算gradient的误差?
         T sum_of_Es = 0;
         std::vector<std::vector<Tensor<T>>> holes;
         std::vector<std::vector<Tensor<T>>> holes_with_Es;
         std::vector<std::vector<Tensor<T>>> gradient;
         if (calculate_gradient) {
            for (auto i = 0; i < M; i++) {
               auto& row_h = holes.emplace_back();
               auto& row_e = holes_with_Es.emplace_back();
               auto& row_g = gradient.emplace_back();
               for (auto j = 0; j < N; j++) {
                  row_h.push_back(lattice[i][j].same_shape().zero());
                  row_e.push_back(lattice[i][j].same_shape().zero());
                  row_g.push_back(lattice[i][j].same_shape().zero());
               }
            }
         }
         // 观测量一定是实数
         auto result = std::map<std::string, std::map<std::vector<std::tuple<int, int>>, real<T>>>();
         auto result_variance_square = std::map<std::string, std::map<std::vector<std::tuple<int, int>>, real<T>>>();
         auto result_square = std::map<std::string, std::map<std::vector<std::tuple<int, int>>, real<T>>>();
         T ws = spin();
         std::cout << clear_line << "Markov sampling start, total_step=" << total_step << ", dimension=" << dimension_virtual
                   << ", dimension_cut=" << dimension_cut << ", First ws is " << ws << "\n"
                   << std::flush;
         auto positions_sequence = _markov_sampling_positions_sequence();
         random::split_seed();
         for (std::uint64_t step = 0; step < total_step; step++) {
            ws = _markov_spin(ws, positions_sequence);
            T Es = 0;
            for (const auto& [kind, group] : observers) {
               bool is_energy = kind == "Energy";
               for (const auto& [positions, tensor] : group) {
                  int body = positions.size();
                  auto current_spin = std::vector<int>();
                  for (auto i = 0; i < body; i++) {
                     current_spin.push_back(spin.configuration[std::get<0>(positions[i])][std::get<1>(positions[i])]);
                  }
                  real<T> value = 0;
                  for (const auto& [spins_out, element] : _find_element(*tensor).at(current_spin)) {
                     auto map = std::map<std::tuple<int, int>, int>();
                     for (auto i = 0; i < body; i++) {
                        map[positions[i]] = spins_out[i];
                     }
                     T wss = spin(map, ws);
                     auto this_term = element * wss / ws;
                     if (is_energy) {
                        // 用于求梯度，这个可能是复数
                        Es += this_term;
                     }
                     value += scalar_to<real<T>>(this_term);
                  }
                  result[kind][positions] += value;
                  result_square[kind][positions] += value * value;
               }
            }
            if (calculate_gradient) {
               sum_of_Es += Es;
               for (auto i = 0; i < M; i++) {
                  for (auto j = 0; j < N; j++) {
                     Tensor<T> raw_hole = spin({{i, j}}).edge_rename({{"L0", "L"}, {"R0", "R"}, {"U0", "U"}, {"D0", "D"}});
                     Tensor<T> hole = (raw_hole.conjugate() / conj(ws)).expand({{"P", {spin.configuration[i][j], dimension_physics}}});
                     holes_with_Es[i][j] += hole * Es;
                     holes[i][j] += hole;
                  }
               }
            }
            std::cout << clear_line << "Markov sampling, total_step=" << total_step << ", dimension=" << dimension_virtual
                      << ", dimension_cut=" << dimension_cut << ", step=" << (step + 1) << "\r" << std::flush;
         }
         random::merge_seed();
         // TODO reduce here
         for (auto& [kind, group] : result) {
            const auto& group_square = result_square.at(kind);
            for (auto& [positions, value] : group) {
               value /= total_step;
               auto value_square = group_square.at(positions);
               value_square /= total_step;
               result_variance_square[kind][positions] = (value_square - value * value) / (total_step - 1);
            }
         }
         if (calculate_energy) {
            real<T> energy = 0;
            real<T> energy_variance_square = 0;
            const auto& energy_variance_square_pool = result_variance_square.at("Energy");
            for (const auto& [positions, value] : result.at("Energy")) {
               energy += value;
               energy_variance_square += energy_variance_square_pool.at(positions);
            };
            std::cout << clear_line << "Markov sample done, total_step=" << total_step << ", dimension=" << dimension_virtual
                      << ", dimension_cut=" << dimension_cut << ", Energy=" << energy / (M * N)
                      << " with sigma=" << std::sqrt(energy_variance_square) / (M * N) << "\n"
                      << std::flush;
         } else {
            std::cout << clear_line << "Markov sample done, total_step=" << total_step << ", dimension=" << dimension_virtual
                      << ", dimension_cut=" << dimension_cut << "\n"
                      << std::flush;
         }
         if (calculate_gradient) {
            for (auto i = 0; i < M; i++) {
               for (auto j = 0; j < N; j++) {
                  // TODO reduce gradient
                  gradient[i][j] = 2 * (holes_with_Es[i][j] / total_step) - 2 * (sum_of_Es / total_step) * (holes[i][j] / total_step);
               }
            }
         }
         return std::make_tuple(std::move(result), std::move(result_variance_square), std::move(gradient));
      }

      auto ergodic(
            std::map<std::string, std::map<std::vector<std::tuple<int, int>>, std::shared_ptr<const Tensor<T>>>> observers,
            bool calculate_energy = false) {
         std::cout << clear_line << "Ergodic sampling start, dimension=" << dimension_virtual << ", dimension_cut=" << dimension_cut << "\n"
                   << std::flush;
         if (calculate_energy) {
            observers["Energy"] = hamiltonians;
         }
         auto result = std::map<std::string, std::map<std::vector<std::tuple<int, int>>, real<T>>>();
         real<T> sum_of_ws_square = 0;
         std::uint64_t total_step = std::pow(dimension_physics, M * N);
         for (std::uint64_t step = 0; step < total_step; step++) {
            _ergodic_spin(step);
            T ws = spin();
            sum_of_ws_square += std::norm(ws);
            for (const auto& [kind, group] : observers) {
               for (const auto& [positions, tensor] : group) {
                  int body = positions.size();
                  auto current_spin = std::vector<int>();
                  for (auto i = 0; i < body; i++) {
                     current_spin.push_back(spin.configuration[std::get<0>(positions[i])][std::get<1>(positions[i])]);
                  }
                  real<T> value = 0;

                  for (const auto& [spins_out, element] : _find_element(*tensor).at(current_spin)) {
                     auto map = std::map<std::tuple<int, int>, int>();
                     for (auto i = 0; i < body; i++) {
                        map[positions[i]] = spins_out[i];
                     }
                     T wss = spin(map, ws);
                     value += scalar_to<real<T>>(element * wss / ws);
                  }
                  result[kind][positions] += value * std::norm(ws);
               }
            }
            if (calculate_energy) {
               real<T> energy = 0;
               for (const auto& [positions, value] : result.at("Energy")) {
                  energy += value;
               };
               std::cout << clear_line << "Ergodic sampling, total_step=" << total_step << ", dimension=" << dimension_virtual
                         << ", dimension_cut=" << dimension_cut << ", step=" << (step + 1) << ", Energy=" << energy / (sum_of_ws_square * M * N)
                         << "\r" << std::flush;
            } else {
               std::cout << clear_line << "Ergodic sampling, total_step=" << total_step << ", dimension=" << dimension_virtual
                         << ", dimension_cut=" << dimension_cut << ", step=" << (step + 1) << "\r" << std::flush;
            }
         }
         for (auto& [kind, group] : result) {
            for (auto& [positions, value] : group) {
               value /= sum_of_ws_square;
            }
         }
         if (calculate_energy) {
            real<T> energy = 0;
            for (const auto& [positions, value] : result.at("Energy")) {
               energy += value;
            };
            std::cout << clear_line << "Ergodic sample done, total_step=" << total_step << ", dimension=" << dimension_virtual
                      << ", dimension_cut=" << dimension_cut << ", Energy=" << energy / (M * N) << "\n"
                      << std::flush;
         } else {
            std::cout << clear_line << "Ergodic sample done, total_step=" << total_step << ", dimension=" << dimension_virtual
                      << ", dimension_cut=" << dimension_cut << "\n"
                      << std::flush;
         }
         return result;
      }

      void _ergodic_spin(std::uint64_t step) {
         for (auto i = 0; i < M; i++) {
            for (auto j = 0; j < N; j++) {
               spin.set({i, j}, step % dimension_physics);
               step /= dimension_physics;
            }
         }
      }

      void equilibrate(std::uint64_t total_step) {
         std::cout << clear_line << "Equilibrating start, total_step=" << total_step << ", dimension=" << dimension_virtual
                   << ", dimension_cut=" << dimension_cut << "\n"
                   << std::flush;
         T ws = spin();
         auto positions_sequence = _markov_sampling_positions_sequence();
         for (std::uint64_t step = 0; step < total_step; step++) {
            ws = _markov_spin(ws, positions_sequence);
            std::cout << clear_line << "Equilibrating, total_step=" << total_step << ", dimension=" << dimension_virtual
                      << ", dimension_cut=" << dimension_cut << ", step=" << (step + 1) << "\r" << std::flush;
         }
         std::cout << clear_line << "Equilibrate done, total_step=" << total_step << ", dimension=" << dimension_virtual
                   << ", dimension_cut=" << dimension_cut << "\n"
                   << std::flush;
      }

      T _markov_spin(T ws, const std::vector<std::tuple<std::vector<std::tuple<int, int>>, char>>& positions_sequence) {
         for (auto iter = positions_sequence.begin(); iter != positions_sequence.end(); ++iter) {
            const auto& [positions, hint] = *iter;
            const auto& hamiltonian = hamiltonians.at(positions);
            ws = _markov_single_term(ws, positions, hamiltonian, hint);
         }
         for (auto iter = positions_sequence.rbegin(); iter != positions_sequence.rend(); ++iter) {
            const auto& [positions, hint] = *iter;
            const auto& hamiltonian = hamiltonians.at(positions);
            ws = _markov_single_term(ws, positions, hamiltonian, hint);
         }
         return ws;
      }

      auto _markov_sampling_positions_sequence() const {
         auto result = std::vector<std::tuple<std::vector<std::tuple<int, int>>, char>>();
         // 应该不存在常数项的hamiltonians
         // 双点，横+竖, 单点和横向放在一起
         for (auto i = 0; i < M; i++) {
            for (auto j = 0; j < N; j++) {
               if (auto found = hamiltonians.find({{i, j}}); found != hamiltonians.end()) {
                  result.emplace_back(std::vector<std::tuple<int, int>>{{i, j}}, 'h');
               }
               if (auto found = hamiltonians.find({{i, j}, {i, j + 1}}); found != hamiltonians.end()) {
                  result.emplace_back(std::vector<std::tuple<int, int>>{{i, j}, {i, j + 1}}, 'h');
               }
               if (auto found = hamiltonians.find({{i, j + 1}, {i, j}}); found != hamiltonians.end()) {
                  result.emplace_back(std::vector<std::tuple<int, int>>{{i, j + 1}, {i, j}}, 'h');
               }
            }
         }
         for (auto j = 0; j < N; j++) {
            for (auto i = 0; i < M; i++) {
               if (auto found = hamiltonians.find({{i, j}, {i + 1, j}}); found != hamiltonians.end()) {
                  result.emplace_back(std::vector<std::tuple<int, int>>{{i, j}, {i + 1, j}}, 'v');
               }
               if (auto found = hamiltonians.find({{i + 1, j}, {i, j}}); found != hamiltonians.end()) {
                  result.emplace_back(std::vector<std::tuple<int, int>>{{i + 1, j}, {i, j}}, 'v');
               }
            }
         }
         // 其他类型和更多的格点暂时不支持
         if (result.size() != hamiltonians.size()) {
            throw NotImplementedError("Unsupported markov sampling style");
         }
         return result;
      }

      T _markov_single_term(
            T ws,
            const std::vector<std::tuple<int, int>>& positions,
            const std::shared_ptr<const Tensor<T>>& hamiltonian,
            char hint = ' ') {
#ifdef LAZY_DEBUG
         std::clog << "Hopping at ";
         for (const auto& [x, y] : positions) {
            std::clog << "(" << x << ", " << y << ") ";
         }
         std::clog << "\n";
#endif
         int body = positions.size();
         auto current_spin = std::vector<int>();
         for (auto i = 0; i < body; i++) {
            current_spin.push_back(spin.configuration[std::get<0>(positions[i])][std::get<1>(positions[i])]);
         }
         const auto& hamiltonian_elements = _find_hopping_element(*hamiltonian);
         const auto& possible_hopping = hamiltonian_elements.at(current_spin);
         if (!possible_hopping.empty()) {
            int hopping_number = possible_hopping.size();
            auto random_index = random::uniform<int>(0, hopping_number - 1)();
            auto iter = possible_hopping.begin();
            std::advance(iter, random_index);
            const auto& [spins_new, element] = *iter;
            auto replacement = std::map<std::tuple<int, int>, int>();
            for (auto i = 0; i < body; i++) {
               replacement[positions[i]] = spins_new[i];
            }
            T wss = spin(replacement, ws, hint);
            int hopping_number_s = hamiltonian_elements.at(spins_new).size();
            T wss_over_ws = wss / ws;
            real<T> p = std::norm(wss_over_ws) * real<T>(hopping_number) / real<T>(hopping_number_s);
            if (random::uniform<real<T>>(0, 1)() < p) {
               ws = wss;
               for (auto i = 0; i < body; i++) {
                  spin.set(positions[i], spins_new[i]);
               }
            }
         }
         return ws;
      }

      inline static std::map<const Tensor<T>*, std::map<std::vector<int>, std::map<std::vector<int>, T>>> tensor_element_map = {};

      const std::map<std::vector<int>, std::map<std::vector<int>, T>>& _find_element(const Tensor<T>& tensor) const {
         auto tensor_id = &tensor;
         if (auto found = tensor_element_map.find(tensor_id); found != tensor_element_map.end()) {
            return found->second;
         }
         int body = tensor.names.size() / 2;
         auto& result = tensor_element_map[tensor_id];
         auto names = std::vector<Name>();
         auto index = std::vector<int>();
         for (auto i = 0; i < body; i++) {
            names.push_back("I" + std::to_string(i));
            index.push_back(0);
         }
         for (auto i = 0; i < body; i++) {
            names.push_back("O" + std::to_string(i));
            index.push_back(0);
         }
         while (true) {
            auto map = std::map<Name, Size>();
            for (auto i = 0; i < 2 * body; i++) {
               map[names[i]] = index[i];
            }
            auto value = tensor.const_at(map);
            if (value != T(0)) {
               auto spins_in = std::vector<int>();
               auto spins_out = std::vector<int>();
               for (auto i = 0; i < body; i++) {
                  spins_in.push_back(index[i]);
               }
               for (auto i = body; i < 2 * body; i++) {
                  spins_out.push_back(index[i]);
               }
               result[std::move(spins_in)][std::move(spins_out)] = value;
            }
            int active_position = 0;
            index[active_position] += 1;
            while (index[active_position] == dimension_physics) {
               index[active_position] = 0;
               active_position += 1;
               if (active_position == 2 * body) {
                  return result;
               }
               index[active_position] += 1;
            }
         }
      }

      inline static std::map<const Tensor<T>*, std::map<std::vector<int>, std::map<std::vector<int>, T>>> tensor_hopping_element_map = {};

      const std::map<std::vector<int>, std::map<std::vector<int>, T>>& _find_hopping_element(const Tensor<T>& tensor) const {
         auto tensor_id = &tensor;
         if (auto found = tensor_hopping_element_map.find(tensor_id); found != tensor_hopping_element_map.end()) {
            return found->second;
         }
         int body = tensor.names.size() / 2;
         auto& result = tensor_hopping_element_map[tensor_id];
         auto names = std::vector<Name>();
         auto index = std::vector<int>();
         for (auto i = 0; i < body; i++) {
            names.push_back("I" + std::to_string(i));
            index.push_back(0);
         }
         for (auto i = 0; i < body; i++) {
            names.push_back("O" + std::to_string(i));
            index.push_back(0);
         }
         while (true) {
            auto map = std::map<Name, Size>();
            for (auto i = 0; i < 2 * body; i++) {
               map[names[i]] = index[i];
            }
            auto value = tensor.const_at(map);
            if (value != T(0)) {
               auto spins_in = std::vector<int>();
               auto spins_out = std::vector<int>();
               for (auto i = 0; i < body; i++) {
                  spins_in.push_back(index[i]);
               }
               for (auto i = body; i < 2 * body; i++) {
                  spins_out.push_back(index[i]);
               }
               if (spins_in != spins_out) {
                  result[std::move(spins_in)][std::move(spins_out)] = value;
               } else {
                  result[std::move(spins_in)];
               }
            }
            int active_position = 0;
            index[active_position] += 1;
            while (index[active_position] == dimension_physics) {
               index[active_position] = 0;
               active_position += 1;
               if (active_position == 2 * body) {
                  return result;
               }
               index[active_position] += 1;
            }
         }
      }
   };

   template<typename T>
   std::ostream& operator<(std::ostream& out, const SamplingGradientLattice<T>& lattice) {
      using TAT::operator<;
      out < static_cast<const AbstractNetworkLattice<T>&>(lattice);
      out < lattice.dimension_cut;
      out < lattice.spin.configuration;
      // TODO: output all configuration across mpi process
      return out;
   }

   template<typename T>
   std::istream& operator>(std::istream& in, SamplingGradientLattice<T>& lattice) {
      using TAT::operator>;
      in > static_cast<AbstractNetworkLattice<T>&>(lattice);
      in > lattice.dimension_cut;
      std::vector<std::vector<int>> configuration;
      lattice.spin = SpinConfiguration(&lattice);
      in > configuration;
      if (!configuration.empty()) {
         lattice.initialize_spin(configuration);
      }
      return in;
   }
} // namespace square

#endif
