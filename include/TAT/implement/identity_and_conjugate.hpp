/**
 * \file identity_and_conjugate.hpp
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
#ifndef TAT_IDENTITY_AND_CONJUGATE_HPP
#define TAT_IDENTITY_AND_CONJUGATE_HPP

#include "../structure/tensor.hpp"
#include "../utility/pmr_resource.hpp"
#include "../utility/timer.hpp"

namespace TAT {
   inline timer conjugate_guard("conjugate");

   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::conjugate() const {
      auto timer_guard = conjugate_guard();
      auto pmr_guard = scope_resource<small_buffer_size>();
      if constexpr (Symmetry::length == 0 && is_real<ScalarType>) {
         return *this;
      }
      auto result_edges = pmr::vector<Edge<Symmetry>>();
      result_edges.reserve(names.size());
      for (const auto& edge : core->edges) {
         auto& result_edge = result_edges.emplace_back();
         if constexpr (Symmetry::is_fermi_symmetry) {
            result_edge.arrow = !edge.arrow;
         }
         if constexpr (Symmetry::length != 0) {
            result_edge.conjugated = !edge.arrow;
         }
         for (const auto& [symmetry, dimension] : edge.map) {
            result_edge.map.emplace_back(-symmetry, dimension);
         }
         do_sort(result_edge.map);
      }
      auto transpose_flag = pmr::vector<Rank>(names.size(), 0);
      auto valid_flag = pmr::vector<bool>(1, true);
      auto result = Tensor<ScalarType, Symmetry, Name>(names, result_edges);
      for (const auto& [symmetries, block] : core->blocks) {
         auto result_symmetries = pmr::vector<Symmetry>();
         for (const auto& symmetry : symmetries) {
            result_symmetries.push_back(-symmetry);
         }
         // result.core->blocks.at(result_symmetries) <- block
         const Size total_size = block.size();
         ScalarType* destination = map_at<true>(result.core->blocks, result_symmetries).data();
         const ScalarType* source = block.data();
         bool parity = false;
         if constexpr (Symmetry::is_fermi_symmetry) {
            parity = Symmetry::get_split_merge_parity(symmetries, transpose_flag, valid_flag);
         }
         if constexpr (is_complex<ScalarType>) {
            if (parity) {
               for (Size i = 0; i < total_size; i++) {
                  destination[i] = -std::conj(source[i]);
               }
            } else {
               for (Size i = 0; i < total_size; i++) {
                  destination[i] = std::conj(source[i]);
               }
            }
         } else {
            if (parity) {
               for (Size i = 0; i < total_size; i++) {
                  destination[i] = -source[i];
               }
            } else {
               for (Size i = 0; i < total_size; i++) {
                  destination[i] = source[i];
               }
            }
         }
      }
      return result;
   }

   /// \private
   void set_to_identity(auto* pointer, const std::span<const Size>& dimension, const std::span<const Size>& leading, Rank rank) {
      auto current_index = pmr::vector<Size>(rank, 0);
      while (true) {
         *pointer = 1;
         Rank active_position = rank - 1;

         current_index[active_position]++;
         pointer += leading[active_position];

         while (current_index[active_position] == dimension[active_position]) {
            current_index[active_position] = 0;
            pointer -= dimension[active_position] * leading[active_position];

            if (active_position == 0) [[unlikely]] {
               return;
            }
            active_position--;

            current_index[active_position]++;
            pointer += leading[active_position];
         }
      }
   }

   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   Tensor<ScalarType, Symmetry, Name>& Tensor<ScalarType, Symmetry, Name>::identity_implement(const auto& pairs) & {
      auto rank = names.size();
      auto half_rank = rank / 2;
      auto ordered_pair = pmr::vector<std::tuple<Name, Name>>();
      auto ordered_pair_index = pmr::vector<std::tuple<Rank, Rank>>();
      ordered_pair.reserve(half_rank);
      ordered_pair_index.reserve(half_rank);
      auto valid_index = pmr::vector<bool>(rank, true);
      for (Rank i = 0; i < rank; i++) {
         if (valid_index[i]) {
            const auto& name_to_find = names[i];
            const Name* name_correspond = nullptr;
            for (const auto& [name_1, name_2] : pairs) {
               if (name_1 == name_to_find) {
                  name_correspond = &name_2;
                  break;
               }
               if (name_2 == name_to_find) {
                  name_correspond = &name_1;
                  break;
               }
            }
            ordered_pair.push_back({name_to_find, *name_correspond});
            auto index_correspond = map_at(name_to_index, *name_correspond);
            ordered_pair_index.push_back({i, index_correspond});
            valid_index[index_correspond] = false;
         }
      }

      zero();

      for (auto& [symmetries, block] : core->blocks) {
         auto dimension = pmr::vector<Size>(rank);
         auto leading = pmr::vector<Size>(rank);
         for (Rank i = rank; i-- > 0;) {
            dimension[i] = map_at(core->edges[i].map, symmetries[i]);
            if (i == rank - 1) {
               leading[i] = 1;
            } else {
               leading[i] = leading[i + 1] * dimension[i + 1];
            }
         }
         auto pair_dimension = pmr::vector<Size>();
         auto pair_leading = pmr::vector<Size>();
         pair_dimension.reserve(half_rank);
         pair_leading.reserve(half_rank);
         for (Rank i = 0; i < half_rank; i++) {
            pair_dimension.push_back(dimension[std::get<0>(ordered_pair_index[i])]);
            pair_leading.push_back(leading[std::get<0>(ordered_pair_index[i])] + leading[std::get<1>(ordered_pair_index[i])]);
            // ordered_pair_index使用较大的leading进行从大到小排序，所以pair_leading一定降序
         }
         set_to_identity(block.data(), pair_dimension, pair_leading, half_rank);
      }

      return *this;
   }
} // namespace TAT
#endif
