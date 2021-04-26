/**
 * \file multiple.hpp
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
#ifndef TAT_MULTIPLE_HPP
#define TAT_MULTIPLE_HPP

#include "../structure/tensor.hpp"
#include "../utility/const_integral.hpp"
#include "../utility/pmr_resource.hpp"
#include "../utility/timer.hpp"

namespace TAT {
   template<typename ScalarType, typename ScalarTypeS>
   void multiple_kernel(Size m, Size k, Size n, ScalarType* data_destination, const ScalarType* data_source, const ScalarTypeS* S) {
      const auto const_n_variant = to_const_integral<Size, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(n);
      std::visit(
            [&](const auto& const_n) {
               const auto n = const_n.value();
               for (Size a = 0; a < m; a++) {
                  for (Size b = 0; b < k; b++) {
                     auto v = S[b];
                     const auto* data_source_block = &data_source[(a * k + b) * n];
                     auto* data_destination_block = &data_destination[(a * k + b) * n];
                     for (Size c = 0; c < n; c++) {
                        data_destination_block[c] = data_source_block[c] * v;
                     }
                  }
               }
            },
            const_n_variant);
   }

   inline timer multiple_guard("multiple");

   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   Tensor<ScalarType, Symmetry, Name>
   Tensor<ScalarType, Symmetry, Name>::multiple(const SingularType& S, const Name& name, char direction, bool division) const {
      auto timer_guard = multiple_guard();
      auto pmr_guard = scope_resource<small_buffer_size>();
      bool different_direction;
      if (direction == 'u' || direction == 'U') {
         different_direction = false;
      } else if (direction == 'v' || direction == 'V') {
         different_direction = true;
      } else {
         TAT_error("Direction invalid in multiple");
         return *this;
      }
      const auto found = map_find(name_to_index, name);
      if (found == name_to_index.end()) [[unlikely]] {
         TAT_warning_or_error_when_name_missing("Name not found in multiple");
         return *this;
      }
      auto result = same_shape();
      auto index = found->second;
      for (const auto& [symmetries, block_source] : core->blocks) {
         auto& block_destination = map_at(result.core->blocks, symmetries);
         auto symmetry_of_s = symmetries[index];
         if (different_direction) {
            symmetry_of_s = -symmetry_of_s;
         }
         const auto& vector_in_S = map_at<true>(S.core->blocks, pmr::vector<Symmetry>{-symmetry_of_s, symmetry_of_s});
         auto dimension = map_at(S.core->edges[1].map, symmetry_of_s);
         auto dimension_plus_one = dimension + 1;
         Rank i = 0;
         Size m = 1;
         for (; i < index; i++) {
            m *= map_at(core->edges[i].map, symmetries[i]);
         }
         Size k = map_at(core->edges[i].map, symmetries[i]);
         Size n = 1;
         for (i++; i < names.size(); i++) {
            n *= map_at(core->edges[i].map, symmetries[i]);
         }
         if (dimension != k) [[unlikely]] {
            TAT_error("Vector Size incompatible in Multiple with a tensor");
         }
         const auto* data_source = block_source.data();
         auto* data_destination = block_destination.data();

         using ScalarTypeS = typename std::remove_cv_t<std::remove_reference_t<decltype(vector_in_S)>>::value_type;
         pmr::content_vector<ScalarTypeS> realS(k);
         const auto* pointS = realS.data();
         if (division) {
            for (Size i = 0; i < k; i++) {
               realS[i] = ScalarTypeS(1) / vector_in_S[i * dimension_plus_one];
            }
         } else {
            for (Size i = 0; i < k; i++) {
               realS[i] = vector_in_S[i * dimension_plus_one];
            }
         }

         multiple_kernel(m, k, n, data_destination, data_source, pointS);
      }
      return result;
   }
} // namespace TAT
#endif
