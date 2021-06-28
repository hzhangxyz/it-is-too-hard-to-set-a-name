/**
 * \file get_item.hpp
 *
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
#ifndef TAT_GET_ITEM_HPP
#define TAT_GET_ITEM_HPP

#include "../structure/tensor.hpp"
#include "../utility/allocator.hpp"

namespace TAT {

   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   const auto& Tensor<ScalarType, Symmetry, Name>::get_block(const auto& map) const& {
      auto symmetries = pmr::vector<Symmetry>();
      symmetries.reserve(names.size());
      for (const auto& name : names) {
         symmetries.push_back(map_at(map, name));
      }
      return map_at<true>(core->blocks, symmetries);
   }

   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   const ScalarType& Tensor<ScalarType, Symmetry, Name>::get_item(const auto& position) const& {
      auto rank = names.size();
      auto symmetries = pmr::vector<Symmetry>();
      auto scalar_position = pmr::vector<Size>();
      auto dimensions = pmr::vector<Size>();
      symmetries.reserve(rank);
      scalar_position.reserve(rank);
      dimensions.reserve(rank);
      for (auto i = 0; i < rank; i++) {
         const auto& name = names[i];
         auto found = map_find(position, name);
         if (found == position.end()) {
            detail::error("Name not found in position map when finding block and offset");
         }
         const auto& [symmetry, index] = [edge = edges(i)](const auto& point_or_index) {
            if constexpr (std::is_integral_v<std::remove_cvref_t<decltype(point_or_index)>>) {
               return edge.get_point_from_index(point_or_index);
            } else {
               return point_or_index;
            }
         }(found->second);
         symmetries.push_back(symmetry);
         scalar_position.push_back(index);
         dimensions.push_back(edges(i).get_dimension_from_symmetry(symmetry));
      }
      Size offset = 0;
      for (Rank j = 0; j < rank; j++) {
         offset *= dimensions[j];
         offset += scalar_position[j];
      }
      return map_at<true>(core->blocks, symmetries)[offset];
   }
} // namespace TAT
#endif
