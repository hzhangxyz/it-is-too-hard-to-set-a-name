/**
 * \file shrink_and_expand.hpp
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
#ifndef TAT_SHRINK_AND_EXPAND_HPP
#define TAT_SHRINK_AND_EXPAND_HPP

#include "../structure/tensor.hpp"
#include "../utility/timer.hpp"

namespace TAT {
   inline timer expand_guard("expand");

   // TODO 这些都可以优化，不使用contract
   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::expand_implement(const auto& configure, const Name& old_name) const {
      auto timer_guard = expand_guard();
      // using EdgeInfoWithArrowForExpand = std::conditional_t<
      //            Symmetry::length == 0,
      //            std::tuple<Size, Size>,
      //            std::conditional_t<Symmetry::is_fermi_symmetry, std::tuple<Arrow, Symmetry, Size, Size>, std::tuple<Symmetry, Size, Size>>>;
      constexpr bool is_no_symmetry = Symmetry::length == 0;
      constexpr bool is_fermi = Symmetry::is_fermi_symmetry;
      auto new_names = pmr::vector<Name>();
      auto new_edges = pmr::vector<Edge<Symmetry>>();
      auto reserve_size = configure.size() + 1;
      new_names.reserve(reserve_size);
      new_edges.reserve(reserve_size);
      auto total_symmetry = Symmetry();
      Size total_offset = 0;
      for (const auto& [name, information] : configure) {
         new_names.push_back(name);
         if constexpr (is_no_symmetry) {
            const auto& [index, dimension] = information;
            total_offset *= dimension;
            total_offset += index;
            new_edges.push_back({{{Symmetry(), dimension}}});
         } else if constexpr (is_fermi) {
            const auto& [arrow, symmetry, index, dimension] = information;
            total_offset *= dimension;
            total_offset += index;
            new_edges.push_back({arrow, {{symmetry, dimension}}});
         } else {
            const auto& [symmetry, index, dimension] = information;
            total_offset *= dimension;
            total_offset += index;
            new_edges.push_back({{{symmetry, dimension}}});
         }
      }
      auto contract_names = pmr::set<std::pair<Name, Name>>();
      if (old_name != InternalName<Name>::No_Old_Name) {
         contract_names.insert({old_name, old_name});
         new_names.push_back(old_name);
         // 调整使得可以缩并
         auto& old_edge = core->edges[map_at(name_to_index, old_name)];
         if (old_edge.map.size() != 1 || old_edge.map.begin()->second != 1) {
            detail::error("Cannot Expand a Edge which dimension is not one");
         }
         if constexpr (is_no_symmetry) {
            new_edges.push_back({{{Symmetry(), 1}}});
         } else {
            if constexpr (is_fermi) {
               new_edges.push_back({!old_edge.arrow, {{-total_symmetry, 1}}});
            } else {
               new_edges.push_back({{{-total_symmetry, 1}}});
            }
            if (old_edge.map.front().first != total_symmetry) [[unlikely]] {
               detail::error("Cannot Expand to such Edges whose total Symmetry is not Compatible with origin Edge");
            }
         }
      } else {
         if constexpr (!is_no_symmetry) {
            if (total_symmetry != Symmetry()) [[unlikely]] {
               detail::error("Cannot Expand to such Edges whose total Symmetry is not zero");
            }
         }
      }
      auto helper = Tensor<ScalarType, Symmetry, Name>(new_names, new_edges);
      helper.zero();
      helper.core->blocks.begin()->second[total_offset] = 1;
      return contract(helper, std::move(contract_names));
   }

   inline timer shrink_guard("shrink");

   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   Tensor<ScalarType, Symmetry, Name>
   Tensor<ScalarType, Symmetry, Name>::shrink_implement(const auto& configure, const Name& new_name, Arrow arrow) const {
      auto timer_guard = shrink_guard();
      constexpr bool is_no_symmetry = Symmetry::length == 0;
      constexpr bool is_fermi = Symmetry::is_fermi_symmetry;
      auto new_names = pmr::vector<Name>();
      auto new_edges = pmr::vector<Edge<Symmetry>>();
      auto reserve_size = configure.size() + 1;
      new_names.reserve(reserve_size);
      new_edges.reserve(reserve_size);
      auto total_symmetry = Symmetry();
      Size total_offset = 0;
      auto contract_names = pmr::set<std::pair<Name, Name>>();
      for (const auto& name : names) {
         if (auto found_position = map_find(configure, name); found_position != configure.end()) {
            const auto& position = found_position->second;
            Symmetry symmetry;
            Size index;
            if constexpr (is_no_symmetry) {
               index = position;
            } else {
               symmetry = std::get<0>(position);
               index = std::get<1>(position);
               total_symmetry += symmetry;
            }
            const auto& this_edge = core->edges[map_at(name_to_index, name)];
            Size dimension = map_at(this_edge.map, symmetry);
            total_offset *= dimension;
            total_offset += index;
            new_names.push_back(name);
            contract_names.insert({name, name});
            if constexpr (is_fermi) {
               new_edges.push_back({!this_edge.arrow, {{-symmetry, dimension}}});
            } else {
               new_edges.push_back({{{-symmetry, dimension}}});
            }
         }
      }
      if (new_name != InternalName<Name>::No_New_Name) {
         new_names.push_back(new_name);
         if constexpr (is_fermi) {
            new_edges.push_back({arrow, {{total_symmetry, 1}}});
         } else {
            new_edges.push_back({{{total_symmetry, 1}}});
         }
      } else {
         if constexpr (!is_no_symmetry) {
            if (total_symmetry != Symmetry()) [[unlikely]] {
               detail::error("Need to Create a New Edge but Name not set in Slice");
            }
         }
      }
      auto helper = Tensor<ScalarType, Symmetry, Name>(new_names, new_edges);
      helper.zero();
      helper.core->blocks.begin()->second[total_offset] = 1;
      return contract(helper, std::move(contract_names));
   }
} // namespace TAT
#endif
