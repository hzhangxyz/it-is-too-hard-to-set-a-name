/**
 * \file trace.hpp
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
#ifndef TAT_TRACE_HPP
#define TAT_TRACE_HPP

#include "../structure/tensor.hpp"
#include "../utility/timer.hpp"

namespace TAT {
   inline timer trace_guard("trace");

   // TODO 可以不转置直接trace掉, 但是写起来比较麻烦
   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::trace_implement(const auto& trace_names) const {
      auto timer_guard = trace_guard();

      constexpr bool is_fermi = Symmetry::is_fermi_symmetry;
      auto rank = names.size();
      auto trace_rank = trace_names.size();
      auto free_rank = rank - 2 * trace_rank;
      // 应该转置为a_i = sum_j b_{jji}的形式, 这样局域性最好
      auto traced_names = pmr::set<Name>();
      auto trace_1_names = pmr::vector<Name>();
      auto trace_2_names = pmr::vector<Name>();
      trace_1_names.reserve(trace_rank);
      trace_2_names.reserve(trace_rank);
      // 转置时尽可能保证后面rank的不变
      auto valid_index = pmr::vector<bool>(rank, true);
      for (auto i = rank; i-- > 0;) {
         if (valid_index[i]) {
            const auto& name_to_find = names[i];
            const Name* name_correspond = nullptr;
            for (const auto& [name_1, name_2] : trace_names) {
               if (name_1 == name_to_find) {
                  name_correspond = &name_2;
                  break;
               }
               if (name_2 == name_to_find) {
                  name_correspond = &name_1;
                  break;
               }
            }
            if (name_correspond) {
               // found in trace_names
               // 对于费米子考虑方向, 应是一进一出才合法
               if constexpr (is_fermi) {
                  if (edges(i).arrow) {
                     trace_1_names.push_back(name_to_find);
                     trace_2_names.push_back(*name_correspond);
                  } else {
                     trace_1_names.push_back(*name_correspond);
                     trace_2_names.push_back(name_to_find);
                  }
               } else {
                  trace_1_names.push_back(*name_correspond);
                  trace_2_names.push_back(name_to_find);
                  // trace_1是放在前面的, 而name_correspond也确实在name_to_find前面
               }
               // 统计traced names
               traced_names.insert(name_to_find);
               traced_names.insert(*name_correspond);
               auto index_correspond = map_at(name_to_index, *name_correspond);
               valid_index[index_correspond] = false;
            }
         }
      }
      // 寻找自由脚
      auto result_names = std::vector<Name>();
      auto reverse_names = pmr::set<Name>();
      auto split_plan = pmr::vector<std::tuple<Name, edge_segment_t<Symmetry>>>();
      result_names.reserve(free_rank);
      split_plan.reserve(free_rank);
      for (Rank i = 0; i < rank; i++) {
         const auto& name = names[i];
         if (auto found = traced_names.find(name); found == traced_names.end()) {
            const auto& this_edge = edges(i);
            result_names.push_back(name);
            split_plan.push_back({name, {this_edge.map}});
            if constexpr (is_fermi) {
               if (this_edge.arrow) {
                  reverse_names.insert(name);
               }
            }
         }
      }
      auto merged_tensor = edge_operator_implement(
            std::initializer_list<std::pair<Name, std::initializer_list<std::pair<Name, edge_segment_t<Symmetry>>>>>(),
            reverse_names,
            pmr::map<Name, pmr::vector<Name>>{
                  {InternalName<Name>::Trace_1, std::move(trace_1_names)},
                  {InternalName<Name>::Trace_2, std::move(trace_2_names)},
                  {InternalName<Name>::Trace_3, {result_names.begin(), result_names.end()}}},
            std::vector<Name>{InternalName<Name>::Trace_1, InternalName<Name>::Trace_2, InternalName<Name>::Trace_3},
            false,
            std::initializer_list<Name>(),
            std::initializer_list<Name>(),
            std::initializer_list<Name>(),
            std::initializer_list<Name>(),
            std::initializer_list<std::pair<Name, std::initializer_list<std::pair<Symmetry, Size>>>>());
      // Trace_1和Trace_2一起merge, 而他们相连, 所以要有一个有效, Trace_3等一会会翻转回来, 所以没事
      auto traced_tensor = Tensor<ScalarType, Symmetry, Name>({InternalName<Name>::Trace_3}, {merged_tensor.edges(2)}).zero();
      auto& destination_block = traced_tensor.core->blocks.begin()->second;
      // 应该只有一个边, 所以也只有一个block
      const Size line_size = destination_block.size();

      const auto const_line_size_variant = to_const_integral<Size, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(line_size);
      std::visit(
            [&](const auto& const_line_size) {
               const auto line_size = const_line_size.value();
               for (const auto& [symmetry_1, dimension] : merged_tensor.edges(0).map) {
                  // 而source的形状应该是多个分块对角矩阵, 每个元素是一个向量, 我只需要把正对角的向量们求和
                  auto symmetry_2 = -symmetry_1;
                  auto source_block = map_at<true>(merged_tensor.core->blocks, pmr::vector<Symmetry>{symmetry_1, symmetry_2, Symmetry()});
                  auto dimension_plus_one = dimension + 1;
                  for (Size i = 0; i < dimension; i++) {
                     const ScalarType* __restrict source_data = source_block.data() + dimension_plus_one * i * line_size;
                     ScalarType* __restrict destination_data = destination_block.data();
                     for (Size j = 0; j < line_size; j++) {
                        destination_data[j] += source_data[j];
                     }
                  }
               }
            },
            const_line_size_variant);

      auto result = traced_tensor.edge_operator_implement(
            pmr::map<Name, decltype(split_plan)>{{InternalName<Name>::Trace_3, std::move(split_plan)}},
            reverse_names,
            std::initializer_list<std::pair<Name, std::initializer_list<Name>>>(),
            std::move(result_names),
            false,
            std::initializer_list<Name>(),
            std::initializer_list<Name>(),
            std::initializer_list<Name>(),
            std::initializer_list<Name>(),
            std::initializer_list<std::pair<Name, std::initializer_list<std::pair<Symmetry, Size>>>>());
      return result;
   }
} // namespace TAT
#endif
