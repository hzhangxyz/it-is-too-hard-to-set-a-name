/**
 * \file edge_operator.hpp
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
#ifndef TAT_EDGE_OPERATOR_HPP
#define TAT_EDGE_OPERATOR_HPP

#include "../structure/tensor.hpp"
#include "../utility/pmr_resource.hpp"
#include "../utility/timer.hpp"
#include "transpose.hpp"

namespace TAT {
   inline timer transpose_guard("transpose");

   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   [[nodiscard]] Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::edge_operator_implement(
         const auto& rename_map,
         const auto& split_map,
         const auto& reversed_name,
         const auto& merge_map,
         std::vector<Name> new_names,
         const bool apply_parity,
         const auto& parity_exclude_name_split,
         const auto& parity_exclude_name_reversed_before_transpose,
         const auto& parity_exclude_name_reversed_after_transpose,
         const auto& parity_exclude_name_merge,
         const auto& edge_and_symmetries_to_cut_before_all) const {
      auto timer_guard = transpose_guard();
      // step 1: rename and cut
      // step 2: split
      // step 3: reverse
      // step 4: transpose
      // step 5: reverse before merge
      // step 6: merge
      // 前面三步是因果顺序往下推, 第四步转置需要根据merge和最后的name来确定转置方案
      // 除了第一步, 剩下的步骤沿着transpose对称

      // parity_exclude_name的四部分分别是split, reverse, reverse_for_merge, merge
      // name 均为rename后的名称，split取split前， merge取merge后

      // is_fermi
      constexpr auto is_fermi = Symmetry::is_fermi_symmetry;
      constexpr auto is_no_symmetry = Symmetry::length == 0;

      // 前一半分析edge变化的变量表
      //
      //                     rank    name    edge
      // before_split        O       O       D
      // after_split         ↓       O       O
      // before_transpose    ↓       ↑       O
      // at_transpose        O       -       -
      // after_transpose     ↑       ↓       O
      // before_merge        ↑       O       O
      // after_merge         O       O       D
      //
      //                     flag                offset
      // split               rank[after_split]   (symmetry[]->(symmetry, offset))[before_split]
      // merge               rank[after_split]   (symmetry[]->(symmetry, offset))[after_merge]
      //
      // reversed            flag
      // before_transpose    bool[at_transpose]
      // after_transpose     bool[at_transpose]

      // 1. 先将rank, name, edge信息生成出来
      //
      // 1.1 先弄出来 status 0和status 1, 他们在关于transpose对称的操作之前
      //
      // 1.1.1 status 0
      // status 0 origin
      // rank_0 and name_0
      // create edge 0

      // 1.1.2 status 1
      // status 1 rename
      // rank_1 and edge_1
      const Rank rank_before_split = names.size();
      // create name_1
      auto real_name_before_split = std::vector<Name>(); // length = rank_before_split
      if (rename_map.size() != 0) {
         real_name_before_split.reserve(rank_before_split);
         for (Rank i = 0; i < rank_before_split; i++) {
            auto name = names[i];
            if (auto position = map_find(rename_map, name); position != rename_map.end()) {
               real_name_before_split.push_back(position->second);
            } else {
               real_name_before_split.push_back(name);
            }
         }
      }
      const auto& name_before_split = rename_map.size() != 0 ? real_name_before_split : names;

      // 1.2 检查是否不需要做任何操作 -> rename_edge
      // check no change
      if (name_before_split == new_names && split_map.size() == 0 && reversed_name.size() == 0 && merge_map.size() == 0 &&
          edge_and_symmetries_to_cut_before_all.size() == 0) {
         // share the core
         auto result = Tensor<ScalarType, Symmetry, Name>();
         result.names = std::move(new_names);
         result.name_to_index = construct_name_to_index<Name>(result.names);
         result.core = core; // 因为是rename edge所以不拷贝
         // check_valid_name(result.names, result.core->edges.size());
         return result;
      }

      auto real_edge_before_split = std::vector<Edge<Symmetry>>();
      if (edge_and_symmetries_to_cut_before_all.size() != 0) {
         real_edge_before_split.reserve(rank_before_split);
         for (auto i = 0; i < rank_before_split; i++) {
            // 可能的cut, 这里应该是rename前的名称
            if (auto found = map_find(edge_and_symmetries_to_cut_before_all, names[i]); found != edge_and_symmetries_to_cut_before_all.end()) {
               const auto& symmetry_to_cut_dimension = found->second;
               auto& this_edge = real_edge_before_split.emplace_back();
               if constexpr (is_fermi) {
                  this_edge.arrow = core->edges[i].arrow;
               }
               if constexpr (!is_no_symmetry) {
                  this_edge.conjugated = core->edges[i].conjugated;
               }
               for (const auto& [symmetry, dimension] : core->edges[i].map) {
                  if (auto cut_iterator = map_find(symmetry_to_cut_dimension, symmetry); cut_iterator != symmetry_to_cut_dimension.end()) {
                     if (auto new_dimension = cut_iterator->second; new_dimension != 0) {
                        // auto new_dimension = cut_iterator->second;
                        this_edge.map.emplace_back(symmetry, new_dimension < dimension ? new_dimension : dimension);
                     }
                     // 如果new_dimension=0则直接去掉了这个symmetry
                     // 这个会影响leadings, 故只需修改原来算offset处的代码即可, 替换回core->edges
                  } else {
                     this_edge.map.emplace_back(symmetry, dimension);
                  }
               }
               std::ranges::sort(this_edge.map, [](const auto& a, const auto& b) {
                  return a.first < b.first;
               });
            } else {
               real_edge_before_split.push_back(core->edges[i]);
            }
         }
      }
      const auto& edge_before_split = edge_and_symmetries_to_cut_before_all.size() != 0 ? real_edge_before_split : core->edges;

      // 1.3 对称的几个操作中transpose之前的 rank, name, edge
      // status 2 split
      // create name_2 and edge_2 and split_flag
      auto split_flag = pmr::vector<Rank>();
      auto split_offset = pmr::vector<pmr::map<pmr::vector<Symmetry>, std::tuple<Symmetry, Size>>>();
      auto real_name_after_split = std::vector<Name>();
      auto edge_after_split = pmr::vector<EdgePointer<Symmetry>>();
      if (split_map.size() != 0) {
         // 不知道之后的rank, 这些reserve可能不够, 但是没关系, 差一点问题不大, 足够可以减少new的次数了
         split_flag.reserve(rank_before_split);            // rank_at_transpose
         split_offset.reserve(rank_before_split);          // 没问题
         real_name_after_split.reserve(rank_before_split); // rank_at_transpose
         edge_after_split.reserve(rank_before_split);      // rank_at_transpose
         for (Rank position_before_split = 0, total_split_index = 0; position_before_split < rank_before_split; position_before_split++) {
            if (auto position = map_find(split_map, name_before_split[position_before_split]); position != split_map.end()) {
               bool conjugated = edge_before_split[position_before_split].conjugated;
               const auto& this_split_begin_position_in_edge_after_split = edge_after_split.size();
               // split后的edge合法性并不能保证, 需要调用者保证
               for (const auto& [split_name, split_edge] : position->second) {
                  real_name_after_split.push_back(split_name);
                  if constexpr (is_fermi) {
                     edge_after_split.push_back({edge_before_split[position_before_split].arrow, split_edge.map, conjugated});
                  } else if constexpr (!is_no_symmetry) {
                     edge_after_split.push_back({split_edge.map, conjugated});
                  } else {
                     edge_after_split.push_back({split_edge.map});
                  }
                  split_flag.push_back(total_split_index);
               }
               const auto edge_list_after_split = edge_after_split.data() + this_split_begin_position_in_edge_after_split;
               const auto split_rank = edge_after_split.size() - this_split_begin_position_in_edge_after_split;
               // loop between begin and end, get a map push_Back into split_offset
               // this map is sym -> [sym] -> offset
               auto& this_offset = split_offset.emplace_back();
               auto offset_bank = pmr::map<Symmetry, Size>();
               for (const auto& [sym, dim] : edge_before_split[position_before_split].map) {
                  // 只需要symmetry信息, edge_before_split和core->edge一样
                  offset_bank[sym] = 0;
               }
               auto accumulated_symmetries = pmr::vector<Symmetry>(split_rank);
               auto accumulated_dimensions = pmr::vector<Size>(split_rank);
               auto current_symmetries = pmr::vector<Symmetry>(split_rank);
               loop_edge<polymorphic_allocator, true>(
                     edge_list_after_split,
                     split_rank,
                     [&this_offset]() {
                        this_offset[pmr::vector<Symmetry>{}] = {Symmetry(), 0};
                     },
                     []() {},
                     [&](const auto& symmetry_iterator_list, Rank minimum_changed) {
                        for (auto i = minimum_changed; i < split_rank; i++) {
                           const auto& symmetry_iterator = symmetry_iterator_list[i];
                           accumulated_symmetries[i] = symmetry_iterator->first + (i ? accumulated_symmetries[i - 1] : Symmetry());
                           accumulated_dimensions[i] = symmetry_iterator->second * (i ? accumulated_dimensions[i - 1] : 1);
                           // do not check dim=0, because in constructor, it didn't check
                           current_symmetries[i] = symmetry_iterator->first;
                        }
                        auto target_symmetry = accumulated_symmetries.back();
                        auto target_dimension = accumulated_dimensions.back();
                        // 这是可能不存在的, 不存在的情况是因为此target_symmetry没有有效block
                        // split后的各个symmetry指向了一个不存在的split前symmetry
                        if (auto found = offset_bank.find(target_symmetry); found != offset_bank.end()) {
                           this_offset[current_symmetries] = {target_symmetry, found->second};
                           found->second += target_dimension;
                        }
                        return split_rank;
                     });
               total_split_index++;
            } else {
               real_name_after_split.push_back(name_before_split[position_before_split]);
               if constexpr (is_fermi) {
                  edge_after_split.push_back(
                        {edge_before_split[position_before_split].arrow,
                         edge_before_split[position_before_split].map,
                         edge_before_split[position_before_split].conjugated});
               } else if constexpr (!is_no_symmetry) {
                  edge_after_split.push_back({edge_before_split[position_before_split].map, edge_before_split[position_before_split].conjugated});
               } else {
                  edge_after_split.push_back({edge_before_split[position_before_split].map});
               }
               split_flag.push_back(total_split_index++);
               split_offset.emplace_back();
               // auto& this_offset = split_offset.emplace_back();
               // 后面会判断如果没有实际上的split则不使用this_offset
               // for (const auto& [symmetry, dimension] : edge_before_split[position_before_split].map) {
               //   this_offset[{symmetry}] = {symmetry, 0};
               // }
            }
         }
      } else {
         edge_after_split.reserve(rank_before_split);
         split_flag.reserve(rank_before_split);
         split_offset.reserve(rank_before_split);
         for (auto i = 0; i < rank_before_split; i++) {
            const auto& edge = edge_before_split[i];
            if constexpr (is_fermi) {
               edge_after_split.push_back({edge.arrow, edge.map, edge.conjugated});
            } else if constexpr (!is_no_symmetry) {
               edge_after_split.push_back({edge.map, edge.conjugated});
            } else {
               edge_after_split.push_back({edge.map});
            }
            split_flag.push_back(i);
            // 后面会判断如果没有实际上的split则不使用this_offset, 因为全部没有split所以甚至不需要emplace_back
            // auto& this_offset = split_offset.emplace_back();
            // for (const auto& [symmetry, dimension] : edge.map) {
            //    this_offset[{symmetry}] = {symmetry, 0};
            // }
         }
      }
      const auto& name_after_split = split_map.size() != 0 ? real_name_after_split : name_before_split;
      // rank_2
      const Rank rank_at_transpose = name_after_split.size();

      // status 3 reverse
      // rank_3 and name_3
      // create reversed_flag_src and edge_3
      auto reversed_before_transpose_flag = pmr::vector<bool>(); // length = rank_at_transpose
      auto fermi_edge_before_transpose = pmr::vector<EdgePointer<Symmetry>>();
      if constexpr (is_fermi) {
         if (reversed_name.size() != 0) {
            reversed_before_transpose_flag.reserve(rank_at_transpose);
            fermi_edge_before_transpose.reserve(rank_at_transpose);
            for (auto i = 0; i < rank_at_transpose; i++) {
               fermi_edge_before_transpose.push_back(edge_after_split[i]);
               if (set_find(reversed_name, name_after_split[i]) != reversed_name.end()) {
                  fermi_edge_before_transpose.back().arrow ^= true;
                  reversed_before_transpose_flag.push_back(true);
               } else {
                  reversed_before_transpose_flag.push_back(false);
               }
            }
         } else {
            reversed_before_transpose_flag = pmr::vector<bool>(rank_at_transpose, false);
         }
      }
      const auto& edge_before_transpose = is_fermi && reversed_name.size() != 0 ? fermi_edge_before_transpose : edge_after_split;

      // create res names
      auto result = Tensor<ScalarType, Symmetry, Name>();
      result.names = std::move(new_names);
      result.name_to_index = construct_name_to_index<Name>(result.names);

      // 1.4 transpose之后的rank, name

      // status 4 transpose and status 6 merge and status 5 reverse before merge
      // 逆序确定name, 再顺序确定edge和data
      // name and rank
      // name_6 and rank_6
      const auto& name_after_merge = result.names;
      const Rank rank_after_merge = name_after_merge.size();
      // create merge_flag and name_5
      auto merge_flag = pmr::vector<Rank>();
      auto real_name_before_merge = std::vector<Name>();
      if (merge_map.size() != 0) {
         merge_flag.reserve(rank_at_transpose);
         real_name_before_merge.reserve(rank_at_transpose);
         for (Rank position_after_merge = 0, total_merge_index = 0; position_after_merge < rank_after_merge; position_after_merge++) {
            const auto& merged_name = name_after_merge[position_after_merge];
            if (auto position = map_find(merge_map, merged_name); position != merge_map.end()) {
               for (const auto& merging_names : position->second) {
                  real_name_before_merge.push_back(merging_names);
                  merge_flag.push_back(total_merge_index);
               }
               total_merge_index++;
            } else {
               real_name_before_merge.push_back(merged_name);
               merge_flag.push_back(total_merge_index++);
            }
         }
      } else {
         merge_flag.reserve(rank_after_merge);
         for (auto i = 0; i < rank_after_merge; i++) {
            merge_flag.push_back(i);
         }
      }
      const auto& name_before_merge = merge_map.size() != 0 ? real_name_before_merge : name_after_merge;
      // rank_4 and name_5 and rank_5
      // name build by merge may contain some edge not exist
      if (rank_at_transpose != name_before_merge.size()) {
         TAT_error("Tensor to transpose with Different Rank");
      }

      // 1.5 转置方案
      // to be easy, create name_to_index for name_3
      auto name_to_index_after_split = construct_name_to_index<const Name>(name_after_split);
      // create plan of two way
      auto plan_source_to_destination = pmr::vector<Rank>(rank_at_transpose);
      auto plan_destination_to_source = pmr::vector<Rank>(rank_at_transpose);

      // edge
      // create edge_4
      auto edge_after_transpose = pmr::vector<EdgePointer<Symmetry>>();
      edge_after_transpose.reserve(rank_at_transpose);
      for (auto i = 0; i < rank_at_transpose; i++) {
         if (auto found = map_find(name_to_index_after_split, name_before_merge[i]); found != name_to_index_after_split.end()) {
            plan_destination_to_source[i] = found->second;
         } else {
            TAT_error("Tensor to transpose with incompatible name list");
         }
         plan_source_to_destination[plan_destination_to_source[i]] = i;
         edge_after_transpose.push_back(edge_before_transpose[plan_destination_to_source[i]]);
      }
      // 1.6 考虑转置后的edge
      // reverse 和 merge 需要放在一起考虑, 如果fermi, 那么reverse的edge在前一个edge基础上修改, 否则直接是上一个引用

      // the following code is about merge
      // dealing with edge_5 and res_edge and reversed_flag_dst

      // prepare edge_5
      // if no merge, edge_5 is reference of edge_4, else is copy of edge_4
      auto fermi_edge_before_merge = pmr::vector<EdgePointer<Symmetry>>();
      if constexpr (is_fermi) {
         if (merge_map.size() != 0) {
            fermi_edge_before_merge.reserve(rank_at_transpose);
            for (const auto& edge : edge_after_transpose) {
               fermi_edge_before_merge.push_back(edge);
            }
         }
      }
      auto& edge_before_merge = is_fermi && merge_map.size() != 0 ? fermi_edge_before_merge : edge_after_transpose;
      // fermi无翻转，转置就是merge前

      // prepare reversed_flag_dst
      auto reversed_after_transpose_flag = pmr::vector<bool>();
      // 这个reversed_after_transpose不可以事先通过if empty来优化, 除非无merge
      // prepare res_edge
      // res_edge means edge_6 but type is different
      auto result_edge = pmr::vector<Edge<Symmetry>>();
      auto merge_offset = pmr::vector<pmr::map<pmr::vector<Symmetry>, std::tuple<Symmetry, Size>>>();
      if (merge_map.size() != 0) {
         if constexpr (is_fermi) {
            reversed_after_transpose_flag.reserve(rank_at_transpose);
         }
         result_edge.reserve(rank_after_merge);
         merge_offset.reserve(rank_after_merge);
         for (Rank position_after_merge = 0, start_of_merge = 0, end_of_merge = 0; position_after_merge < rank_after_merge; position_after_merge++) {
            // [start, end) need be merged
            while (end_of_merge < rank_at_transpose && merge_flag[end_of_merge] == position_after_merge) {
               end_of_merge++;
            }
            // arrow begin
            Arrow arrow;
            bool arrow_fixed = false;
            if constexpr (is_fermi) {
               for (auto merge_group_position = start_of_merge; merge_group_position < end_of_merge; merge_group_position++) {
                  if (edge_before_merge[merge_group_position].arrow_valid()) {
                     if (arrow_fixed) {
                        if (arrow == edge_before_merge[merge_group_position].arrow) {
                           reversed_after_transpose_flag.push_back(false);
                        } else {
                           edge_before_merge[merge_group_position].arrow ^= true;
                           reversed_after_transpose_flag.push_back(true);
                        }
                     } else {
                        arrow_fixed = true;
                        arrow = edge_before_merge[merge_group_position].arrow;
                        reversed_after_transpose_flag.push_back(false);
                     }
                  } else {
                     reversed_after_transpose_flag.push_back(false);
                  }
               }
            }
            // arrow end

            // merge edge begin
            auto merged_edge = pmr::map<Symmetry, Size>();
            auto& this_offset = merge_offset.emplace_back();

            const Rank merge_rank = end_of_merge - start_of_merge;
            auto accumulated_symmetries = pmr::vector<Symmetry>(merge_rank);
            auto accumulated_dimensions = pmr::vector<Size>(merge_rank);
            auto current_symmetries = pmr::vector<Symmetry>(merge_rank);

            if (merge_rank != 1) {
               loop_edge<polymorphic_allocator, true>(
                     edge_before_merge.data() + start_of_merge,
                     merge_rank,
                     [&merged_edge, &this_offset]() {
                        merged_edge[Symmetry()] = 1;
                        this_offset[pmr::vector<Symmetry>{}] = {Symmetry(), 0};
                     },
                     []() {},
                     [&](const auto& symmetry_iterator_list, const Rank minimum_changed) {
                        for (auto i = minimum_changed; i < merge_rank; i++) {
                           const auto& symmetry_iterator = symmetry_iterator_list[i];
                           accumulated_symmetries[i] = symmetry_iterator->first + (i ? accumulated_symmetries[i - 1] : Symmetry());
                           accumulated_dimensions[i] = symmetry_iterator->second * (i ? accumulated_dimensions[i - 1] : 1);
                           // do not check dim=0, because in constructor, i didn't check
                           current_symmetries[i] = symmetry_iterator->first;
                        }
                        auto target_symmetry = accumulated_symmetries.back();
                        this_offset[current_symmetries] = {target_symmetry, merged_edge[target_symmetry]};
                        merged_edge[target_symmetry] += accumulated_dimensions.back();
                        return merge_rank;
                     });
               auto& real_merged_edge = result_edge.emplace_back(merged_edge);
               if constexpr (is_fermi) {
                  real_merged_edge.arrow = arrow;
               }
            } else {
               const auto& target_edge = edge_before_merge[start_of_merge];
               auto& real_merged_edge = result_edge.emplace_back(target_edge.map);
               if constexpr (is_fermi) {
                  real_merged_edge.arrow = target_edge.arrow;
               }
               // 不需要修改this_offset因为后面会判断
            }
            // merge edge end
            start_of_merge = end_of_merge;
         }
      } else {
         if constexpr (is_fermi) {
            reversed_after_transpose_flag = pmr::vector<bool>(rank_at_transpose, false);
         }
         result_edge.reserve(rank_after_merge);
         for (auto i = 0; i < rank_after_merge; i++) {
            // 没有merge就没有reversed after transpose, 这个时候auto& &edge_before_merge = edge_after_transpose
            // edge_before_merge也不需要修改的
            const auto& edge = edge_before_merge[i];
            if constexpr (is_fermi) {
               result_edge.push_back({edge.arrow, edge.map});
            } else {
               result_edge.push_back({edge.map});
            }
            // 后面会判断如果没有实际上的merge则不使用this_offset, 因为全部没有merge所以甚至不需要emplace_back
            // auto& this_offset = merge_offset.emplace_back();
            // for (const auto& [symmetry, dimension] : edge.map) {
            //    this_offset[{symmetry}] = {symmetry, 0};
            // }
         }
      }
      // the code above is dealing with edge_5 and res_Edge and reversed_flag_dst

      // put res_edge into res
      result.core = std::make_shared<Core<ScalarType, Symmetry>>(std::move(result_edge));
      check_valid_name<Name>(result.names, result.core->edges.size());
      // edge_6
      const auto& edge_after_merge = result.core->edges;
      // 2. 开始分析data如何移动

      // 后一半分析data变化的变量表
      //
      // before->source          symmetry[at_transpose]->(symmetry[before_split], offset[before_split])
      // after->destination      symmetry[at_transpose]->(symmetry[after_merge], offset[after_merge])
      //
      // marks:
      // auto split_flag_mark
      // reversed_before_transpose_flag_mark
      // reversed_after_transpose_flag_mark
      // merge_flag_mark

      using MapFromTransposeToSourceDestination = pmr::map<pmr::vector<Symmetry>, std::tuple<pmr::vector<Symmetry>, pmr::vector<Size>>>;
      auto data_before_transpose_to_source = MapFromTransposeToSourceDestination();
      if (edge_and_symmetries_to_cut_before_all.size() != 0 || split_map.size() != 0 || (is_fermi && reversed_name.size() != 0)) {
         // 需要使用reversed前的symmetry，所以
         // 1. 上面判断了是否reversed为空，不然else中的edge不正确 2.下面使用edge_after_split而不是edge_before_transpose
         for (auto& [symmetries_before_transpose, size] : initialize_block_symmetries_with_check<polymorphic_allocator>(edge_after_split)) {
            // convert sym -> target_sym and offsets
            // and add to map
            auto symmetries = pmr::vector<Symmetry>();
            auto offsets = pmr::vector<Size>();
            symmetries.reserve(rank_before_split);
            offsets.reserve(rank_before_split);
            bool success = true;
            for (Rank position_before_split = 0, position_after_split = 0; position_before_split < rank_before_split; position_before_split++) {
               // [start, end) be merged
               auto split_group_symmetries = pmr::vector<Symmetry>(); // 长度为单个split后edge数
               while (position_after_split < rank_at_transpose && split_flag[position_after_split] == position_before_split) {
                  split_group_symmetries.push_back(symmetries_before_transpose[position_after_split]);
                  position_after_split++;
               }
               // 如果没有split, 则不使用split_offset
               if (split_group_symmetries.size() != 1) {
                  if (auto found = split_offset[position_before_split].find(split_group_symmetries);
                      found != split_offset[position_before_split].end()) {
                     const auto& [this_symmetry, this_offset] = found->second;
                     symmetries.push_back(this_symmetry);
                     offsets.push_back(this_offset);
                  } else {
                     success = false;
                     break;
                  }
               } else {
                  symmetries.push_back(split_group_symmetries.front());
                  offsets.push_back(0);
               }
            }
            if (success) {
               data_before_transpose_to_source[symmetries_before_transpose] = {std::move(symmetries), std::move(offsets)};
            }
         }
      } else {
         for (const auto& [symmetries, block] : core->blocks) {
            data_before_transpose_to_source[pmr::vector<Symmetry>{symmetries.begin(), symmetries.end()}] = {
                  pmr::vector<Symmetry>{symmetries.begin(), symmetries.end()},
                  pmr::vector<Size>(rank_before_split, 0)};
         }
      }
      auto data_after_transpose_to_destination = MapFromTransposeToSourceDestination();
      if (merge_map.size() != 0) {
         for (auto& [symmetries_after_transpose, size] : initialize_block_symmetries_with_check<polymorphic_allocator>(edge_before_merge)) {
            // convert sym -> target_sym and offsets
            // and add to map
            auto symmetries = pmr::vector<Symmetry>();
            auto offsets = pmr::vector<Size>();
            symmetries.reserve(rank_after_merge);
            offsets.reserve(rank_after_merge);
            bool success = true;
            for (Rank position_after_merge = 0, position_before_merge = 0; position_after_merge < rank_after_merge; position_after_merge++) {
               // [start, end) be merged
               auto merge_group_symmetries = pmr::vector<Symmetry>(); // 长度为单个merge前edge数
               while (position_before_merge < rank_at_transpose && merge_flag[position_before_merge] == position_after_merge) {
                  merge_group_symmetries.push_back(symmetries_after_transpose[position_before_merge]);
                  position_before_merge++;
               }
               if (merge_group_symmetries.size() != 1) {
                  if (auto found = merge_offset[position_after_merge].find(merge_group_symmetries);
                      found != merge_offset[position_after_merge].end()) {
                     const auto& [this_symmetry, this_offset] = found->second;
                     symmetries.push_back(this_symmetry);
                     offsets.push_back(this_offset);
                  } else {
                     success = false;
                     break;
                  }
               } else {
                  symmetries.push_back(merge_group_symmetries.front());
                  offsets.push_back(0);
               }
            }
            if (success) {
               data_after_transpose_to_destination[symmetries_after_transpose] = {std::move(symmetries), std::move(offsets)};
            }
         }
      } else {
         for (const auto& [symmetries, block] : result.core->blocks) {
            data_after_transpose_to_destination[pmr::vector<Symmetry>{symmetries.begin(), symmetries.end()}] = {
                  pmr::vector<Symmetry>{symmetries.begin(), symmetries.end()},
                  pmr::vector<Size>(rank_after_merge, 0)};
         }
      }

      // 3. 4 marks
      auto split_flag_mark = pmr::vector<bool>();
      auto reversed_before_transpose_flag_mark = pmr::vector<bool>();
      auto reversed_after_transpose_flag_mark = pmr::vector<bool>();
      auto merge_flag_mark = pmr::vector<bool>();
      if constexpr (is_fermi) {
         split_flag_mark.reserve(rank_before_split);
         reversed_before_transpose_flag_mark.reserve(rank_at_transpose);
         reversed_after_transpose_flag_mark.reserve(rank_at_transpose);
         merge_flag_mark.reserve(rank_after_merge);
         // true => 应用parity
         if (apply_parity) {
            // 默认应用, 故应用需不在exclude中, 即find==end
            for (auto i = 0; i < rank_before_split; i++) {
               split_flag_mark.push_back(set_find(parity_exclude_name_split, name_before_split[i]) == parity_exclude_name_split.end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_before_transpose_flag_mark.push_back(
                     set_find(parity_exclude_name_reversed_before_transpose, name_after_split[i]) ==
                     parity_exclude_name_reversed_before_transpose.end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_after_transpose_flag_mark.push_back(
                     set_find(parity_exclude_name_reversed_after_transpose, name_before_merge[i]) ==
                     parity_exclude_name_reversed_after_transpose.end());
            }
            for (auto i = 0; i < rank_after_merge; i++) {
               merge_flag_mark.push_back(set_find(parity_exclude_name_merge, name_after_merge[i]) == parity_exclude_name_merge.end());
            }
         } else {
            for (auto i = 0; i < rank_before_split; i++) {
               split_flag_mark.push_back(set_find(parity_exclude_name_split, name_before_split[i]) != parity_exclude_name_split.end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_before_transpose_flag_mark.push_back(
                     set_find(parity_exclude_name_reversed_before_transpose, name_after_split[i]) !=
                     parity_exclude_name_reversed_before_transpose.end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_after_transpose_flag_mark.push_back(
                     set_find(parity_exclude_name_reversed_after_transpose, name_before_merge[i]) !=
                     parity_exclude_name_reversed_after_transpose.end());
            }
            for (auto i = 0; i < rank_after_merge; i++) {
               merge_flag_mark.push_back(set_find(parity_exclude_name_merge, name_after_merge[i]) != parity_exclude_name_merge.end());
            }
         }
      }
      // 可能会产生空的无源分块
      if constexpr (Symmetry::length != 0) {
         // TODO 这部分可以优化, call zero的条件可以更加苛刻
         result.zero();
      }
      // 缩并时在这里产生无源分块是很正常的事, 这正是对称性ansatz所假设的, 每个张量都是对称守恒的
      // 但是缩并后, 他便可以表示两个对称性不守恒的张量的乘积, 或者说缩并后表达能力变强了, 所以会产生多余的0
      // 5. main copy loop
      for (const auto& [symmetries_before_transpose, source_symmetries_and_offsets] : data_before_transpose_to_source) {
         // 最终转置的变量表
         //                   source                     destination
         // symmetries        symmetry[before_split]     symmetry[after_merge]
         // offsets           offset[before_split]       offset[after_merge]
         //
         //                   before_transpose           after_transpose
         // symmetries        symmetry[at_transpose]     symmetry[at_transpose]
         // dimensions        dimension[at_transpose]    dimension[at_transpose]
         //
         //                   source         destination
         // total_offset      O              O
         // block             O              O
         //
         // leadings
         // leadings_of_source[before_split] -> leadings_before_transpose[at_transpose]
         // leadings_of_destination[after_merge] -> leadings_after_transpose[at_transpose]
         const auto& [source_symmetries, source_offsets] = source_symmetries_and_offsets;

         auto symmetries_after_transpose = pmr::vector<Symmetry>(rank_at_transpose);
         auto dimensions_before_transpose = pmr::vector<Size>(rank_at_transpose);
         auto dimensions_after_transpose = pmr::vector<Size>(rank_at_transpose);
         Size total_size = 1;
         for (auto i = 0; i < rank_at_transpose; i++) {
            auto dimension = map_at(edge_before_transpose[i].map, symmetries_before_transpose[i]);
            dimensions_before_transpose[i] = dimension;
            dimensions_after_transpose[plan_source_to_destination[i]] = dimension;
            symmetries_after_transpose[plan_source_to_destination[i]] = symmetries_before_transpose[i];
            total_size *= dimension;
         }

         const auto& [destination_symmetries, destination_offsets] = data_after_transpose_to_destination.at(symmetries_after_transpose);

         // 已经获得四个symmetry, 两个offset, 两个dimension
         // 现在获得leadings和开始点

         const auto& source_block = map_at<true>(core->blocks, source_symmetries);
         auto& destination_block = map_at<true>(result.core->blocks, destination_symmetries);

         Size total_source_offset = 0;
         for (auto i = 0; i < rank_before_split; i++) {
            // 这里将edge_before_split换为core->edges
            total_source_offset *= map_at(core->edges[i].map, source_symmetries[i]);
            total_source_offset += source_offsets[i];
         }
         Size total_destination_offset = 0;
         for (auto i = 0; i < rank_after_merge; i++) {
            total_destination_offset *= map_at(edge_after_merge[i].map, destination_symmetries[i]);
            total_destination_offset += destination_offsets[i];
         }

         auto leadings_of_source = pmr::vector<Size>(rank_before_split);
         for (auto i = rank_before_split; i-- > 0;) {
            if (i == rank_before_split - 1) {
               leadings_of_source[i] = 1;
            } else {
               // 这里将edge_before_split换为core->edges
               leadings_of_source[i] = leadings_of_source[i + 1] * map_at(core->edges[i + 1].map, source_symmetries[i + 1]);
            }
         }
         auto leadings_before_transpose = pmr::vector<Size>(rank_at_transpose);
         for (auto i = rank_at_transpose; i-- > 0;) {
            if (i != rank_at_transpose - 1 && split_flag[i] == split_flag[i + 1]) {
               leadings_before_transpose[i] = leadings_before_transpose[i + 1] * dimensions_before_transpose[i + 1];
               // dimensions_before_transpose[i + 1] == edge_before_transpose[i + 1].map.at(symmetries_before_transpose[i + 1]);
            } else {
               leadings_before_transpose[i] = leadings_of_source[split_flag[i]];
            }
         }

         auto leadings_of_destination = pmr::vector<Size>(rank_after_merge);
         for (auto i = rank_after_merge; i-- > 0;) {
            if (i == rank_after_merge - 1) {
               leadings_of_destination[i] = 1;
            } else {
               leadings_of_destination[i] = leadings_of_destination[i + 1] * map_at(edge_after_merge[i + 1].map, destination_symmetries[i + 1]);
            }
         }
         auto leadings_after_transpose = pmr::vector<Size>(rank_at_transpose);
         for (auto i = rank_at_transpose; i-- > 0;) {
            if (i != rank_at_transpose - 1 && merge_flag[i] == merge_flag[i + 1]) {
               leadings_after_transpose[i] = leadings_after_transpose[i + 1] * dimensions_after_transpose[i + 1];
               // dimensions_after_transpose[i + 1] == edge_after_transpose[i + 1].map.at(symmetries_after_transpose[i + 1]);
            } else {
               leadings_after_transpose[i] = leadings_of_destination[merge_flag[i]];
            }
         }

         // parity
         auto parity = false;
         if constexpr (is_fermi) {
            parity ^= Symmetry::get_transpose_parity(symmetries_before_transpose, plan_source_to_destination);

            parity ^= Symmetry::get_reverse_parity(symmetries_before_transpose, reversed_before_transpose_flag, reversed_before_transpose_flag_mark);
            parity ^= Symmetry::get_split_merge_parity(symmetries_before_transpose, split_flag, split_flag_mark);
            parity ^= Symmetry::get_reverse_parity(symmetries_after_transpose, reversed_after_transpose_flag, reversed_after_transpose_flag_mark);
            parity ^= Symmetry::get_split_merge_parity(symmetries_after_transpose, merge_flag, merge_flag_mark);
         }

         do_transpose(
               source_block.data() + total_source_offset,
               destination_block.data() + total_destination_offset,
               plan_source_to_destination,
               plan_destination_to_source,
               dimensions_before_transpose,
               dimensions_after_transpose,
               leadings_before_transpose,
               leadings_after_transpose,
               rank_at_transpose,
               total_size,
               parity);
      }

      return result;
   }
} // namespace TAT
#endif
