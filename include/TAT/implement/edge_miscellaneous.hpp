/**
 * \file edge_miscellaneous.hpp
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
#ifndef TAT_EDGE_MISCELLANEOUS_HPP
#define TAT_EDGE_MISCELLANEOUS_HPP

#include "../structure/tensor.hpp"

namespace TAT {
   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   template<typename ResultName>
   auto Tensor<ScalarType, Symmetry, Name>::edge_rename_implement(const auto& dictionary) const {
      auto result = Tensor<ScalarType, Symmetry, ResultName>{};
      result.core = core;
      result.names.reserve(names.size());
      std::transform(names.begin(), names.end(), std::back_inserter(result.names), [&dictionary](const Name& name) {
         if (auto position = map_find(dictionary, name); position == dictionary.end()) {
            if constexpr (std::is_same_v<ResultName, Name>) {
               return name;
            } else {
               detail::error("New names not found in edge_rename which change type of name");
               return ResultName();
            }
         } else {
            return position->second;
         }
      });
      result.name_to_index = construct_name_to_index<ResultName>(result.names);
      return result;
   }

   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::merge_edge_implement(
         auto merge,
         const bool apply_parity,
         const auto& parity_exclude_name_merge,
         const auto& parity_exclude_name_reverse) const {
      // delete edge from names_before_merge if not exist
      for (auto& [name_after_merge, names_before_merge] : merge) {
         auto new_names_before_merge = decltype(names_before_merge)();
         new_names_before_merge.reserve(names_before_merge.size());
         for (const auto& i : names_before_merge) {
            if (auto found = map_find(name_to_index, i); found != name_to_index.end()) {
               new_names_before_merge.push_back(i);
            }
         }
         names_before_merge.swap(new_names_before_merge);
         // if len(names_before_merge) == 0, it create a trivial edge?
         // 不, 在反转target_name中, 如果原merge列表为空则不产生新的target_name
         // 根据edge_operator中的操作实际上就是消除了这个merge
      }
      std::vector<Name> target_name;
      target_name.reserve(names.size());
      for (auto iterator = names.rbegin(); iterator != names.rend(); ++iterator) {
         // 找到且最后 -> 添加新的
         // 找到不最后 -> 不做事
         // 没找到 -> 添加
         auto found_in_merge = false;
         for (const auto& [name_after_merge, names_before_merge] : merge) {
            if (auto position_in_group = std::find(names_before_merge.begin(), names_before_merge.end(), *iterator);
                position_in_group != names_before_merge.end()) {
               if (*iterator == names_before_merge.back()) {
                  target_name.push_back(name_after_merge);
               }
               found_in_merge = true;
               break;
            }
         }
         if (!found_in_merge) {
            target_name.push_back(*iterator);
         }
      }
      // 翻转target_name
      for (const auto& [name_after_merge, names_before_merge] : merge) {
         if (names_before_merge.empty()) {
            target_name.push_back(name_after_merge);
         }
      }
      std::reverse(target_name.begin(), target_name.end());
      return edge_operator_implement(
            empty_list<std::pair<Name, Name>>(),
            empty_list<std::pair<Name, std::initializer_list<std::pair<Name, edge_map_t<Symmetry>>>>>(),
            empty_list<Name>(),
            merge,
            std::move(target_name),
            apply_parity,
            empty_list<Name>(),
            empty_list<Name>(),
            std::forward<decltype(parity_exclude_name_reverse)>(parity_exclude_name_reverse),
            std::forward<decltype(parity_exclude_name_merge)>(parity_exclude_name_merge),
            empty_list<std::pair<Name, std::initializer_list<std::pair<Symmetry, Size>>>>());
   }

   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   Tensor<ScalarType, Symmetry, Name>
   Tensor<ScalarType, Symmetry, Name>::split_edge_implement(auto split, const bool apply_parity, const auto& parity_exclude_name_split) const {
      // 删除不存在的边
      // 根据edge_operator中的操作, 这应该是多余的, 不在names中的元素会自动忽略
      for (auto iterator = split.begin(); iterator != split.end();) {
         if (auto found = map_find(name_to_index, iterator->first); found == name_to_index.end()) {
            iterator = split.erase(iterator);
         } else {
            ++iterator;
         }
      }
      // 生成target_name
      std::vector<Name> target_name;
      target_name.reserve(names.size()); // 不够, 但是可以减少new的次数
      for (const auto& n : names) {
         if (auto found = map_find(split, n); found != split.end()) {
            for (const auto& edge_after_split : found->second) {
               target_name.push_back(std::get<0>(edge_after_split));
            }
         } else {
            target_name.push_back(n);
         }
      }
      return edge_operator_implement(
            empty_list<std::pair<Name, Name>>(),
            split,
            empty_list<Name>(),
            empty_list<std::pair<Name, std::initializer_list<Name>>>(),
            std::move(target_name),
            apply_parity,
            std::forward<decltype(parity_exclude_name_split)>(parity_exclude_name_split),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<std::pair<Name, std::initializer_list<std::pair<Symmetry, Size>>>>());
   }
} // namespace TAT
#endif
