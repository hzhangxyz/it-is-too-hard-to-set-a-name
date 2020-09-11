/**
 * \file core.hpp
 *
 * Copyright (C) 2019-2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_CORE_HPP
#define TAT_CORE_HPP

#include <vector>

#include "edge.hpp"

namespace TAT {
   /**
    * \brief 用于不初始化的vector的allocator, 仅用于张量数据的存储
    */
   template<class T>
   struct allocator_without_initialize : std::allocator<T> {
      template<class U>
      struct rebind {
         using other = allocator_without_initialize<U>;
      };

      /**
       * \brief 初始化函数, 如果没有参数, 且类型T可以被平凡的析构, 则不做任何初始化操作, 否则进行正常的就地初始化
       * \tparam Args 初始化的参数类型
       * \param pointer 被初始化的值的地址
       * \param arguments 初始化的参数
       */
      template<class... Args>
      void construct([[maybe_unused]] T* pointer, Args&&... arguments) {
         if constexpr (!((sizeof...(arguments) == 0) && (std::is_trivially_destructible_v<T>))) {
            new (pointer) T(arguments...);
         }
      }

      allocator_without_initialize() = default;
      template<class U>
      explicit allocator_without_initialize(allocator_without_initialize<U>) {}
   };

   /**
    * \brief 尽可能不做初始化的vector容器
    * \see allocator_without_initialize
    * \note 为了其他部分与stl兼容性, 仅在张量的数据处使用
    */
   template<class T>
   using vector = std::vector<T, allocator_without_initialize<T>>;

   /**
    * \brief 记录了张量的核心数据的类型, 核心数据指的是除了角标名称之外的信息, 包括边的形状, 以及张量内本身的数据
    * \tparam ScalarType 张量内本身的数据的标量类型
    * \tparam Symmetry 张量所拥有的对称性
    * \note Core的存在是为了让边的名称的重命名节省时间
    */
   template<class ScalarType, class Symmetry>
   struct Core {
      /**
       * \brief 张量的形状, 是边的形状的列表, 列表长度为张量的秩, 每个边是一个对称性值到子边长度的映射表
       * \see Edge
       */
      std::vector<Edge<Symmetry>> edges = {};
      /**
       * \brief 张量内本身的数据, 是对称性列表到数据列表的映射表, 数据列表就是张量内本身的数据,
       * 而对称性列表表示此子块各个子边在各自的边上所对应的对称性值
       */
      std::map<std::vector<Symmetry>, vector<ScalarType>> blocks = {};

      /**
       * \brief 根据边的形状构造张量, 然后根据对称性条件自动构造张量的分块
       * \param initial_edge 边的形状的列表
       * \param auto_reverse 对于费米张量是否自动对含有负对称值的边整个取反
       * \note 使用auto_reverse时, 原则上构造时费米对称性值应该全正或全负, 如果不是这样, 结果会难以理解
       * \note 将会自动删除不出现于数据中的对称性
       */
      Core(std::vector<Edge<Symmetry>> initial_edge, [[maybe_unused]] const bool auto_reverse = false) : edges(std::move(initial_edge)) {
         // 自动翻转边
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            if (auto_reverse) {
               for (auto& edge : edges) {
                  edge.possible_reverse();
               }
            }
         }
         // 生成数据
         auto symmetries_list = initialize_block_symmetries_with_check(edges);
         for (auto& [symmetries, size] : symmetries_list) {
            blocks[symmetries] = vector<ScalarType>(size);
         }
         // 删除不在block中用到的symmetry
         const Rank rank = edges.size();
         auto edge_mark = std::vector<std::map<Symmetry, bool>>();
         edge_mark.reserve(rank);
         for (const auto& edge : edges) {
            auto& this_mark = edge_mark.emplace_back();
            for (const auto& [symmetry, _] : edge.map) {
               this_mark[symmetry] = true;
            }
         }
         for (const auto& [symmetries, _] : blocks) {
            for (Rank i = 0; i < rank; i++) {
               edge_mark[i].at(symmetries[i]) = false;
            }
         }
         for (Rank i = 0; i < rank; i++) {
            for (const auto& [symmetry, flag] : edge_mark[i]) {
               if (flag) {
                  edges[i].map.erase(symmetry);
               }
            }
         }
      }

      Core() = default;
      Core(const Core&) = default;
      Core(Core&&) = default;
      Core& operator=(const Core&) = default;
      Core& operator=(Core&&) = default;
      ~Core() = default;
   };
} // namespace TAT
#endif
