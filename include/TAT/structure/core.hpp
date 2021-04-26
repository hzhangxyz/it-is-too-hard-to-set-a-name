/**
 * \file core.hpp
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
#ifndef TAT_CORE_HPP
#define TAT_CORE_HPP

#include <numeric>
#include <vector>

#include "../TAT.hpp"
#include "../utility/concepts_and_fake_map_set.hpp"
#include "../utility/no_initialize_allocator.hpp"
#include "edge.hpp"

namespace TAT {
   template<typename T, typename Iter>
   auto forward_iterator(Iter it) {
      if constexpr (std::is_rvalue_reference_v<T&&>) {
         return std::make_move_iterator(it);
      } else {
         return it;
      }
   }

   // 使用这个后，如果可以，可以直接移动vector，如果不可以，也会先尝试移动元素
   template<typename Result, typename Vector>
   auto forward_vector(Vector&& v) {
      // 可以移动的话，需要相同，Vector&&需要时右值，也就是Vector正好是Result
      if constexpr (std::is_same_v<Vector, Result>) {
         return Result(std::move(v));
      } else {
         // 可能是initializer_list, 他没有empty()函数
         if (v.size() == 0) {
            return Result();
         } else {
            return Result(forward_iterator<Vector>(v.begin()), forward_iterator<Vector>(v.end()));
         }
      }
   }

   /**
    * \defgroup Tensor
    * @{
    */

   template<is_symmetry Symmetry>
   struct core_edges_t {
      using symmetry_t = Symmetry;
      using edge_t = Edge<symmetry_t>;
      using edge_vector_t = std::vector<edge_t>;

      /**
       * 张量的形状, 是边的形状的列表, 列表长度为张量的秩, 每个边是一个对称性值到子边长度的映射表
       * \see Edge
       */
      edge_vector_t edges = {};

      template<range_of<Edge<Symmetry>> EdgeVector>
      core_edges_t(EdgeVector&& initial_edge, const bool auto_reverse = false) :
            edges(forward_vector<edge_vector_t>(std::forward<EdgeVector>(initial_edge))) {
         check_edge_reverse(auto_reverse);
      }

      core_edges_t() = default;
      core_edges_t(const core_edges_t& other) = default;
      core_edges_t(core_edges_t&& other) = default;
      core_edges_t& operator=(const core_edges_t&) = default;
      core_edges_t& operator=(core_edges_t&&) = default;
      ~core_edges_t() = default;

      void check_edge_reverse([[maybe_unused]] bool auto_reverse) {
         // 自动翻转边
         if constexpr (Symmetry::is_fermi_symmetry) {
            if (auto_reverse) {
               for (auto& edge : edges) {
                  edge.possible_reverse();
               }
            }
         }
      }
   };

   template<is_scalar ScalarType, is_symmetry Symmetry>
   struct core_blocks_t {
      using symmetry_vector_t = std::vector<Symmetry>;
      using content_vector_t = pmr::content_vector<ScalarType>;

      using block_map_t = std::vector<std::pair<symmetry_vector_t, content_vector_t>>;

      no_initialize::vector<ScalarType> storage;
      monotonic_buffer_resource resource;

      /**
       * 张量内本身的数据, 是对称性列表到数据列表的映射表, 数据列表就是张量内本身的数据,
       * 而对称性列表表示此子块各个子边在各自的边上所对应的对称性值
       */
      block_map_t blocks;

      core_blocks_t(std::vector<std::pair<std::vector<Symmetry>, Size>>&& symmetries_list) :
            storage(std::accumulate(
                  symmetries_list.begin(),
                  symmetries_list.end(),
                  0,
                  [&](const Size total_size, const auto& p) {
                     return total_size + p.second;
                  })),
            resource(storage.data(), storage.size() * sizeof(ScalarType)),
            blocks() {
         for (auto&& [symmetries, size] : symmetries_list) {
            // symmetries list一定是右值，肯定可以被移动的
            blocks.push_back({std::move(symmetries), content_vector_t(size, &resource)});
         }
      }

      // 复制构造
      core_blocks_t(const core_blocks_t& other) : storage(other.storage), resource(storage.data(), storage.size() * sizeof(ScalarType)), blocks() {
         for (const auto& [symmetries, block] : other.blocks) {
            blocks.push_back({symmetries, content_vector_t(block.size(), &resource)});
         }
      }
      core_blocks_t(core_blocks_t&& other) :
            storage(std::move(other.storage)),
            resource(storage.data(), storage.size() * sizeof(ScalarType)),
            blocks() {
         for (auto&& [symmetries, block] : other.blocks) {
            blocks.push_back({std::move(symmetries), content_vector_t(block.size(), &resource)});
         }
      }

      core_blocks_t() = delete;
      core_blocks_t& operator=(const core_blocks_t&) = delete;
      core_blocks_t& operator=(core_blocks_t&&) = delete;
   };

   /**
    * 记录了张量的核心数据的类型, 核心数据指的是除了角标名称之外的信息, 包括边的形状, 以及张量内本身的数据
    * \tparam ScalarType 张量内本身的数据的标量类型
    * \tparam Symmetry 张量所拥有的对称性
    * \note Core的存在是为了让边的名称的重命名节省时间
    */
   template<is_scalar ScalarType, is_symmetry Symmetry>
   struct Core : core_edges_t<Symmetry>, core_blocks_t<ScalarType, Symmetry> {
      using base_edges = core_edges_t<Symmetry>;
      using base_blocks = core_blocks_t<ScalarType, Symmetry>;

      using base_blocks::blocks;
      using base_blocks::storage;
      using base_edges::edges;

      /**
       * 根据边的形状构造张量, 然后根据对称性条件自动构造张量的分块
       * \param initial_edge 边的形状的列表
       * \param auto_reverse 对于费米张量是否自动对含有负对称值的边整个取反
       * \note 使用auto_reverse时, 原则上构造时费米对称性值应该全正或全负, 如果不是这样, 结果会难以理解
       * \note 将会自动删除不出现于数据中的对称性
       */
      template<range_of<Edge<Symmetry>> VectorEdge>
      Core(VectorEdge&& initial_edge, const bool auto_reverse = false) :
            base_edges(std::forward<VectorEdge>(initial_edge), auto_reverse),
            base_blocks(initialize_block_symmetries_with_check(edges)) {
         // 删除不在block中用到的symmetry
         if constexpr (Symmetry::length != 0) {
            const Rank rank = edges.size();
            auto edge_mark = std::vector<std::vector<std::pair<Symmetry, bool>>>(rank);
            for (Rank i = 0; i < rank; i++) {
               const auto& edge = edges[i];
               auto& this_mark = edge_mark[i];
               for (const auto& [symmetry, _] : edge.map) {
                  this_mark.push_back({symmetry, false});
               }
            }
            for (const auto& [symmetries, _] : blocks) {
               for (Rank i = 0; i < rank; i++) {
                  map_at(edge_mark[i], symmetries[i]) = true;
               }
            }
            for (Rank i = 0; i < rank; i++) {
               auto& edge = edges[i];
               const auto& this_mark = edge_mark[i];
               const Nums number = edge.map.size();
               Nums k = 0;
               for (Nums j = 0; j < number; j++) {
                  if (this_mark[j].second) {
                     edge.map[k++] = edge.map[j];
                  }
               }
               edge.map.resize(k);
            }
         }
      }

      Core() = delete;
      Core(const Core& other) = default;
      Core(Core&& other) = default;
      Core& operator=(const Core&) = delete;
      Core& operator=(Core&&) = delete;
   };
   /**@}*/
} // namespace TAT
#endif
