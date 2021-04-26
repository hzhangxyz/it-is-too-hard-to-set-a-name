/**
 * \file edge.hpp
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
#ifndef TAT_EDGE_HPP
#define TAT_EDGE_HPP

#include <algorithm>
#include <map>
#include <ranges>
#include <set>

#include "../TAT.hpp"
#include "../utility/concepts_and_fake_map_set.hpp"
#include "symmetry.hpp"

namespace TAT {
   /** \defgroup Edge
    * @{
    */
   template<is_symmetry Symmetry, bool is_pointer = false>
   struct edge_map_t {
      static constexpr bool is_not_pointer = !is_pointer;

      using pair_initializer_list_t = std::initializer_list<std::pair<Symmetry, Size>>;
      using symmetry_initializer_list_t = std::initializer_list<Symmetry>;

      using symmetry_t = Symmetry;
      using map_t = std::vector<std::pair<Symmetry, Size>>;
      std::conditional_t<is_pointer, const map_t&, map_t> map;

      edge_map_t() = default;
      edge_map_t(const edge_map_t& edge) = default;
      edge_map_t(edge_map_t&& edge) = default;
      edge_map_t& operator=(const edge_map_t&) = default;
      edge_map_t& operator=(edge_map_t&&) noexcept = default;
      ~edge_map_t() = default;

      // map不可以由非const的initializer list创建
      template<map_like_range_of<Symmetry, Size> MapSymmetrySize = pair_initializer_list_t>
         requires(is_not_pointer)
      edge_map_t(MapSymmetrySize&& map_) : map(map_.begin(), map_.end()) {
         if constexpr (!findable<MapSymmetrySize, Symmetry>) {
            std::ranges::sort(map, [](const auto& a, const auto& b) {
               return a.first < b.first;
            });
         }
      }

      /**
       * 由一些对称性的集合构造, 意味着每一个对称性对应的维度都为1
       */
      template<range_of<Symmetry> SetSymmetry = symmetry_initializer_list_t>
         requires(is_not_pointer)
      edge_map_t(const SetSymmetry& symmetries) {
         for (const auto& symmetry : symmetries) {
            map.emplace_back(symmetry, 1);
         }
         if constexpr (!findable<SetSymmetry, Symmetry>) {
            std::ranges::sort(map, [](const auto& a, const auto& b) {
               return a.first < b.first;
            });
         }
      }

      /**
       * 构造一个平凡的边, 仅含一个对称性
       */
      template<typename = void>
         requires(is_not_pointer)
      edge_map_t(const Size dimension) : map({{Symmetry(), dimension}}) {}

      template<typename = void>
         requires(is_pointer)
      edge_map_t(const map_t& map_) : map(map_) {}
   };

   // used only when sampling merged edge, use this to modify order of edges
   // map is always increasing, this variable only works when looping edge
   struct edge_nosymmetry_conjugated_t {
      static constexpr bool conjugated = false;
      edge_nosymmetry_conjugated_t() {}
      edge_nosymmetry_conjugated_t(bool) {}
   };
   struct edge_symmetry_conjugated_t {
      bool conjugated;
      edge_symmetry_conjugated_t() : conjugated(0) {}
      edge_symmetry_conjugated_t(bool conjugated) : conjugated(conjugated) {}
   };
   template<bool is_nosymmetry>
   using edge_conjugated_t = std::conditional_t<is_nosymmetry, edge_nosymmetry_conjugated_t, edge_symmetry_conjugated_t>;

   struct edge_bose_arrow_t {
      static constexpr Arrow arrow = 0;
      edge_bose_arrow_t() {}
      edge_bose_arrow_t(Arrow) {}
   };
   struct edge_fermi_arrow_t {
      Arrow arrow;
      edge_fermi_arrow_t() : arrow(0) {}
      edge_fermi_arrow_t(Arrow arrow) : arrow(arrow) {}
   };
   template<bool is_fermi>
   using edge_arrow_t = std::conditional_t<is_fermi, edge_fermi_arrow_t, edge_bose_arrow_t>;

   /**
    * 张量的边的形状的类型, 是一个Symmetry到Size的映射表, 如果是费米对称性, 还会含有一个箭头方向
    * \tparam Symmetry 张量所拥有的对称性
    * \tparam is_pointer map是否为引用而非真是存储着数据的伪边
    */
   template<is_symmetry Symmetry, bool is_pointer = false>
   struct Edge : edge_map_t<Symmetry, is_pointer>, edge_conjugated_t<Symmetry::length == 0>, edge_arrow_t<Symmetry::is_fermi_symmetry> {
      using base_map_t = edge_map_t<Symmetry, is_pointer>;
      using base_conjugated_t = edge_conjugated_t<Symmetry::length == 0>;
      using base_arrow_t = edge_arrow_t<Symmetry::is_fermi_symmetry>;

      using base_arrow_t::arrow;
      using base_conjugated_t::conjugated;
      using base_map_t::map;

      Edge() = default;
      Edge(const Edge&) = default;
      Edge(Edge&&) noexcept = default;
      Edge& operator=(const Edge&) = default;
      Edge& operator=(Edge&&) noexcept = default;
      ~Edge() = default;

      // 这里不可以用typename ... Args不然会和initialzier list产生歧义
      // 不知道为啥移动构造会走这一条，所以加个sfinae
      // TODO move arrow to the last argument
      template<typename Arg>
         requires(!std::is_same_v<std::remove_cvref_t<Arg>, Edge<Symmetry, is_pointer>>)
      Edge(Arg&& arg, bool conjugated = false) : base_map_t(std::forward<Arg>(arg)), base_conjugated_t(conjugated) {}
      Edge(const typename base_map_t::pair_initializer_list_t& map_, bool conjugated = false) : base_map_t(map_), base_conjugated_t(conjugated) {}
      Edge(const typename base_map_t::symmetry_initializer_list_t& symmetries, bool conjugated = false) :
            base_map_t(symmetries),
            base_conjugated_t(conjugated) {}

      template<typename Arg>
      Edge(Arrow arrow, Arg&& arg, bool conjugated = false) :
            base_map_t(std::forward<Arg>(arg)),
            base_conjugated_t(conjugated),
            base_arrow_t(arrow) {}
      Edge(Arrow arrow, const typename base_map_t::pair_initializer_list_t& map, bool conjugated = false) :
            base_map_t(map),
            base_conjugated_t(conjugated),
            base_arrow_t(arrow) {}
      Edge(Arrow arrow, const typename base_map_t::symmetry_initializer_list_t& symmetries, bool conjugated = false) :
            base_map_t(symmetries),
            base_conjugated_t(conjugated),
            base_arrow_t(arrow) {}

      /**
       * 由费米子数自动构造箭头方向, 虽然这个不一定需要一致, 仅仅在只含有一个Fermi对称性时有效
       */
      void possible_reverse() {
         for (const auto& [symmetry, size] : map) {
            if (symmetry.get_first_parity() < 0) {
               arrow ^= true;
               return;
            }
         }
      }

      /**
       * 检查箭头是否有效, 即含有非零的费米子数
       */
      [[nodiscard]] bool arrow_valid() const {
         for (const auto& [symmetry, size] : map) {
            if (symmetry.get_first_parity() != 0) {
               return true;
            }
         }
         return false;
      }

      static constexpr bool i_am_an_edge = !is_pointer;
      static constexpr bool i_am_an_edge_pointer = is_pointer;
   };
   template<is_symmetry Symmetry, bool is_pointer>
   bool operator==(const Edge<Symmetry, is_pointer>& edge_1, const Edge<Symmetry, is_pointer>& edge_2) {
      return edge_1.arrow == edge_2.arrow && edge_1.conjugated == edge_2.conjugated && std::ranges::equal(edge_1.map, edge_2.map);
   }

   /**
    * 中间处理中常用到的数据类型, 类似Edge但是其中对称性值到子边长的映射表为指针
    * \see Edge
    */
   template<is_symmetry Symmetry>
   using EdgePointer = Edge<Symmetry, true>;

   /**
    * 判断一个类型是否为Edge类型, 这里不认为map为引用的Edge类型为Edge
    * \tparam T 如果T是Edge类型, 则value为true
    * \see is_edge_v
    */
   template<typename T>
   concept is_edge = T::i_am_an_edge;

   template<typename T>
   concept is_edge_pointer = T::i_am_an_edge_pointer;

   template<typename T>
   concept is_general_edge = is_edge<T> || is_edge_pointer<T>;

   /**
    * 对一个边的形状列表进行枚举分块, 并做一些其他操作
    * \tparam T 应是vector<Edge>或者vector<EdgePointer>的iterator
    * \param edges 即将要枚举的边列表的开头指针
    * \param rank 即将要枚举的边列表的大小
    * \param rank0 如果边列表为空，则调用rank0后返回
    * \param dims0 如果边列表中存在零维的边，则调用dims0后返回
    * \param operate 对枚举的每一个情况做操作
    * \note operate输入两个参数, 一个是每个边所在的位置列表, 一个是需要更新信息的位置开头, 并返回操作后需要更新的位置开头
    * \see initialize_block_symmetries_with_check, get_merged_edge
    */
   template<
         template<typename> class Allocator = std::allocator,
         bool check_conjugated = false,
         is_general_edge Edge,
         typename Rank0,
         typename Dims0,
         typename Operate>
   void loop_edge(const Edge* edges, const Rank rank, Rank0&& rank0, Dims0&& dims0, Operate&& operate) {
      if (rank == 0) [[unlikely]] {
         rank0();
         return;
      }
      using Iterator = typename Edge::map_t::const_iterator;
      auto symmetry_iterator_list = std::vector<Iterator, Allocator<Iterator>>();
      symmetry_iterator_list.reserve(rank);
      for (auto i = 0; i != rank; ++i) {
         const auto& map = edges[i].map;
         if (edges[i].map.empty()) [[unlikely]] {
            dims0();
            return;
         }
         symmetry_iterator_list.push_back((check_conjugated && edges[i].conjugated) ? std::prev(map.end()) : map.begin());
      }
      Rank minimum_changed = 0;
      while (true) {
         minimum_changed = operate(symmetry_iterator_list, minimum_changed);
         auto edge_position = rank - 1;

         while ((check_conjugated && edges[edge_position].conjugated) ? symmetry_iterator_list[edge_position]-- == edges[edge_position].map.begin() :
                                                                        ++symmetry_iterator_list[edge_position] == edges[edge_position].map.end()) {
            if (edge_position == 0) [[unlikely]] {
               return;
            }
            symmetry_iterator_list[edge_position] = (check_conjugated && edges[edge_position].conjugated) ?
                                                          std::prev(edges[edge_position].map.end()) :
                                                          edges[edge_position].map.begin();
            --edge_position;
         }
         minimum_changed = minimum_changed < edge_position ? minimum_changed : edge_position;
      }
   }

   /// \private
   template<template<typename> class Allocator = std::allocator, bool check_conjugated = false, std::ranges::contiguous_range Edges>
      requires is_general_edge<std::ranges::range_value_t<Edges>>
   [[nodiscard]] auto initialize_block_symmetries_with_check(const Edges& edges) {
      using Symmetry = typename Edges::value_type::symmetry_t;
      Rank rank = edges.size();
      // 对称性列表和大小
      using ResultItem = std::pair<std::vector<Symmetry, Allocator<Symmetry>>, Size>;
      auto result = std::vector<ResultItem, Allocator<ResultItem>>();
      auto symmetries = std::vector<Symmetry, Allocator<Symmetry>>(rank);
      auto sizes = std::vector<Size, Allocator<Size>>(rank);
      loop_edge<Allocator, check_conjugated>(
            edges.data(),
            rank,
            [&] {
               result.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{1});
            },
            [] {},
            [&](const auto& symmetry_iterator_list, Rank minimum_changed) {
               auto symmetry_summary = Symmetry();
               for (const auto& symmetry_iterator : symmetry_iterator_list) {
                  symmetry_summary += symmetry_iterator->first;
               }
               if (symmetry_summary == Symmetry()) [[unlikely]] {
                  for (auto i = minimum_changed; i < rank; i++) {
                     symmetries[i] = symmetry_iterator_list[i]->first;
                     sizes[i] = symmetry_iterator_list[i]->second * (i ? sizes[i - 1] : 1);
                  }
                  result.emplace_back(std::piecewise_construct, std::tuple{symmetries.begin(), symmetries.end()}, std::tuple{sizes.back()});
                  return rank;
               }
               return minimum_changed;
            });
      return result;
   }
   /**@}*/
} // namespace TAT
#endif
