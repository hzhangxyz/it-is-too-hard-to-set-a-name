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
#include "../utility/concepts.hpp"
#include "symmetry.hpp"

namespace TAT {
   template<is_symmetry Symmetry, bool is_pointer = false>
   struct edge_segment_t {
      static constexpr bool is_not_pointer = !is_pointer;

      using pair_initializer_list_t = std::initializer_list<std::pair<Symmetry, Size>>;
      using symmetry_initializer_list_t = std::initializer_list<Symmetry>;

      using symmetry_t = Symmetry;
      using segment_t = std::vector<std::pair<Symmetry, Size>>;
      std::conditional_t<is_pointer, const segment_t&, segment_t> segment;

      edge_segment_t() = default;
      edge_segment_t(const edge_segment_t& edge) = default;
      edge_segment_t(edge_segment_t&& edge) = default;
      edge_segment_t& operator=(const edge_segment_t&) = default;
      edge_segment_t& operator=(edge_segment_t&&) noexcept = default;
      ~edge_segment_t() = default;

      template<pair_range_of<Symmetry, Size> SymmetrySizeList = pair_initializer_list_t>
         requires(is_not_pointer)
      edge_segment_t(SymmetrySizeList&& s) {
         // if it is okey to copy or move directly, do it, otherwise set elementwisely
         if constexpr (requires(segment_t & a, SymmetrySizeList && b) { a = std::forward<SymmetrySizeList>(b); }) {
            segment = std::forward<SymmetrySizeList>(s);
         } else {
            segment.reserve(s.size());
            for (const auto& [symmetry, size] : s) {
               segment.push_back({symmetry, size});
            }
         }
      }

      /**
       * construct the edge with list of symmetry, each size of them are 1
       */
      template<range_of<Symmetry> SymmetryList = symmetry_initializer_list_t>
         requires(is_not_pointer)
      edge_segment_t(const SymmetryList& symmetries) {
         segment.reserve(symmetries.size());
         for (const auto& symmetry : symmetries) {
            segment.push_back({symmetry, 1});
         }
      }

      /**
       * construct a trivial edge, only contain a single symmetry
       */
      template<typename = void>
         requires(is_not_pointer)
      edge_segment_t(const Size dimension, const Symmetry symmetry = Symmetry()) : segment({{symmetry, dimension}}) {}

      template<typename = void>
         requires(is_pointer)
      edge_segment_t(const segment_t& s) : segment(s) {}

      std::pair<Symmetry, Size> get_point_from_index(Size index) const {
         for (const auto& [symmetry, size] : segment) {
            if (index < size) {
               return {symmetry, index};
            } else {
               index -= size;
            }
         }
         detail::error("Index is more than edge total dimension");
      }

      Size get_index_from_point(const std::pair<Symmetry, Size>& pair) const {
         Size result = std::get<1>(pair);
         for (const auto& [symmetry, size] : segment) {
            if (symmetry == std::get<0>(pair)) {
               return result;
            }
            result += size;
         }
         detail::error("The symmetry not found in this edge");
      }

      Size get_dimension_from_symmetry(const Symmetry& symmetry) const {
         return std::ranges::find(segment, symmetry, &std::pair<Symmetry, Size>::first)->second;
      }

      template<range_of<Symmetry> SymmetryOrder>
         requires(is_pointer)
      void exchange_symmetry(const SymmetryOrder& symmetry_order) {
         auto new_segment = segment_t();
         new_segment.reserve(segment.size());
         for (const auto& symmetry : symmetry_order) {
            new_segment.emplace_back(symmetry, get_dimension_from_symmetry(symmetry));
         }
         segment = std::move(new_segment);
      }
   };

   struct edge_bose_arrow_t {
      static constexpr Arrow arrow = false;
      edge_bose_arrow_t() {}
      edge_bose_arrow_t(Arrow) {}
   };

   // there are background EPR pair for each edge, for fermi edge, it is needed to record the order of this EPR pair, which is so called fermi arrow
   struct edge_fermi_arrow_t {
      Arrow arrow;
      edge_fermi_arrow_t() : arrow(false) {}
      edge_fermi_arrow_t(Arrow arrow) : arrow(arrow) {}
   };
   template<is_symmetry Symmetry>
   using edge_arrow_t = std::conditional_t<Symmetry::is_fermi_symmetry, edge_fermi_arrow_t, edge_bose_arrow_t>;

   /**
    * The shape of tensor edge, is a list of pair of symmetry and size, which construct a structure like line segment.
    * If it is fermi edge, an arrow is also included
    *
    * \tparam Symmetry The symmetry of the tensor
    * \tparam is_pointer whether it is just point to the data or the edge containing the real structure.
    */
   template<is_symmetry Symmetry, bool is_pointer = false>
   struct Edge : edge_segment_t<Symmetry, is_pointer>, edge_arrow_t<Symmetry> {
      using base_arrow_t = edge_arrow_t<Symmetry>;
      using base_segment_t = edge_segment_t<Symmetry, is_pointer>;

      using base_arrow_t::arrow;
      using base_segment_t::segment;

      Edge() = default;
      Edge(const Edge&) = default;
      Edge(Edge&&) noexcept = default;
      Edge& operator=(const Edge&) = default;
      Edge& operator=(Edge&&) noexcept = default;
      ~Edge() = default;

      template<typename Arg>
         requires(!std::is_same_v<std::remove_cvref_t<Arg>, Edge<Symmetry, is_pointer>>)
      Edge(Arg&& arg, Arrow arrow = false) : base_segment_t(std::forward<Arg>(arg)), base_arrow_t(arrow) {}
      Edge(const typename base_segment_t::pair_initializer_list_t& segment, Arrow arrow = false) : base_segment_t(segment), base_arrow_t(arrow) {}
      Edge(const typename base_segment_t::symmetry_initializer_list_t& symmetries, Arrow arrow = false) :
            base_segment_t(symmetries),
            base_arrow_t(arrow) {}
   };

   template<is_symmetry Symmetry, bool is_pointer>
   bool operator==(const Edge<Symmetry, is_pointer>& edge_1, const Edge<Symmetry, is_pointer>& edge_2) {
      return edge_1.arrow == edge_2.arrow && std::ranges::equal(edge_1.segment, edge_2.segment);
   }

   /**
    * An edge but only containing a pointer to other edge's segment data
    * \see Edge
    */
   template<is_symmetry Symmetry>
   using EdgePointer = Edge<Symmetry, true>;

   namespace detail {
      template<typename T>
      struct is_edge_helper : std::bool_constant<false> {};

      template<typename Symmetry>
      struct is_edge_helper<Edge<Symmetry, false>> : std::bool_constant<true> {};

      template<typename T>
      struct is_edge_pointer_helper : std::bool_constant<false> {};

      template<typename T>
      struct is_edge_pointer_helper<Edge<T, true>> : std::bool_constant<true> {};
   } // namespace detail

   template<typename T>
   concept is_edge = detail::is_edge_helper<T>::value;

   template<typename T>
   concept is_edge_pointer = detail::is_edge_pointer_helper<T>::value;

   template<typename T>
   concept is_general_edge = is_edge<T> || is_edge_pointer<T>;

   /**
    * Loop over each block generated by list of edge
    *
    * \tparam Allocator allocator of iterator vector
    * \param edges edges list
    * \param rank0 if edges list is empty, call rank0
    * \param dims0 if edges list contains emtpy edge, call dims0
    * \param operate call operator for each combination of different symmetries in edges list
    * \note operate has two arguments, first is vector of iterators from each edge segment,
    * another is a record for point needed to be updated because it is changed by loop_edge
    * \see initialize_block_symmetries_with_check, get_merged_edge
    */
   template<template<typename> class Allocator = std::allocator, std::ranges::contiguous_range Edges>
      requires is_general_edge<std::ranges::range_value_t<Edges>>
   void loop_edge(Edges edges, std::invocable auto&& rank0, std::invocable auto&& dims0, auto&& operate) {
      Rank rank = edges.size();
      if (rank == 0) [[unlikely]] {
         rank0();
         return;
      }
      using Edge = std::ranges::range_value_t<Edges>;
      using Iterator = typename Edge::segment_t::const_iterator;
      auto symmetry_iterator_list = std::vector<Iterator, Allocator<Iterator>>();
      symmetry_iterator_list.reserve(rank);
      for (auto i = 0; i != rank; ++i) {
         const auto& segment = edges[i].segment;
         if (segment.empty()) [[unlikely]] {
            dims0();
            return;
         }
         symmetry_iterator_list.push_back(segment.begin());
      }
      Rank minimum_changed = 0; // included, need update
      while (true) {
         minimum_changed = operate(symmetry_iterator_list, minimum_changed);
         auto edge_position = rank - 1;

         while (++symmetry_iterator_list[edge_position] == edges[edge_position].segment.end()) {
            if (edge_position == 0) [[unlikely]] {
               return;
            }
            symmetry_iterator_list[edge_position] = edges[edge_position].segment.begin();
            --edge_position;
         }
         minimum_changed = minimum_changed < edge_position ? minimum_changed : edge_position;
      }
   }

   template<template<typename> class Allocator = std::allocator, std::ranges::contiguous_range Edges>
      requires is_general_edge<std::ranges::range_value_t<Edges>>
   [[nodiscard]] auto initialize_block_symmetries_with_check(const Edges& edges) {
      using Edge = std::ranges::range_value_t<Edges>;
      using Symmetry = typename Edge::symmetry_t;
      Rank rank = edges.size();
      // symmetries list and its size
      using ResultItem = std::pair<std::vector<Symmetry, Allocator<Symmetry>>, Size>;
      auto result = std::vector<ResultItem, Allocator<ResultItem>>(); // following the normal order of blocks
      auto symmetries = std::vector<Symmetry, Allocator<Symmetry>>(rank);
      auto sizes = std::vector<Size, Allocator<Size>>(rank);
      loop_edge<Allocator>(
            edges,
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
                  // Symmetry iterator list is changed from minimum_changed since last call of this function
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
} // namespace TAT
#endif
