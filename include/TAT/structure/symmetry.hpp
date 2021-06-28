/**
 * \file symmetry.hpp
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
#ifndef TAT_SYMMETRY_HPP
#define TAT_SYMMETRY_HPP

#include <tuple>

#include "../utility/concepts.hpp"

namespace TAT {
   // A symmetry is a tuple of several int or bool, which may be fermi or bose
   // fermi_wrap is used to select fermi, otherwise bose by default.
   template<typename T>
   struct fermi_wrap {};

   namespace detail {
      template<typename T>
      struct fermi_unwrap_helper : std::type_identity<T> {};
      template<typename T>
      struct fermi_unwrap_helper<fermi_wrap<T>> : std::type_identity<T> {};
      template<typename T>
      using fermi_unwrap = typename fermi_unwrap_helper<T>::type;

      template<typename T>
      struct fermi_wrapped_helper : std::false_type {};
      template<typename T>
      struct fermi_wrapped_helper<fermi_wrap<T>> : std::true_type {};
      template<typename T>
      constexpr bool fermi_wrapped = fermi_wrapped_helper<T>::value;
   } // namespace detail

   /**
    * General symmetry type, used to mark edge index the different transform rule when some symmetric operation is applied.
    *
    * For example, Symmetry<int> is U1 symmetry and Symmetry<bool> is Z2 symmetry.
    * While Symmetry<int, femri_wrap<int>> represent the conservation of particle number of two particle, one is boson and the other is fermion.
    *
    * \param T integral type, such as bool or int, or use fermi_wrap<...> to mark the fermi symmetry
    */
   template<typename... T>
      requires(std::is_integral_v<detail::fermi_unwrap<T>>&&...)
   struct Symmetry : std::tuple<detail::fermi_unwrap<T>...> {
    private:
      using self_t = Symmetry<T...>; // used freq too high so alias it

    public:
      using base_tuple = std::tuple<detail::fermi_unwrap<T>...>;
      static constexpr int length = sizeof...(T);
      static constexpr std::array<bool, length> is_fermi_item = {detail::fermi_wrapped<T>...};
      static constexpr bool is_fermi_symmetry = (detail::fermi_wrapped<T> || ...);
      using index_sequence = std::index_sequence_for<T...>;

    private:
      template<typename... Args>
      base_tuple construct_base_tuple(const Args&... args) {
         if constexpr (sizeof...(Args) < length) {
            return construct_base_tuple(args..., 0);
         } else {
            return base_tuple(args...);
         }
      }

    public:
      template<typename... Args>
         requires((sizeof...(Args) <= length) && (std::is_integral_v<Args> && ...))
      Symmetry(const Args&... args) : base_tuple(construct_base_tuple(args...)) {}

    private:
      template<typename Item>
      static Item& inplace_plus_item(Item& a, const Item& b) {
         if constexpr (std::is_same_v<Item, bool>) {
            return a ^= b;
         } else {
            return a += b;
         }
      }
      template<std::size_t... Is>
      static self_t& inplace_plus_symmetry(self_t& symmetry_1, const self_t& symmetry_2, std::index_sequence<Is...>) {
         (inplace_plus_item(std::get<Is>(symmetry_1), std::get<Is>(symmetry_2)), ...);
         return symmetry_1;
      }

    public:
      self_t& operator+=(const self_t& other_symmetry) & {
         return inplace_plus_symmetry(*this, other_symmetry, index_sequence());
      }

    private:
      template<typename Item>
      static Item plus_item(const Item& a, const Item& b) {
         if constexpr (std::is_same_v<Item, bool>) {
            return a ^ b;
         } else {
            return a + b;
         }
      }
      template<size_t... Is>
      static self_t plus_symmetry(const self_t& symmetry_1, const self_t& symmetry_2, std::index_sequence<Is...>) {
         return self_t(plus_item(std::get<Is>(symmetry_1), std::get<Is>(symmetry_2))...);
      }

    public:
      [[nodiscard]] self_t operator+(const self_t& other_symmetry) const& {
         return plus_symmetry(*this, other_symmetry, index_sequence());
      }

    private:
      template<typename Item>
      static Item negative_item(const Item& a) {
         if constexpr (std::is_same_v<Item, bool>) {
            return a;
         } else {
            return -a;
         }
      }
      template<size_t... Is>
      static self_t negative_symmetry(const self_t& symmetry, std::index_sequence<Is...>) {
         return self_t(negative_item(std::get<Is>(symmetry))...);
      }

    public:
      [[nodiscard]] self_t operator-() const& {
         return negative_symmetry(*this, index_sequence());
      }

    private:
      /**
       * get parity of certain term of symmetry, if it is bose term, return false
       */
      template<std::size_t Index>
      [[nodiscard]] bool single_term_get_parity_helper() const {
         if constexpr (is_fermi_item[Index]) {
            const auto quantum_number = std::get<Index>(*this);
            if constexpr (std::is_same_v<decltype(quantum_number), bool>) {
               return quantum_number;
            } else {
               return bool(quantum_number % 2);
            }
         } else {
            return false;
         }
      }

      template<std::size_t... Is>
      bool get_parity_helper(std::index_sequence<Is...>) const {
         return (single_term_get_parity_helper<Is>() ^ ...);
      }

    public:
      /**
       * Get the total parity, whether fermion or boson
       */
      [[nodiscard]] bool get_parity() const {
         if constexpr (is_fermi_symmetry) {
            return get_parity_helper(index_sequence());
         } else {
            return false;
         }
      }

    public:
      /**
       * Give the parity when reverse some edge
       *
       * \param symmetries symmetry list for each rank of some block
       * \param reverse_flag flags marking whether to reverse for each rank
       * \param valid_mark flags marking the validity for each rank
       *
       * when reversing edge in edge_operator, all reversed edge when parity is odd,
       * and this function will give the total parity summation of all the edge.
       * \see Tensor::edge_operator
       */
      [[nodiscard]] static bool
      get_reverse_parity(const range_of<self_t> auto& symmetries, const range_of<bool> auto& reverse_flag, const range_of<bool> auto& valid_mark) {
         auto result = false;
         for (Rank i = 0; i < symmetries.size(); i++) {
            if (reverse_flag[i] && valid_mark[i]) {
               result ^= symmetries[i].get_parity();
            }
         }
         return result;
      }

    public:
      /**
       * Give the parity of transposation
       *
       * \param symmetries symmetry list for each rank of some block
       * \param transpose_plan plan of transposition
       *
       * There is no valid mark like reversing edge, since the parity of transposition is always valid, because
       * transposition is the operation of a single tensor.
       * \see Tensor::edge_operator
       */
      [[nodiscard]] static bool get_transpose_parity(const range_of<self_t> auto& symmetries, const range_of<Rank> auto& transpose_plan) {
         auto result = false;
         for (Rank i = 0; i < symmetries.size(); i++) {
            for (Rank j = i + 1; j < symmetries.size(); j++) {
               if (transpose_plan[i] > transpose_plan[j]) {
                  result ^= symmetries[i].get_parity() && symmetries[j].get_parity();
               }
            }
         }
         return result;
      }

    public:
      /**
       * Give the parity when merging or spliting edges.
       *
       * \param symmetries symmetry list for each rank of some block
       * \param split_merge_flag The plan of merging or splitting
       * \param valid_mark validity of each merging or splitting operation.
       *
       * \note This is equivalant to total transposition over the merging list or splitting list,
       * while \f$\sum_{i\le j} s_i s_j = \frac{(\sum s_i)^2 - \sum s_i^2}{2}\f$, so it can be simplified.
       */
      [[nodiscard]] static bool get_split_merge_parity(
            const range_of<self_t> auto& symmetries,     // before merge length
            const range_of<Rank> auto& split_merge_flag, // before merge length
            const range_of<bool> auto& valid_mark) {     // after merge length
         auto result = false;
         for (Rank split_merge_group_position = 0, split_merge_begin_position = 0, split_merge_end_position = 0;
              split_merge_group_position < valid_mark.size();
              split_merge_group_position++) {
            // split_merge_group_position point to after merge position
            // begin_position and end_position point to before merge position
            while (split_merge_end_position < symmetries.size() && split_merge_flag[split_merge_end_position] == split_merge_group_position) {
               split_merge_end_position++;
            }
            if (valid_mark[split_merge_group_position]) {
               // (sum(x)^2 - sum(x^2)) / 2 % 2 = (sum(x)^2 - sum(x)) & 2 != 0 = sum(x) & 2 != 0
               auto sum_of_parity = 0l;
               for (auto position_in_group = split_merge_begin_position; position_in_group < split_merge_end_position; position_in_group++) {
                  sum_of_parity += symmetries[position_in_group].get_parity();
               }
               result ^= ((sum_of_parity & 2) != 0);
            }
            split_merge_begin_position = split_merge_end_position;
         }
         return result;
      }
   };

   namespace detail {
      template<typename T>
      struct is_symmetry_helper : std::bool_constant<false> {};

      template<typename... T>
      struct is_symmetry_helper<Symmetry<T...>> : std::bool_constant<true> {};
   } // namespace detail

   /**
    * Check whether a type is a symmetry type
    *
    * Only type specialized by Symmetry is symmetry type
    *
    * \see Symmetry
    */
   template<typename T>
   concept is_symmetry = detail::is_symmetry_helper<T>::value;

   // common used symmetries alias
   /**
    * Z2 Symmetry
    */
   using Z2 = bool;
   /**
    * U1 symmetry
    */
   using U1 = std::int32_t;

   using NoSymmetry = Symmetry<>;
   using Z2Symmetry = Symmetry<Z2>;
   using U1Symmetry = Symmetry<U1>;
   using FermiSymmetry = Symmetry<fermi_wrap<U1>>;
   using FermiZ2Symmetry = Symmetry<fermi_wrap<U1>, Z2>;
   using FermiU1Symmetry = Symmetry<fermi_wrap<U1>, U1>;
} // namespace TAT
#endif
