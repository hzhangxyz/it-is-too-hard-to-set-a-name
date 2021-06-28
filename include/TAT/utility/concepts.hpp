/**
 * \file concepts.hpp
 *
 * Copyright (C) 2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_CONCEPTS_HPP
#define TAT_CONCEPTS_HPP

#include <concepts>
#include <ranges>

namespace TAT {
   template<typename T>
   using empty_list = std::array<T, 0>;

   template<typename T, typename Target = void>
   concept exist = std::same_as<T, T> &&(std::same_as<Target, void> || std::same_as<Target, std::remove_cvref_t<T>>);

   template<typename T, typename... Targets>
   concept one_of = (exist<T, Targets> || ...);

   template<typename Range, typename Value = void>
   concept range_of = std::ranges::range<Range> && exist<std::ranges::range_value_t<Range>, Value>;

   template<typename Pair, typename First = void, typename Second = void>
   concept pair_of = requires(const Pair pair) {
      { std::get<0>(pair) } -> exist<First>;
      { std::get<1>(pair) } -> exist<Second>;
   };

   template<typename Range, typename Key = void, typename Value = void>
   concept pair_range_of = std::ranges::range<Range> && pair_of<typename std::ranges::range_value_t<Range>, Key, Value>;

   template<typename T, typename Name>
   concept same_pair_range_of = pair_range_of<T, Name, Name>;

   // The following is used to simulate map or set by vector

   // set or map
   template<typename Container, typename Key = void>
   concept findable = requires(Container c, std::conditional_t<std::same_as<Key, void>, typename std::remove_cvref_t<Container>::key_type, Key> k) {
      c.find(k);
   };

   // A maybe just key or pair of key and value
   template<typename Key, typename A>
   const auto& get_key(const A& a) {
      if constexpr (std::same_as<std::remove_cvref_t<Key>, std::remove_cvref_t<A>>) {
         return a;
      } else {
         return std::get<0>(a);
      }
   }

   template<typename A, typename B>
   concept lexicographical_comparable = requires(A a, B b) {
      std::ranges::lexicographical_compare(a, b);
      std::ranges::equal(a, b);
   };

   template<typename A, typename B>
   concept value_first_type_lexicographical_comparable = lexicographical_comparable<typename std::ranges::range_value_t<A>::first_type, B>;

   // find for map of map like array
   template<bool Lexicographic = false, typename Container, typename Key>
      requires(Lexicographic ? value_first_type_lexicographical_comparable<Container, Key> : pair_range_of<Container, Key>)
   constexpr auto map_find(Container& v, const Key& key) {
      // it may be slow when v is real map but find lexicographically
      if constexpr (Lexicographic) {
         auto result = std::lower_bound(v.begin(), v.end(), key, [](const auto& a, const auto& b) {
            return std::ranges::lexicographical_compare(get_key<Key>(a), get_key<Key>(b));
         });
         if (result == v.end()) {
            // result may be un dereferencable
            return v.end();
         } else if (std::ranges::equal(result->first, key)) {
            return result;
         } else {
            return v.end();
         }
      } else {
         if constexpr (findable<Container, Key>) {
            return v.find(key);
         } else {
            auto result = std::lower_bound(v.begin(), v.end(), key, [](const auto& a, const auto& b) {
               return get_key<Key>(a) < get_key<Key>(b);
            });
            if (result == v.end()) {
               // result may be un dereferencable
               return v.end();
            } else if (result->first == key) {
               return result;
            } else {
               return v.end();
            }
         }
      }
   }

   // at for map of map like array
   template<bool Lexicographic = false, typename Container, typename Key>
      requires(Lexicographic ? value_first_type_lexicographical_comparable<Container, Key> : pair_range_of<Container, Key>)
   auto& map_at(Container& v, const Key& key) {
      auto found = map_find<Lexicographic>(v, key);
      if (found == v.end()) {
         throw std::out_of_range("fake map at");
      } else {
         return found->second;
      }
   }

   template<typename Container, typename Key>
      requires range_of<Container, Key>
   auto set_find(Container& v, const Key& key) {
      if constexpr (findable<Container, Key>) {
         return v.find(key);
      } else {
         auto result = std::lower_bound(v.begin(), v.end(), key, [](const auto& a, const auto& b) {
            return a < b;
         });
         if (result == v.end()) {
            return result;
         } else if (*result == key) {
            return result;
         } else {
            return v.end();
         }
      }
   }

   // generate map/set like array

   auto default_compare = [](const auto& a, const auto& b) {
      return a < b;
   };

   template<bool force_set = false, typename Container, typename Compare = const decltype(default_compare)&>
   void do_sort(Container& c, Compare&& compare = default_compare) {
      if constexpr (force_set || !pair_range_of<Container>) {
         // set like
         std::ranges::sort(c);
      } else {
         // map like
         std::ranges::sort(c, [&](const auto& a, const auto& b) {
            return compare(std::get<0>(a), std::get<0>(b));
         });
      }
   }

   // forward map/set like container, if need sort, sort it
   /**
    * forward map/set like container, if need sort, sort it
    *
    * if the input is a map or set, return itself
    * if it is array, try to sort inplace(move from caller), otherwise sort outplace and return sorted array
    *
    * if outplace sort is needed, Result is preferred result type
    */
   template<typename Result, typename Container>
   decltype(auto) may_need_sort(Container&& c) {
      if constexpr (findable<Container>) {
         return std::forward<Container>(c);
      } else {
         // it is strange that if I requires do_sort(l) here, program cannot compile.
         if constexpr (requires(Container && l) { std::ranges::sort(l); }) {
            // can change inplace
            do_sort(c);
            return std::move(c);
         } else {
            // can not change inplace
            auto result = Result(c.begin(), c.end());
            do_sort(result);
            return result;
         }
      }
   }
} // namespace TAT

#endif
