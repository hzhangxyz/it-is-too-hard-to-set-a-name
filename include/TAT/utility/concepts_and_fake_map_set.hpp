/**
 * \file concepts_and_fake_map_set.hpp
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
#ifndef TAT_CONCEPTS_AND_FAKE_MAP_SET_HPP
#define TAT_CONCEPTS_AND_FAKE_MAP_SET_HPP

#include <concepts>
#include <ranges>

namespace TAT {
   template<typename T>
   using empty_list = std::array<T, 0>;

   template<typename Range, typename Value = void>
   concept range_of = std::ranges::range<Range> &&(std::is_same_v<Value, void> || std::same_as<Value, std::ranges::range_value_t<Range>>);

   template<typename T, typename Target = void>
   concept exist = std::is_same_v<T, T> &&(std::is_same_v<Target, void> || std::is_same_v<Target, T>);

   template<typename Range, typename Key = void, typename Value = void>
   concept map_like_range_of = std::ranges::range<Range> && exist<std::remove_cvref_t<typename std::ranges::range_value_t<Range>::first_type>, Key> &&
         exist<std::remove_cvref_t<typename std::ranges::range_value_t<Range>::second_type>, Value>;

   // set or map
   template<typename Container, typename Key = void>
   concept findable = requires(Container c, std::conditional_t<std::is_same_v<Key, void>, typename std::remove_cvref_t<Container>::key_type, Key> k) {
      c.find(k);
   };

   // A maybe just key or pair of key and value
   template<typename Key, typename A>
   const auto& get_key(const A& a) {
      if constexpr (std::is_same_v<std::remove_cvref_t<Key>, std::remove_cvref_t<A>>) {
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

   // find for map of map like array
   template<bool Lexicographic = false, typename Container, typename Key>
      requires(
            Lexicographic ? lexicographical_comparable<typename std::ranges::range_value_t<Container>::first_type, Key> :
                            map_like_range_of<Container, Key>)
   constexpr auto map_find(Container& v, const Key& key) {
      if constexpr (Lexicographic) {
         auto result = std::lower_bound(v.begin(), v.end(), key, [](const auto& a, const auto& b) {
            return std::ranges::lexicographical_compare(get_key<Key>(a), get_key<Key>(b));
         });
         if (result == v.end() || std::ranges::equal(result->first, key)) {
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
            if (result == v.end() || result->first == key) {
               return result;
            } else {
               return v.end();
            }
         }
      }
   }

   // at for map of map like array
   template<bool Lexicographic = false, typename Container, typename Key>
      requires(
            Lexicographic ? lexicographical_comparable<typename std::ranges::range_value_t<Container>::first_type, Key> :
                            map_like_range_of<Container, Key>)
   auto& map_at(Container& v, const Key& key) {
      if constexpr (Lexicographic) {
         auto result = std::lower_bound(v.begin(), v.end(), key, [](const auto& a, const auto& b) {
            return std::ranges::lexicographical_compare(get_key<Key>(a), get_key<Key>(b));
         });
         if (result == v.end() || !std::ranges::equal(result->first, key)) {
            throw std::out_of_range("fake map at");
         } else {
            return result->second;
         }
      } else {
         if constexpr (findable<Container, Key>) {
            return v.at(key);
         } else {
            auto result = std::lower_bound(v.begin(), v.end(), key, [](const auto& a, const auto& b) {
               return get_key<Key>(a) < get_key<Key>(b);
            });
            if (result == v.end() || result->first != key) {
               throw std::out_of_range("fake map at");
            } else {
               return result->second;
            }
         }
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
         if (result == v.end() || *result == key) {
            return result;
         } else {
            return v.end();
         }
      }
   }

   template<typename Container>
   void do_sort(Container& c) {
      if constexpr (map_like_range_of<Container>) {
         // map like
         std::ranges::sort(c, [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
         });
      } else {
         // set like
         std::ranges::sort(c);
      }
   }

   // forward map/set like container, if need sort, sort it
   template<typename Result, typename Container>
   decltype(auto) may_need_sort(Container&& c) {
      if constexpr (findable<Container>) {
         return std::forward<Container>(c);
      } else {
         // it is stange that if I requires do_sort(l) here, program cannot compile.
         if constexpr (requires(Container && l) { std::ranges::sort(l); }) {
            // 可以修改
            do_sort(c);
            return std::move(c);
         } else {
            // 不可以修改
            auto result = Result(c.begin(), c.end());
            do_sort(result);
            return result;
         }
      }
   }

   template<typename T, typename Name>
   concept pair_range_of = requires(typename std::remove_cvref_t<T>::value_type item) {
      requires std::ranges::range<T>;
      Name(std::get<0>(item));
      Name(std::get<1>(item));
   };
} // namespace TAT

#endif
