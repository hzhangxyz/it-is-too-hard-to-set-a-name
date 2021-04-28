/**
 * \file no_initialize_allocator.hpp
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
#ifndef TAT_NO_INITIALIZE_ALLOCATOR_HPP
#define TAT_NO_INITIALIZE_ALLOCATOR_HPP

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace TAT {
   namespace detail {
      /**
       * Allocator without initialize the element if no parameter given
       *
       * Inherit from std::allocator
       */
      template<typename T>
      struct no_initialize_allocator : std::allocator<T> {
         template<typename U>
         struct rebind {
            using other = no_initialize_allocator<U>;
         };

         template<typename U, typename... Args>
         void construct([[maybe_unused]] U* p, Args&&... args) {
            if constexpr (!((sizeof...(args) == 0) && (std::is_trivially_destructible_v<T>))) {
               new (p) U(std::forward<Args>(args)...);
            }
         }
      };

      template<typename T>
      using no_initialize_vector = std::vector<T, no_initialize_allocator<T>>;

      using no_initialize_string = std::basic_string<char, std::char_traits<char>, no_initialize_allocator<char>>;
      using no_initialize_istringstream = std::basic_istringstream<char, std::char_traits<char>, no_initialize_allocator<char>>;
   } // namespace detail
} // namespace TAT

#endif
