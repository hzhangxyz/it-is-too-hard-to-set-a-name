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

namespace TAT {
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
} // namespace TAT

#include <vector>

namespace TAT {
   namespace no_initialize {
      template<typename T>
      using vector = std::vector<T, no_initialize_allocator<T>>;
   }
} // namespace TAT

#endif
