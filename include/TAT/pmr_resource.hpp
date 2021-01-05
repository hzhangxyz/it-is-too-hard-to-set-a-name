/**
 * \file pmr_resource.hpp
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
#ifndef TAT_PMR_RESOURCE_HPP
#define TAT_PMR_RESOURCE_HPP

// TODO 虽然pmr在c++17中, 但是gcc 7并不支持pmr, 所以增加一个使用boost的选项
#ifdef TAT_USE_BOOST_PMR
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <boost/container/pmr/polymorphic_allocator.hpp>
#else
#include <memory_resource>
#endif

namespace TAT {
#ifdef TAT_USE_BOOST_PMR
   namespace pmr {
      using boost::container::pmr::get_default_resource;
      using boost::container::pmr::memory_resource;
      using boost::container::pmr::monotonic_buffer_resource;
      using boost::container::pmr::polymorphic_allocator;
      using boost::container::pmr::set_default_resource;

      template<typename T>
      using vector = std::vector<T, polymorphic_allocator<T>>;

      template<typename Key, typename T, typename Compare = std::less<Key>>
      using map = std::map<Key, T, Compare, polymorphic_allocator<std::pair<const Key, T>>>;

      template<class Key, class Compare = std::less<Key>>
      using set = std::set<Key, Compare, polymorphic_allocator<Key>>;
   } // namespace pmr
#else
   namespace pmr = std::pmr;
#endif

   // on windows stack size is 1MB(1<<20), and on linux, stack size is 8M(1<<23)
   constexpr std::size_t default_buffer_size = 1 << 15;

   template<std::size_t buffer_size = default_buffer_size>
   struct scope_resource {
      std::byte buffer[buffer_size];
      pmr::monotonic_buffer_resource resource;
      pmr::memory_resource* upstream;
      scope_resource() : resource(buffer, sizeof(buffer)), upstream(pmr::set_default_resource(&resource)) {}
      ~scope_resource() {
         pmr::set_default_resource(upstream);
      }
   };

   struct scope_resource_adapter {
      pmr::monotonic_buffer_resource resource;
      pmr::memory_resource* upstream;
      scope_resource_adapter(void* buffer, std::size_t size) : resource(buffer, size), upstream(pmr::set_default_resource(&resource)) {}
      ~scope_resource_adapter() {
         pmr::set_default_resource(upstream);
      }
   };
} // namespace TAT
#endif
