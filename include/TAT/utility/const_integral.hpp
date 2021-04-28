/**
 * \file const_integral.hpp
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
#ifndef TAT_CONST_INTEGRAL_HPP
#define TAT_CONST_INTEGRAL_HPP

#include <variant>

namespace TAT {
   namespace detail {
      /**
       * Maybe const integer
       *
       * Used in the situation where optmization can be applied for some small number
       *
       * \see to_const_integral
       */
      template<auto, typename DynamicType = void>
      struct const_integral_t {
         using value_type = DynamicType;
         value_type m_value;
         const_integral_t() = delete;
         const_integral_t(value_type v) : m_value(v) {}
         value_type value() const {
            return m_value;
         }
         static constexpr bool is_static = false;
         static constexpr bool is_dynamic = true;
      };

      template<auto StaticValue>
      struct const_integral_t<StaticValue, void> {
         using value_type = decltype(StaticValue);
         const_integral_t() = default;
         static constexpr value_type value() {
            return StaticValue;
         }
         static constexpr bool is_static = true;
         static constexpr bool is_dynamic = false;
      };

      template<typename T>
      const_integral_t(T) -> const_integral_t<0, T>;

      template<typename R, typename T>
      R to_const_integral_helper(T value) {
         return const_integral_t(value);
      }
      template<typename R, typename T, T first_value, T... possible_value>
      R to_const_integral_helper(T value) {
         if (first_value == value) {
            return const_integral_t<first_value>();
         } else {
            return to_const_integral_helper<R, T, possible_value...>(value);
         }
      }
   } // namespace detail

   /**
    * Convert a variable to std::variant<run time variable, compile time variable 1, compile time variable 2, ...>
    *
    * Use std::visit([](const auto& variable) { xxx; variable.value(); xxx}, v) to dispatch compile time variable and run time variable
    */
   template<std::integral T, T... possible_value>
   auto to_const_integral(T value) {
      using result_type = std::variant<detail::const_integral_t<0, T>, detail::const_integral_t<possible_value>...>;
      return detail::to_const_integral_helper<result_type, T, possible_value...>(value);
   }
} // namespace TAT

#endif
