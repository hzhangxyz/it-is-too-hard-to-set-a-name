/**
 * \file scalar.hpp
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
#ifndef TAT_SCALAR_HPP
#define TAT_SCALAR_HPP

#include "../structure/tensor.hpp"
#include "../utility/timer.hpp"

namespace TAT {
   /**
    * \defgroup Scalar
    * @{
    */

#define TAT_DEFINE_SCALAR_OPERATOR(OP, EVAL) \
   template<is_scalar ScalarType1, is_scalar ScalarType2> \
      requires(is_complex<ScalarType1> || is_complex<ScalarType2>) \
   &&(!std::is_same_v<real_scalar<ScalarType1>, real_scalar<ScalarType2>>)inline auto OP(const ScalarType1& a, const ScalarType2& b) { \
      using t = std::common_type_t<decltype(a), decltype(b)>; \
      return EVAL; \
   }
   TAT_DEFINE_SCALAR_OPERATOR(operator+, t(a) + t(b))
   TAT_DEFINE_SCALAR_OPERATOR(operator-, t(a) - t(b))
   TAT_DEFINE_SCALAR_OPERATOR(operator*, t(a) * t(b))
   TAT_DEFINE_SCALAR_OPERATOR(operator/, t(a) / t(b))
#undef TAT_DEFINE_SCALAR_OPERATOR

   inline timer scalar_outplace_guard("scalar_outplace");

#define TAT_DEFINE_SCALAR_OPERATOR(OP, EVAL1, EVAL2, EVAL3) \
   template<is_scalar ScalarType1, is_scalar ScalarType2, is_symmetry Symmetry, is_name Name> \
   [[nodiscard]] auto OP(const Tensor<ScalarType1, Symmetry, Name>& tensor_1, const Tensor<ScalarType2, Symmetry, Name>& tensor_2) { \
      auto timer_guard = scalar_outplace_guard(); \
      using ScalarType = std::common_type_t<ScalarType1, ScalarType2>; \
      if (tensor_1.names.size() != tensor_2.names.size()) [[unlikely]] { \
         detail::error("Try to do scalar operator on two different rank tensor"); \
      } \
      auto real_tensor_2_pointer = &tensor_2; \
      auto new_tensor_2 = Tensor<ScalarType2, Symmetry, Name>(); \
      if (tensor_1.names != tensor_2.names) [[unlikely]] { \
         new_tensor_2 = tensor_2.transpose(tensor_1.names); \
         real_tensor_2_pointer = &new_tensor_2; \
      } \
      const auto& real_tensor_2 = *real_tensor_2_pointer; \
      if (tensor_1.core->edges != real_tensor_2.core->edges) [[unlikely]] { \
         detail::error("Try to do scalar operator on two tensors which edges not compatible"); \
      } \
      auto result = Tensor<ScalarType, Symmetry, Name>{tensor_1.names, tensor_1.core->edges}; \
      const ScalarType1* __restrict a = tensor_1.storage().data(); \
      const ScalarType2* __restrict b = real_tensor_2.storage().data(); \
      ScalarType* __restrict c = result.storage().data(); \
      const auto size = result.storage().size(); \
      for (Size j = 0; j < size; j++) { \
         EVAL3; \
      } \
      return result; \
   } \
   template<is_scalar ScalarType1, is_scalar ScalarType2, is_symmetry Symmetry, is_name Name> \
   [[nodiscard]] auto OP(const Tensor<ScalarType1, Symmetry, Name>& tensor_1, const ScalarType2& number_2) { \
      auto timer_guard = scalar_outplace_guard(); \
      using ScalarType = std::common_type_t<ScalarType1, ScalarType2>; \
      const ScalarType1* __restrict a = tensor_1.storage().data(); \
      const auto& y = number_2; \
      auto result = Tensor<ScalarType, Symmetry, Name>{tensor_1.names, tensor_1.core->edges}; \
      ScalarType* __restrict c = result.storage().data(); \
      const auto size = result.storage().size(); \
      for (Size j = 0; j < size; j++) { \
         EVAL2; \
      } \
      return result; \
   } \
   template<is_scalar ScalarType1, is_scalar ScalarType2, is_symmetry Symmetry, is_name Name> \
   [[nodiscard]] auto OP(const ScalarType1& number_1, const Tensor<ScalarType2, Symmetry, Name>& tensor_2) { \
      auto timer_guard = scalar_outplace_guard(); \
      using ScalarType = std::common_type_t<ScalarType1, ScalarType2>; \
      const auto& x = number_1; \
      const ScalarType2* __restrict b = tensor_2.storage().data(); \
      auto result = Tensor<ScalarType, Symmetry, Name>{tensor_2.names, tensor_2.core->edges}; \
      ScalarType* __restrict c = result.storage().data(); \
      const auto size = result.storage().size(); \
      for (Size j = 0; j < size; j++) { \
         EVAL1; \
      } \
      return result; \
   }

   TAT_DEFINE_SCALAR_OPERATOR(operator+, c[j] = x + b[j], c[j] = a[j] + y, c[j] = a[j] + b[j])
   TAT_DEFINE_SCALAR_OPERATOR(operator-, c[j] = x - b[j], c[j] = a[j] - y, c[j] = a[j] - b[j])
   TAT_DEFINE_SCALAR_OPERATOR(operator*, c[j] = x * b[j], c[j] = a[j] * y, c[j] = a[j] * b[j])
   TAT_DEFINE_SCALAR_OPERATOR(operator/, c[j] = x / b[j], c[j] = a[j] / y, c[j] = a[j] / b[j])
#undef TAT_DEFINE_SCALAR_OPERATOR

   inline timer scalar_inplace_guard("scalar_inplace");

#define TAT_DEFINE_SCALAR_OPERATOR(OP, EVAL1, EVAL2) \
   template<is_scalar ScalarType1, is_scalar ScalarType2, is_symmetry Symmetry, is_name Name> \
      requires(is_complex<ScalarType1> || is_real<ScalarType2>) \
   Tensor<ScalarType1, Symmetry, Name> \
   &OP(Tensor<ScalarType1, Symmetry, Name>& tensor_1, const Tensor<ScalarType2, Symmetry, Name>& tensor_2) { \
      auto timer_guard = scalar_inplace_guard(); \
      if (tensor_1.core.use_count() != 1) [[unlikely]] { \
         tensor_1.core = std::make_shared<Core<ScalarType1, Symmetry>>(*tensor_1.core); \
         detail::what_if_copy_shared("Inplace operator on tensor shared, copy happened here"); \
      } \
      auto real_tensor_2_pointer = &tensor_2; \
      auto new_tensor_2 = Tensor<ScalarType2, Symmetry, Name>(); \
      if (tensor_1.names != tensor_2.names) [[unlikely]] { \
         new_tensor_2 = tensor_2.transpose(tensor_1.names); \
         real_tensor_2_pointer = &new_tensor_2; \
      } \
      const auto& real_tensor_2 = *real_tensor_2_pointer; \
      if (tensor_1.core->edges != real_tensor_2.core->edges) [[unlikely]] { \
         detail::error("Try to do scalar operator on two tensors which edges not compatible"); \
      } \
      ScalarType1* __restrict a = tensor_1.storage().data(); \
      const ScalarType2* __restrict b = real_tensor_2.storage().data(); \
      const auto size = tensor_1.storage().size(); \
      for (Size j = 0; j < size; j++) { \
         EVAL2; \
      } \
      return tensor_1; \
   } \
   template<is_scalar ScalarType1, is_scalar ScalarType2, is_symmetry Symmetry, is_name Name> \
      requires(is_complex<ScalarType1> || is_real<ScalarType2>) \
   Tensor<ScalarType1, Symmetry, Name> \
   &OP(Tensor<ScalarType1, Symmetry, Name>& tensor_1, const ScalarType2& number_2) { \
      auto timer_guard = scalar_inplace_guard(); \
      if (tensor_1.core.use_count() != 1) [[unlikely]] { \
         tensor_1.core = std::make_shared<Core<ScalarType1, Symmetry>>(*tensor_1.core); \
         detail::what_if_copy_shared("Inplace operator on tensor shared, copy happened here"); \
      } \
      ScalarType1* __restrict a = tensor_1.storage().data(); \
      const auto& y = number_2; \
      const auto size = tensor_1.storage().size(); \
      for (Size j = 0; j < size; j++) { \
         EVAL1; \
      } \
      return tensor_1; \
   }
   TAT_DEFINE_SCALAR_OPERATOR(operator+=, a[j] += y, a[j] += b[j])
   TAT_DEFINE_SCALAR_OPERATOR(operator-=, a[j] -= y, a[j] -= b[j])
   TAT_DEFINE_SCALAR_OPERATOR(operator*=, a[j] *= y, a[j] *= b[j])
   TAT_DEFINE_SCALAR_OPERATOR(operator/=, a[j] /= y, a[j] /= b[j])
#undef TAT_DEFINE_SCALAR_OPERATOR
   /**@}*/
} // namespace TAT
#endif
