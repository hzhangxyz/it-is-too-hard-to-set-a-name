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
         TAT_error("Try to do scalar operator on two different rank tensor"); \
      } \
      auto real_tensor_2_pointer = &tensor_2; \
      auto new_tensor_2 = Tensor<ScalarType2, Symmetry, Name>(); \
      if (tensor_1.names != tensor_2.names) [[unlikely]] { \
         new_tensor_2 = tensor_2.transpose(tensor_1.names); \
         real_tensor_2_pointer = &new_tensor_2; \
      } \
      const auto& real_tensor_2 = *real_tensor_2_pointer; \
      if (tensor_1.core->edges != real_tensor_2.core->edges) [[unlikely]] { \
         auto new_result_edge = decltype(tensor_1.core->edges)(); \
         new_result_edge.reserve(tensor_1.names.size()); \
         for (Rank i = 0; i < tensor_1.names.size(); i++) { \
            auto& single_new_edge = new_result_edge.emplace_back(tensor_1.core->edges.at(i)); \
            for (auto [symmetry, dimension] : real_tensor_2.core->edges.at(i).map) { \
               auto found = map_find(single_new_edge.map, symmetry); \
               if (found == single_new_edge.map.end()) [[unlikely]] { \
                  single_new_edge.map.emplace_back(symmetry, dimension); \
                  std::inplace_merge( \
                        single_new_edge.map.begin(), \
                        std::prev(single_new_edge.map.end()), \
                        single_new_edge.map.end(), \
                        [](const auto& a, const auto& b) { \
                           return a.first < b.first; \
                        }); \
               } else if (found->second != dimension) [[unlikely]] { \
                  TAT_error("Try to do scalar operator on two tensors which edges not compatible"); \
               } \
            } \
         } \
\
         const ScalarType x = 0; \
         const ScalarType y = 0; \
         auto result = Tensor<ScalarType, Symmetry, Name>{tensor_1.names, std::move(new_result_edge)}; \
         for (auto& [symmetries, block] : result.core->blocks) { \
            auto found_1 = map_find(tensor_1.core->blocks, symmetries); \
            auto found_2 = map_find(real_tensor_2.core->blocks, symmetries); \
            if (found_1 != tensor_1.core->blocks.end()) { \
               if (found_2 != real_tensor_2.core->blocks.end()) { \
                  const ScalarType1* __restrict a = map_at(tensor_1.core->blocks, symmetries).data(); \
                  const ScalarType2* __restrict b = map_at(real_tensor_2.core->blocks, symmetries).data(); \
                  ScalarType* __restrict c = block.data(); \
                  for (Size j = 0; j < block.size(); j++) { \
                     EVAL3; \
                  } \
               } else { \
                  const ScalarType1* __restrict a = map_at(tensor_1.core->blocks, symmetries).data(); \
                  ScalarType* __restrict c = block.data(); \
                  for (Size j = 0; j < block.size(); j++) { \
                     EVAL2; \
                  } \
               } \
            } else { \
               if (found_2 != real_tensor_2.core->blocks.end()) { \
                  const ScalarType2* __restrict b = map_at(real_tensor_2.core->blocks, symmetries).data(); \
                  ScalarType* __restrict c = block.data(); \
                  for (Size j = 0; j < block.size(); j++) { \
                     EVAL1; \
                  } \
               } else { \
                  std::fill(block.begin(), block.end(), 0); \
               } \
            } \
         } \
         return result; \
      } else [[likely]] { \
         auto result = Tensor<ScalarType, Symmetry, Name>{tensor_1.names, tensor_1.core->edges}; \
         const ScalarType1* __restrict a = tensor_1.core->storage.data(); \
         const ScalarType2* __restrict b = real_tensor_2.core->storage.data(); \
         ScalarType* __restrict c = result.core->storage.data(); \
         const auto size = result.core->storage.size(); \
         for (Size j = 0; j < size; j++) { \
            EVAL3; \
         } \
         return result; \
      } \
   } \
   template<is_scalar ScalarType1, is_scalar ScalarType2, is_symmetry Symmetry, is_name Name> \
   [[nodiscard]] auto OP(const Tensor<ScalarType1, Symmetry, Name>& tensor_1, const ScalarType2& number_2) { \
      auto timer_guard = scalar_outplace_guard(); \
      using ScalarType = std::common_type_t<ScalarType1, ScalarType2>; \
      const ScalarType1* __restrict a = tensor_1.core->storage.data(); \
      const auto& y = number_2; \
      auto result = Tensor<ScalarType, Symmetry, Name>{tensor_1.names, tensor_1.core->edges}; \
      ScalarType* __restrict c = result.core->storage.data(); \
      const auto size = result.core->storage.size(); \
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
      const ScalarType2* __restrict b = tensor_2.core->storage.data(); \
      auto result = Tensor<ScalarType, Symmetry, Name>{tensor_2.names, tensor_2.core->edges}; \
      ScalarType* __restrict c = result.core->storage.data(); \
      const auto size = result.core->storage.size(); \
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
         TAT_warning_or_error_when_copy_shared("Inplace operator on tensor shared, copy happened here"); \
      } \
      auto real_tensor_2_pointer = &tensor_2; \
      auto new_tensor_2 = Tensor<ScalarType2, Symmetry, Name>(); \
      if (tensor_1.names != tensor_2.names) [[unlikely]] { \
         new_tensor_2 = tensor_2.transpose(tensor_1.names); \
         real_tensor_2_pointer = &new_tensor_2; \
      } \
      const auto& real_tensor_2 = *real_tensor_2_pointer; \
      if (tensor_1.core->edges != real_tensor_2.core->edges) [[unlikely]] { \
         TAT_error("Scalar Operator In Different Shape Tensor, Maybe You Need Outplace Operator"); \
      } \
      ScalarType1* __restrict a = tensor_1.core->storage.data(); \
      const ScalarType2* __restrict b = real_tensor_2.core->storage.data(); \
      const auto size = tensor_1.core->storage.size(); \
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
         TAT_warning_or_error_when_copy_shared("Inplace operator on tensor shared, copy happened here"); \
      } \
      ScalarType1* __restrict a = tensor_1.core->storage.data(); \
      const auto& y = number_2; \
      const auto size = tensor_1.core->storage.size(); \
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
