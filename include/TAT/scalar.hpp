/**
 * \file scalar.hpp
 *
 * Copyright (C) 2019-2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include "tensor.hpp"

namespace TAT {
#define TAT_DEFINE_SCALAR_OPERATOR(OP, EVAL1, EVAL2, EVAL3)                                                           \
   template<class ScalarType1, class ScalarType2, class Symmetry>                                                     \
   auto OP(const Tensor<ScalarType1, Symmetry>& tensor_1, const Tensor<ScalarType2, Symmetry>& tensor_2) {            \
      using ScalarType = std::common_type_t<ScalarType1, ScalarType2>;                                                \
      if (tensor_1.names.empty()) {                                                                                   \
         const auto& x = tensor_1.at({});                                                                             \
         auto result = Tensor<ScalarType, Symmetry>{tensor_2.names, tensor_2.core->edges};                            \
         for (auto& [symmetries, block] : result.core->blocks) {                                                      \
            const ScalarType2* __restrict b = tensor_2.core->blocks[symmetries].data();                               \
            ScalarType* __restrict c = block.data();                                                                  \
            for (Size j = 0; j < block.size(); j++) {                                                                 \
               EVAL1;                                                                                                 \
            }                                                                                                         \
         }                                                                                                            \
         return result;                                                                                               \
      } else if (tensor_2.names.empty()) {                                                                            \
         const auto& y = tensor_2.at({});                                                                             \
         auto result = Tensor<ScalarType, Symmetry>{tensor_1.names, tensor_1.core->edges};                            \
         for (auto& [symmetries, block] : result.core->blocks) {                                                      \
            const ScalarType1* __restrict a = tensor_1.core->blocks[symmetries].data();                               \
            ScalarType* __restrict c = block.data();                                                                  \
            for (Size j = 0; j < block.size(); j++) {                                                                 \
               EVAL2;                                                                                                 \
            }                                                                                                         \
         }                                                                                                            \
         return result;                                                                                               \
      } else {                                                                                                        \
         auto real_tensor_2 = &tensor_2;                                                                              \
         auto new_tensor_2 = Tensor<ScalarType2, Symmetry>();                                                         \
         if (tensor_1.names != tensor_2.names) {                                                                      \
            new_tensor_2 = tensor_2.transpose(tensor_1.names);                                                        \
            real_tensor_2 = &new_tensor_2;                                                                            \
         }                                                                                                            \
         auto real_result_edge = &tensor_1.core->edges;                                                               \
         auto new_result_edge = std::vector<Edge<Symmetry>>();                                                        \
         if (tensor_1.core->edges != real_tensor_2->core->edges) {                                                    \
            for (auto i = 0; i < tensor_1.names.size(); i++) {                                                        \
               new_result_edge.push_back(tensor_1.core->edges[i]);                                                    \
               for (auto [symmetry, dimension] : real_tensor_2->core->edges[i].map) {                                 \
                  new_result_edge.back().map[symmetry] = dimension;                                                   \
               }                                                                                                      \
            }                                                                                                         \
            real_result_edge = &new_result_edge;                                                                      \
         }                                                                                                            \
         const ScalarType x = 0;                                                                                      \
         const ScalarType y = 0;                                                                                      \
         auto result = Tensor<ScalarType, Symmetry>{tensor_1.names, *real_result_edge};                               \
         for (auto& [symmetries, block] : result.core->blocks) {                                                      \
            auto found_1 = tensor_1.core->blocks.find(symmetries);                                                    \
            auto found_2 = real_tensor_2->core->blocks.find(symmetries);                                              \
            if (found_1 != tensor_1.core->blocks.end()) {                                                             \
               if (found_2 != real_tensor_2->core->blocks.end()) {                                                    \
                  const ScalarType1* __restrict a = tensor_1.core->blocks[symmetries].data();                         \
                  const ScalarType2* __restrict b = real_tensor_2->core->blocks[symmetries].data();                   \
                  ScalarType* __restrict c = block.data();                                                            \
                  for (Size j = 0; j < block.size(); j++) {                                                           \
                     EVAL3;                                                                                           \
                  }                                                                                                   \
               } else {                                                                                               \
                  const ScalarType1* __restrict a = tensor_1.core->blocks[symmetries].data();                         \
                  ScalarType* __restrict c = block.data();                                                            \
                  for (Size j = 0; j < block.size(); j++) {                                                           \
                     EVAL2;                                                                                           \
                  }                                                                                                   \
               }                                                                                                      \
            } else {                                                                                                  \
               if (found_2 != real_tensor_2->core->blocks.end()) {                                                    \
                  const ScalarType2* __restrict b = real_tensor_2->core->blocks[symmetries].data();                   \
                  ScalarType* __restrict c = block.data();                                                            \
                  for (Size j = 0; j < block.size(); j++) {                                                           \
                     EVAL1;                                                                                           \
                  }                                                                                                   \
               } else {                                                                                               \
                  std::fill(block.begin(), block.end(), 0);                                                           \
               }                                                                                                      \
            }                                                                                                         \
         }                                                                                                            \
         return result;                                                                                               \
      }                                                                                                               \
   }                                                                                                                  \
   template<class ScalarType1, class ScalarType2, class Symmetry, class = std::enable_if_t<is_scalar_v<ScalarType2>>> \
   auto OP(const Tensor<ScalarType1, Symmetry>& tensor_1, const ScalarType2& number_2) {                              \
      return OP(tensor_1, Tensor<ScalarType2, Symmetry>{number_2});                                                   \
   }                                                                                                                  \
   template<class ScalarType1, class ScalarType2, class Symmetry, class = std::enable_if_t<is_scalar_v<ScalarType1>>> \
   auto OP(const ScalarType1& number_1, const Tensor<ScalarType2, Symmetry>& tensor_2) {                              \
      return OP(Tensor<ScalarType1, Symmetry>{number_1}, tensor_2);                                                   \
   }

   TAT_DEFINE_SCALAR_OPERATOR(operator+, c[j] = x + b[j], c[j] = a[j] + y, c[j] = a[j] + b[j])
   TAT_DEFINE_SCALAR_OPERATOR(operator-, c[j] = x - b[j], c[j] = a[j] - y, c[j] = a[j] - b[j])
   TAT_DEFINE_SCALAR_OPERATOR(operator*, c[j] = x * b[j], c[j] = a[j] * y, c[j] = a[j] * b[j])
   TAT_DEFINE_SCALAR_OPERATOR(operator/, c[j] = x / b[j], c[j] = a[j] / y, c[j] = a[j] / b[j])
#undef TAT_DEFINE_SCALAR_OPERATOR

#define TAT_DEFINE_SCALAR_OPERATOR(OP, EVAL1, EVAL2)                                                                           \
   template<class ScalarType1, class ScalarType2, class Symmetry>                                                              \
   Tensor<ScalarType1, Symmetry>& OP(Tensor<ScalarType1, Symmetry>& tensor_1, const Tensor<ScalarType2, Symmetry>& tensor_2) { \
      if (tensor_1.core.use_count() != 1) {                                                                                    \
         warning_or_error("Inplace Operator On Tensor Shared");                                                                \
      }                                                                                                                        \
      if (tensor_2.names.empty()) {                                                                                            \
         const auto& y = tensor_2.at({});                                                                                      \
         for (auto& [symmetries, block] : tensor_1.core->blocks) {                                                             \
            ScalarType1* __restrict a = block.data();                                                                          \
            for (Size j = 0; j < block.size(); j++) {                                                                          \
               EVAL1;                                                                                                          \
            }                                                                                                                  \
         }                                                                                                                     \
      } else {                                                                                                                 \
         auto real_tensor_2 = &tensor_2;                                                                                       \
         auto new_tensor_2 = Tensor<ScalarType2, Symmetry>();                                                                  \
         if (tensor_1.names != tensor_2.names) {                                                                               \
            new_tensor_2 = tensor_2.transpose(tensor_1.names);                                                                 \
            real_tensor_2 = &new_tensor_2;                                                                                     \
         }                                                                                                                     \
         if (tensor_1.core->edges != real_tensor_2->core->edges) {                                                             \
            warning_or_error("Scalar Operator In Different Shape Tensor, Maybe You Need Outplace Operator");                   \
         }                                                                                                                     \
         for (auto& [symmetries, block] : tensor_1.core->blocks) {                                                             \
            ScalarType1* __restrict a = block.data();                                                                          \
            const ScalarType2* __restrict b = real_tensor_2->core->blocks[symmetries].data();                                  \
            for (Size j = 0; j < block.size(); j++) {                                                                          \
               EVAL2;                                                                                                          \
            }                                                                                                                  \
         }                                                                                                                     \
      }                                                                                                                        \
      return tensor_1;                                                                                                         \
   }                                                                                                                           \
   template<class ScalarType1, class ScalarType2, class Symmetry, class = std::enable_if_t<is_scalar_v<ScalarType2>>>          \
   Tensor<ScalarType1, Symmetry>& OP(Tensor<ScalarType1, Symmetry>& tensor_1, const ScalarType2& number_2) {                   \
      return OP(tensor_1, Tensor<ScalarType2, Symmetry>{number_2});                                                            \
   }
   TAT_DEFINE_SCALAR_OPERATOR(operator+=, a[j] += y, a[j] += b[j])
   TAT_DEFINE_SCALAR_OPERATOR(operator-=, a[j] -= y, a[j] -= b[j])
   TAT_DEFINE_SCALAR_OPERATOR(operator*=, a[j] *= y, a[j] *= b[j])
   TAT_DEFINE_SCALAR_OPERATOR(operator/=, a[j] /= y, a[j] /= b[j])
#undef TAT_DEFINE_SCALAR_OPERATOR
} // namespace TAT
#endif
