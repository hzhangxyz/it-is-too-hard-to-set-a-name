/**
 * \file transpose.hpp
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
#ifndef TAT_TRANSPOSE_HPP
#define TAT_TRANSPOSE_HPP

#include <tuple>

#include "../utility/const_integral.hpp"

#ifndef TAT_DOXYGEN_SHOULD_SKIP_THIS
#ifdef TAT_USE_MKL_TRANSPOSE
extern "C" {
   void mkl_somatcopy_(const char*, const char*, const int*, const int*, const float*, const float*, const int*, float*, const int*);
   void mkl_domatcopy_(const char*, const char*, const int*, const int*, const double*, const double*, const int*, double*, const int*);
   void mkl_comatcopy_(
         const char*,
         const char*,
         const int*,
         const int*,
         const std::complex<float>*,
         const std::complex<float>*,
         const int*,
         std::complex<float>*,
         const int*);
   void mkl_zomatcopy_(
         const char*,
         const char*,
         const int*,
         const int*,
         const std::complex<double>*,
         const std::complex<double>*,
         const int*,
         const std::complex<double>*,
         const int*);
}
#endif
#endif

namespace TAT {
   constexpr unsigned long l1_cache =
#ifdef TAT_L1_CACHE
         TAT_L1_CACHE
#else
         98304
#endif
         ;

   constexpr unsigned long l2_cache =
#ifdef TAT_L2_CACHE
         TAT_L2_CACHE
#else
         786432
#endif
         ;

   constexpr unsigned long l3_cache =
#ifdef TAT_L3_CACHE
         TAT_L3_CACHE
#else
         4718592
#endif
         ;

#ifndef TAT_DOXYGEN_SHOULD_SKIP_THIS
   template<typename ScalarType>
   void mkl_transpose(
         int dimension_of_M,
         int dimension_of_N,
         const ScalarType* data_source,
         ScalarType* data_destination,
         int leading_source,
         int leading_destination,
         ScalarType alpha);

#ifdef TAT_USE_MKL_TRANSPOSE
   template<>
   inline void mkl_transpose<float>(
         const int dimension_of_M,
         const int dimension_of_N,
         const float* const data_source,
         float* const data_destination,
         const int leading_source,
         const int leading_destination,
         const float alpha) {
      mkl_somatcopy_("R", "T", &dimension_of_M, &dimension_of_N, &alpha, data_source, &leading_source, data_destination, &leading_destination);
   }
   template<>
   inline void mkl_transpose<double>(
         const int dimension_of_M,
         const int dimension_of_N,
         const double* const data_source,
         double* const data_destination,
         const int leading_source,
         const int leading_destination,
         const double alpha) {
      mkl_domatcopy_("R", "T", &dimension_of_M, &dimension_of_N, &alpha, data_source, &leading_source, data_destination, &leading_destination);
   }
   template<>
   inline void mkl_transpose<std::complex<float>>(
         const int dimension_of_M,
         const int dimension_of_N,
         const std::complex<float>* const data_source,
         std::complex<float>* const data_destination,
         const int leading_source,
         const int leading_destination,
         const std::complex<float> alpha) {
      mkl_comatcopy_("R", "T", &dimension_of_M, &dimension_of_N, &alpha, data_source, &leading_source, data_destination, &leading_destination);
   }
   template<>
   inline void mkl_transpose<std::complex<double>>(
         const int dimension_of_M,
         const int dimension_of_N,
         const std::complex<double>* const data_source,
         std::complex<double>* const data_destination,
         const int leading_source,
         const int leading_destination,
         const std::complex<double> alpha) {
      mkl_zomatcopy_("R", "T", &dimension_of_M, &dimension_of_N, &alpha, data_source, &leading_source, data_destination, &leading_destination);
   }
#endif

   inline timer transpose_kernel_core_guard("transpose_kernel_core");

   // 这个是最简单的张量转置中实际搬运数据的部分，numpy也是这么写的，区别在于dimension和两个leading的顺序是可以一同交换的
   // numpy保证destination的leading是降序的， simple_transpose就是这么调用tensor_transpose_kernel的
   // 另外一个正在写的inturn_transpose是src dst轮流来, 可能会对cache更加友好, 日后还会根据cache大小split边，这样类似于矩阵转置中的预分块
   template<typename ScalarType, bool parity, bool loop_last = false, typename LineSizeType = int>
   void tensor_transpose_kernel(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const Size* const __restrict dimension,
         const Size* const __restrict leading_source,
         const Size* const __restrict leading_destination,
         const Rank rank,
         const Rank line_rank = 0,
         const LineSizeType line_size = 0) {
      auto timer_guard = transpose_kernel_core_guard();

      // 经过测试使用mkl的transpose有时会变慢
#if 0
#ifdef TAT_USE_MKL_TRANSPOSE
      if (rank == 2) [[unlikely]] {
         if (leading_source[1] == 1 && leading_destination[0] == 1) [[unlikely]] {
            mkl_transpose<ScalarType>(
                  dimension[0], dimension[1], data_source, data_destination, leading_source[0], leading_destination[1], parity ? -1 : 1);
            return;
         }
         if (leading_source[0] == 1 && leading_destination[1] == 1) [[unlikely]] {
            mkl_transpose<ScalarType>(
                  dimension[1], dimension[0], data_source, data_destination, leading_source[1], leading_destination[0], parity ? -1 : 1);
            return;
         }
      }
#endif
#endif

      const ScalarType* current_source = data_source;
      ScalarType* current_destination = data_destination;
      pmr::vector<Size> index_list(rank, 0);
      while (true) {
         Rank active_position = rank - 1;

         // TODO 小矩阵形式的特化
         if constexpr (loop_last) {
            // 只有最后的维度相同且leading为1的时候才会进入此分支
            // 如果只有最后一维的话, line_rank = rank-1
            active_position = line_rank;
            index_list[active_position] = dimension[active_position];
            const Size line_size_value = line_size.value();
            for (Size i = 0; i < line_size_value; i++) {
               if constexpr (parity) {
                  current_destination[i] = -current_source[i];
               } else {
                  current_destination[i] = current_source[i];
               }
            }
            current_source += line_size_value;
            current_destination += line_size_value;
         } else {
            if constexpr (parity) {
               *current_destination = -*current_source;
            } else {
               *current_destination = *current_source;
            }

            index_list[active_position]++;
            current_source += leading_source[active_position];
            current_destination += leading_destination[active_position];
         }

         while (index_list[active_position] == dimension[active_position]) {
            index_list[active_position] = 0;
            current_source -= dimension[active_position] * leading_source[active_position];
            current_destination -= dimension[active_position] * leading_destination[active_position];

            if (active_position == 0) [[unlikely]] {
               return;
            }
            active_position--;

            index_list[active_position]++;
            current_source += leading_source[active_position];
            current_destination += leading_destination[active_position];
         }
      }
   }

   // TODO 更好的转置方法?
   // TODO: l3太大了, 所以只按着l2和l1来划分, 这样合适么
   // 注意，现在这段代码被我暂时删掉了
   //
   // TODO 类似矩阵转置的优化方式优化张量转置
   // O...O...O.O..XXX
   //  X..X..X.....OOOO
   //              ^
   // 首先找到一段前后各自连续的部分，然后分块地做转置，分块的方案是cut掉这里的维度
   //
   // TODO block transpose
   // 现在的问题是虽然block的算法我能设计，但是换成block方式时带来的多判断一次的副作用就会导致转置慢很多
   // 按照cache来split边
   // 再交替iter dim
   // 即可兼容矩阵转置的优化方式

   template<typename ScalarType, bool parity>
   void simple_transpose(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const pmr::vector<Rank>& plan_source_to_destination,
         const pmr::vector<Rank>& plan_destination_to_source,
         const pmr::vector<Size>& dimensions_source,
         const pmr::vector<Size>& dimensions_destination,
         const pmr::vector<Size>& leadings_source,
         const pmr::vector<Size>& leadings_destination,
         const Rank rank) {
      auto leadings_source_by_destination = pmr::vector<Size>();
      leadings_source_by_destination.reserve(rank);
      for (auto i = 0; i < rank; i++) {
         auto j = plan_destination_to_source[i];
         leadings_source_by_destination.push_back(leadings_source[j]);
      }

      if (leadings_source_by_destination[rank - 1] == 1 && leadings_destination[rank - 1] == 1) {
         Rank line_rank = rank - 1;
         Size expect_leading = 1;
         while (expect_leading *= dimensions_destination[line_rank],
                leadings_source_by_destination[line_rank - 1] == expect_leading && leadings_destination[line_rank - 1] == expect_leading) {
            if (line_rank == 0) [[unlikely]] {
               // 完全线性copy
               break;
            }
            line_rank--;
         }
         const auto const_expect_leading_variant = to_const_integral<Size, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(expect_leading);
         std::visit(
               [&](const auto& const_expect_leading) {
                  tensor_transpose_kernel<ScalarType, parity, true>(
                        data_source,
                        data_destination,
                        dimensions_destination.data(),
                        leadings_source_by_destination.data(),
                        leadings_destination.data(),
                        rank,
                        line_rank,
                        const_expect_leading);
               },
               const_expect_leading_variant);
      } else {
         tensor_transpose_kernel<ScalarType, parity>(
               data_source,
               data_destination,
               dimensions_destination.data(),
               leadings_source_by_destination.data(),
               leadings_destination.data(),
               rank);
      }
   }

   template<typename ScalarType, bool parity>
   void inturn_transpose(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const pmr::vector<Rank>& plan_source_to_destination,
         const pmr::vector<Rank>& plan_destination_to_source,
         const pmr::vector<Size>& dimensions_source,
         const pmr::vector<Size>& dimensions_destination,
         const pmr::vector<Size>& leadings_source,
         const pmr::vector<Size>& leadings_destination,
         const Rank rank) {
      auto mask_source = pmr::vector<bool>(rank, false);
      auto mask_destination = pmr::vector<bool>(rank, false);
      auto real_dimensions = pmr::vector<Size>(rank);
      auto real_leadings_source = pmr::vector<Size>(rank);
      auto real_leadings_destination = pmr::vector<Size>(rank);

      bool source_exhausted = false;
      bool destination_exhausted = false;

      Rank current_index = rank;
      Rank current_index_source = rank - 1;
      Rank current_index_destination = rank - 1;

      auto use_source = [&]() {
         auto response_index_destination = plan_source_to_destination[current_index_source];

         real_dimensions[current_index] = dimensions_source[current_index_source];
         real_leadings_source[current_index] = leadings_source[current_index_source];
         real_leadings_destination[current_index] = leadings_destination[response_index_destination];

         mask_destination[response_index_destination] = true;
         while (mask_destination[current_index_destination]) {
            if (current_index_destination == 0) {
               destination_exhausted = true;
               break;
            }
            current_index_destination--;
         }

         do {
            if (current_index_source == 0) {
               source_exhausted = true;
               break;
            }
            current_index_source--;
         } while (mask_source[current_index_source]);
      };
      auto use_destination = [&]() {
         auto response_index_source = plan_destination_to_source[current_index_destination];

         real_dimensions[current_index] = dimensions_destination[current_index_destination];
         real_leadings_destination[current_index] = leadings_destination[current_index_destination];
         real_leadings_source[current_index] = leadings_source[response_index_source];

         mask_source[response_index_source] = true;
         while (mask_source[current_index_source]) {
            if (current_index_source == 0) {
               source_exhausted = true;
               break;
            }
            current_index_source--;
         }

         do {
            if (current_index_destination == 0) {
               destination_exhausted = true;
               break;
            }
            current_index_destination--;
         } while (mask_destination[current_index_destination]);
      };

      while (current_index-- > 0) {
         if (destination_exhausted) {
            use_source();
         } else if (source_exhausted) {
            use_destination();
         } else if (leadings_destination[current_index_destination] > leadings_source[current_index_source]) {
            use_source();
         } else {
            use_destination();
         }
      }

      if (real_leadings_source[rank - 1] == 1 && real_leadings_destination[rank - 1] == 1) {
         Rank line_rank = rank - 1;
         Size expect_leading = 1;
         while (expect_leading *= real_dimensions[line_rank],
                real_leadings_source[line_rank - 1] == expect_leading && real_leadings_destination[line_rank - 1] == expect_leading) {
            if (line_rank == 0) [[unlikely]] {
               // 完全线性copy
               break;
            }
            line_rank--;
         }
         auto const_expect_leading_variant = to_const_integral<Size, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(expect_leading);
         std::visit(
               [&](const auto& const_expect_leading) {
                  tensor_transpose_kernel<ScalarType, parity, true>(
                        data_source,
                        data_destination,
                        real_dimensions.data(),
                        real_leadings_source.data(),
                        real_leadings_destination.data(),
                        rank,
                        line_rank,
                        const_expect_leading);
               },
               const_expect_leading_variant);
      } else {
         tensor_transpose_kernel<ScalarType, parity>(
               data_source,
               data_destination,
               real_dimensions.data(),
               real_leadings_source.data(),
               real_leadings_destination.data(),
               rank);
      }
   }

   inline timer transpose_kernel_guard("transpose_kernel");

   template<typename ScalarType>
   void do_transpose(
         const ScalarType* data_source,
         ScalarType* data_destination,
         const pmr::vector<Rank>& plan_source_to_destination,
         const pmr::vector<Rank>& plan_destination_to_source,
         const pmr::vector<Size>& dimensions_source,
         const pmr::vector<Size>& dimensions_destination,
         const pmr::vector<Size>& leadings_source,
         const pmr::vector<Size>& leadings_destination,
         Rank rank,
         Size total_size,
         bool parity) {
      auto timer_guard = transpose_kernel_guard();

      if (total_size == 0) {
         return;
      }
      if (total_size == 1) {
         if (parity) {
            *data_destination = -*data_source;
         } else {
            *data_destination = *data_source;
         }
         return;
      }
      // rank != 0, dimension != 0

      if (parity) {
         // 小于l1_cache的系统不需要inturn
         if (total_size * sizeof(ScalarType) < l1_cache) {
            simple_transpose<ScalarType, true>(
                  data_source,
                  data_destination,
                  plan_source_to_destination,
                  plan_destination_to_source,
                  dimensions_source,
                  dimensions_destination,
                  leadings_source,
                  leadings_destination,
                  rank);
         } else {
            inturn_transpose<ScalarType, true>(
                  data_source,
                  data_destination,
                  plan_source_to_destination,
                  plan_destination_to_source,
                  dimensions_source,
                  dimensions_destination,
                  leadings_source,
                  leadings_destination,
                  rank);
         }
      } else {
         if (total_size * sizeof(ScalarType) < l1_cache) {
            simple_transpose<ScalarType, false>(
                  data_source,
                  data_destination,
                  plan_source_to_destination,
                  plan_destination_to_source,
                  dimensions_source,
                  dimensions_destination,
                  leadings_source,
                  leadings_destination,
                  rank);
         } else {
            inturn_transpose<ScalarType, false>(
                  data_source,
                  data_destination,
                  plan_source_to_destination,
                  plan_destination_to_source,
                  dimensions_source,
                  dimensions_destination,
                  leadings_source,
                  leadings_destination,
                  rank);
         }
      }
   }

   template<typename ScalarType>
   void matrix_transpose(Size m, Size n, const ScalarType* const source, ScalarType* const destination) {
      auto dimension = pmr::vector<Size>{m, n};
      auto leading_source = pmr::vector<Size>{n, 1};
      auto leading_destination = pmr::vector<Size>{1, m};
      tensor_transpose_kernel<ScalarType, false>(source, destination, dimension.data(), leading_source.data(), leading_destination.data(), 2);
   }
#endif
} // namespace TAT
#endif
