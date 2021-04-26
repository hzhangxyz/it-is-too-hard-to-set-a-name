/**
 * \file qr.hpp
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
#ifndef TAT_QR_HPP
#define TAT_QR_HPP

#include "../structure/tensor.hpp"
#include "../utility/timer.hpp"
#include "transpose.hpp"

#ifndef TAT_DOXYGEN_SHOULD_SKIP_THIS
extern "C" {
   void sgeqrf_(const int* m, const int* n, float* A, const int* lda, float* tau, float* work, const int* lwork, int* info);
   void dgeqrf_(const int* m, const int* n, double* A, const int* lda, double* tau, double* work, const int* lwork, int* info);
   void cgeqrf_(
         const int* m,
         const int* n,
         std::complex<float>* A,
         const int* lda,
         std::complex<float>* tau,
         std::complex<float>* work,
         const int* lwork,
         int* info);
   void zgeqrf_(
         const int* m,
         const int* n,
         std::complex<double>* A,
         const int* lda,
         std::complex<double>* tau,
         std::complex<double>* work,
         const int* lwork,
         int* info);
   void sgelqf_(const int* m, const int* n, float* A, const int* lda, float* tau, float* work, const int* lwork, int* info);
   void dgelqf_(const int* m, const int* n, double* A, const int* lda, double* tau, double* work, const int* lwork, int* info);
   void cgelqf_(
         const int* m,
         const int* n,
         std::complex<float>* A,
         const int* lda,
         std::complex<float>* tau,
         std::complex<float>* work,
         const int* lwork,
         int* info);
   void zgelqf_(
         const int* m,
         const int* n,
         std::complex<double>* A,
         const int* lda,
         std::complex<double>* tau,
         std::complex<double>* work,
         const int* lwork,
         int* info);
   void sorgqr_(const int* m, const int* n, const int* k, float* A, const int* lda, float const* tau, float* work, const int* lwork, int* info);
   void dorgqr_(const int* m, const int* n, const int* k, double* A, const int* lda, double const* tau, double* work, const int* lwork, int* info);
   void cungqr_(
         const int* m,
         const int* n,
         const int* k,
         std::complex<float>* A,
         const int* lda,
         std::complex<float> const* tau,
         std::complex<float>* work,
         const int* lwork,
         int* info);
   void zungqr_(
         const int* m,
         const int* n,
         const int* k,
         std::complex<double>* A,
         const int* lda,
         std::complex<double> const* tau,
         std::complex<double>* work,
         const int* lwork,
         int* info);
   void sorglq_(const int* m, const int* n, const int* k, float* A, const int* lda, float const* tau, float* work, const int* lwork, int* info);
   void dorglq_(const int* m, const int* n, const int* k, double* A, const int* lda, double const* tau, double* work, const int* lwork, int* info);
   void cunglq_(
         const int* m,
         const int* n,
         const int* k,
         std::complex<float>* A,
         const int* lda,
         std::complex<float> const* tau,
         std::complex<float>* work,
         const int* lwork,
         int* info);
   void zunglq_(
         const int* m,
         const int* n,
         const int* k,
         std::complex<double>* A,
         const int* lda,
         std::complex<double> const* tau,
         std::complex<double>* work,
         const int* lwork,
         int* info);
}
#endif

namespace TAT {
#ifndef TAT_DOXYGEN_SHOULD_SKIP_THIS
   template<typename ScalarType>
   constexpr void (
         *geqrf)(const int* m, const int* n, ScalarType* A, const int* lda, ScalarType* tau, ScalarType* work, const int* lwork, int* info) = nullptr;
   template<>
   inline auto geqrf<float> = sgeqrf_;
   template<>
   inline auto geqrf<double> = dgeqrf_;
   template<>
   inline auto geqrf<std::complex<float>> = cgeqrf_;
   template<>
   inline auto geqrf<std::complex<double>> = zgeqrf_;
   template<typename ScalarType>
   constexpr void (
         *gelqf)(const int* m, const int* n, ScalarType* A, const int* lda, ScalarType* tau, ScalarType* work, const int* lwork, int* info) = nullptr;
   template<>
   inline auto gelqf<float> = sgelqf_;
   template<>
   inline auto gelqf<double> = dgelqf_;
   template<>
   inline auto gelqf<std::complex<float>> = cgelqf_;
   template<>
   inline auto gelqf<std::complex<double>> = zgelqf_;
   template<typename ScalarType>
   constexpr void (*orgqr)(
         const int* m,
         const int* n,
         const int* k,
         ScalarType* A,
         const int* lda,
         ScalarType* tau,
         ScalarType* work,
         const int* lwork,
         int* info) = nullptr;
   template<>
   inline auto orgqr<float> = sorgqr_;
   template<>
   inline auto orgqr<double> = dorgqr_;
   template<>
   inline auto orgqr<std::complex<float>> = cungqr_;
   template<>
   inline auto orgqr<std::complex<double>> = zungqr_;
   template<typename ScalarType>
   constexpr void (*orglq)(
         const int* m,
         const int* n,
         const int* k,
         ScalarType* A,
         const int* lda,
         ScalarType* tau,
         ScalarType* work,
         const int* lwork,
         int* info) = nullptr;
   template<>
   inline auto orglq<float> = sorglq_;
   template<>
   inline auto orglq<double> = dorglq_;
   template<>
   inline auto orglq<std::complex<float>> = cunglq_;
   template<>
   inline auto orglq<std::complex<double>> = zunglq_;

   template<typename ScalarType>
   int to_int(const ScalarType& value) {
      if constexpr (is_complex<ScalarType>) {
         return int(value.real());
      } else {
         return int(value);
      }
   }

   template<typename ScalarType>
   void calculate_qr_kernel(
         const int& m,
         const int& n,
         const int& min,
         const int& max,
         ScalarType* __restrict data,
         ScalarType* __restrict data_1,
         ScalarType* __restrict data_2,
         bool use_qr_not_lq) {
      // m*n c matrix at data do lq
      // n*m fortran matrix at data do qr
      if (use_qr_not_lq) {
         // c qr -> fortran lq
         // LQ
         //
         // XX   X        XQ
         // XX   XX XX    XX
         // XX = XX XX -> XX
         //
         // XXX   X  XXX    XQQ
         // XXX = XX XXX -> XXQ
         int result;
         auto tau = pmr::content_vector<ScalarType>(min);
         const int lwork_query = -1;
         ScalarType float_lwork;
         gelqf<ScalarType>(&n, &m, data, &n, tau.data(), &float_lwork, &lwork_query, &result);
         if (result != 0) {
            TAT_error("Error in LQ");
         }
         const int lwork = to_int(float_lwork);
         auto work = pmr::content_vector<ScalarType>(lwork);
         gelqf<ScalarType>(&n, &m, data, &n, tau.data(), work.data(), &lwork, &result);
         if (result != 0) {
            TAT_error("Error in LQ");
         }
         // Q matrix
         // data n*m
         // data_1 min*m
         for (auto i = 0; i < m; i++) {
            std::copy(data + i * n, data + i * n + min, data_1 + i * min);
         }
         orglq<ScalarType>(&min, &m, &min, data_1, &min, tau.data(), work.data(), &lwork, &result);
         // WRONG -> orglq<ScalarType>(&min, &min, &min, data_1, &min, tau.data(), work.data(), &lwork, &result);
         if (result != 0) {
            TAT_error("Error in LQ");
         }
         // L matrix
         for (auto i = 0; i < min; i++) {
            std::fill(data_2 + i * n, data_2 + i * n + i, 0);
            std::copy(data + i * n + i, data + i * n + n, data_2 + i * n + i);
         }
      } else {
         // c lq -> fortran qr
         // QR
         //
         // XX   XX       XX
         // XX   XX XX    QX
         // XX = XX  X -> QQ
         //
         // XXX   XX XXX    XXX
         // XXX = XX  XX -> QXX
         int result;
         auto tau = pmr::content_vector<ScalarType>(min);
         const int lwork_query = -1;
         ScalarType float_lwork;
         geqrf<ScalarType>(&n, &m, data, &n, tau.data(), &float_lwork, &lwork_query, &result);
         if (result != 0) {
            TAT_error("Error in LQ");
         }
         const int lwork = to_int(float_lwork);
         auto work = pmr::content_vector<ScalarType>(lwork);
         geqrf<ScalarType>(&n, &m, data, &n, tau.data(), work.data(), &lwork, &result);
         if (result != 0) {
            TAT_error("Error in QR");
         }
         // Q matrix
         std::copy(data, data + n * min, data_2); // 多复制了无用的上三角部分
         // fortran
         // data n*m
         // data_2 n*min
         orgqr<ScalarType>(&n, &min, &min, data_2, &n, tau.data(), work.data(), &lwork, &result);
         // WRONG -> orgqr<ScalarType>(&min, &min, &min, data_2, &n, tau.data(), work.data(), &lwork, &result);
         // same size of lwork
         if (result != 0) {
            TAT_error("Error in QR");
         }
         // R matrix
         for (auto i = 0; i < min; i++) {
            std::copy(data + n * i, data + n * i + i + 1, data_1 + min * i);
            std::fill(data_1 + min * i + i + 1, data_1 + min * i + min, 0);
         }
         std::copy(data + n * min, data + n * m, data_1 + min * min);
         // 若为第一种, 则这个copy不做事
      }
   }

   inline timer qr_kernel_guard("qr_kernel");

   template<typename ScalarType>
   void calculate_qr(
         const int& m,
         const int& n,
         const int& min,
         const int& max,
         ScalarType* __restrict data,
         ScalarType* __restrict data_1,
         ScalarType* __restrict data_2,
         bool use_qr_not_lq) {
      auto kernel_guard = qr_kernel_guard();
      // 有时可能多转置一下更快，参见svd中的做法
      // 经过初步测试m > n看起来最好
      if (m > n) {
         auto new_data = pmr::content_vector<ScalarType>(n * m);
         auto old_data_1 = pmr::content_vector<ScalarType>(n * min);
         auto old_data_2 = pmr::content_vector<ScalarType>(min * m);
         matrix_transpose(m, n, data, new_data.data());
         calculate_qr_kernel(n, m, min, max, new_data.data(), old_data_1.data(), old_data_2.data(), !use_qr_not_lq);
         matrix_transpose(n, min, old_data_1.data(), data_2);
         matrix_transpose(min, m, old_data_2.data(), data_1);
      } else {
         calculate_qr_kernel(m, n, min, max, data, data_1, data_2, use_qr_not_lq);
      }
   }
#endif

   inline timer qr_guard("qr");

   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   typename Tensor<ScalarType, Symmetry, Name>::qr_result Tensor<ScalarType, Symmetry, Name>::qr_implement(
         char free_name_direction,
         const auto& free_name_set,
         const Name& common_name_q,
         const Name& common_name_r) const {
      auto timer_guard = qr_guard();
      // free_name_set不需要做特殊处理即可自动处理不准确的边名
      constexpr bool is_fermi = Symmetry::is_fermi_symmetry;
      const auto rank = names.size();
      // 判断使用lq还是qr
      bool use_r_name;
      if (free_name_direction == 'r' || free_name_direction == 'R') {
         use_r_name = true;
      } else if (free_name_direction == 'q' || free_name_direction == 'Q') {
         use_r_name = false;
      } else {
         TAT_error("Invalid direction in QR");
      };
      bool use_qr_not_lq = names.empty() || ((set_find(free_name_set, names.back()) != free_name_set.end()) == use_r_name);
      // merge
      auto free_name_1 = pmr::vector<Name>();
      auto free_name_2 = pmr::vector<Name>();
      auto reversed_set_1 = pmr::set<Name>();
      auto reversed_set_2 = pmr::set<Name>();
      auto reversed_set_origin = pmr::set<Name>();
      auto result_name_1 = std::vector<Name>();
      auto result_name_2 = std::vector<Name>();
      auto free_names_and_edges_1 = pmr::vector<std::tuple<Name, edge_map_t<Symmetry, true>>>();
      auto free_names_and_edges_2 = pmr::vector<std::tuple<Name, edge_map_t<Symmetry, true>>>();
      free_name_1.reserve(rank);
      free_name_2.reserve(rank);
      result_name_1.reserve(rank + 1);
      result_name_2.reserve(rank + 1);
      free_names_and_edges_1.reserve(rank);
      free_names_and_edges_2.reserve(rank);
      result_name_2.push_back(use_qr_not_lq ? common_name_r : common_name_q);
      for (Rank i = 0; i < names.size(); i++) {
         const auto& n = names[i];
         // set.find() != set.end() => n in the set
         // (!=) == use_r_name => n in the r name
         // (!=) == use_r_name == use_qr_not_lq => in the second name
         if ((set_find(free_name_set, n) != free_name_set.end()) == use_r_name == use_qr_not_lq) {
            free_name_2.push_back(n);
            result_name_2.push_back(n);
            free_names_and_edges_2.push_back({n, {core->edges[i].map}});
            if constexpr (is_fermi) {
               if (core->edges[i].arrow) {
                  reversed_set_2.insert(n);
                  reversed_set_origin.insert(n);
               }
            }
         } else {
            free_name_1.push_back(n);
            result_name_1.push_back(n);
            free_names_and_edges_1.push_back({n, {core->edges[i].map}});
            if constexpr (is_fermi) {
               if (core->edges[i].arrow) {
                  reversed_set_1.insert(n);
                  reversed_set_origin.insert(n);
               }
            }
         }
      }
      if (use_r_name == use_qr_not_lq) {
         // set is the second name
         if (free_name_2.size() != free_name_set.size()) {
            TAT_warning_or_error_when_name_missing("Name missing in QR");
         }
      } else {
         // set is the first name
         if (free_name_1.size() != free_name_set.size()) {
            TAT_warning_or_error_when_name_missing("Name missing in QR");
         }
      }
      result_name_1.push_back(use_qr_not_lq ? common_name_q : common_name_r);
      auto tensor_merged = edge_operator_implement(
            empty_list<std::pair<Name, Name>>(),
            empty_list<std::pair<Name, std::initializer_list<std::pair<Name, edge_map_t<Symmetry>>>>>(),
            reversed_set_origin,
            pmr::map<Name, pmr::vector<Name>>{{InternalName<Name>::QR_1, std::move(free_name_1)}, {InternalName<Name>::QR_2, std::move(free_name_2)}},
            std::vector<Name>{InternalName<Name>::QR_1, InternalName<Name>::QR_2},
            false,
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<std::pair<Name, std::initializer_list<std::pair<Symmetry, Size>>>>());
      // call lapack
      auto common_edge_1 = Edge<Symmetry>();
      auto common_edge_2 = Edge<Symmetry>();
      for (const auto& [sym, _] : tensor_merged.core->blocks) {
         auto m = map_at(tensor_merged.core->edges[0].map, sym[0]);
         auto n = map_at(tensor_merged.core->edges[1].map, sym[1]);
         auto k = m > n ? n : m;
         common_edge_1.map.emplace_back(sym[1], k);
         common_edge_2.map.emplace_back(sym[0], k);
      }
      do_sort(common_edge_1.map);
      do_sort(common_edge_2.map);
      auto tensor_1 = Tensor<ScalarType, Symmetry, Name>{
            {InternalName<Name>::QR_1, use_qr_not_lq ? common_name_q : common_name_r},
            {std::move(tensor_merged.core->edges[0]), std::move(common_edge_1)}};
      auto tensor_2 = Tensor<ScalarType, Symmetry, Name>{
            {use_qr_not_lq ? common_name_r : common_name_q, InternalName<Name>::QR_2},
            {std::move(common_edge_2), std::move(tensor_merged.core->edges[1])}};
      for (auto& [symmetries, block] : tensor_merged.core->blocks) {
         auto* data_1 = map_at(tensor_1.core->blocks, symmetries).data();
         auto* data_2 = map_at(tensor_2.core->blocks, symmetries).data();
         auto* data = block.data();
         const int m = map_at(tensor_1.core->edges[0].map, symmetries[0]);
         const int n = map_at(tensor_2.core->edges[1].map, symmetries[1]);
         const int k = m > n ? n : m;
         const int max = m > n ? m : n;
         if (m * n != 0) {
            calculate_qr<ScalarType>(m, n, k, max, data, data_1, data_2, use_qr_not_lq);
         }
      }
      // 参考svd中的情况
      // 应 1 nr, 然后再考虑是否在q和r中是否分别左有无符号的反转
      // tensor_1 == tensor_q -> q nr             -> nothing  // use_qr_not_lq
      // tensor_2 == tensor_q -> r nr (r nr q yr) -> q yr -> 2 yr
      if constexpr (is_fermi) {
         (use_qr_not_lq ? reversed_set_1 : reversed_set_2).insert(common_name_q);
      }
      auto new_tensor_1 = tensor_1.edge_operator_implement(
            empty_list<std::pair<Name, Name>>(),
            pmr::map<Name, pmr::vector<std::tuple<Name, edge_map_t<Symmetry, true>>>>{{InternalName<Name>::QR_1, std::move(free_names_and_edges_1)}},
            reversed_set_1,
            empty_list<std::pair<Name, std::initializer_list<Name>>>(),
            std::move(result_name_1),
            false,
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<std::pair<Name, std::initializer_list<std::pair<Symmetry, Size>>>>());
      auto new_tensor_2 = tensor_2.edge_operator_implement(
            empty_list<std::pair<Name, Name>>(),
            pmr::map<Name, pmr::vector<std::tuple<Name, edge_map_t<Symmetry, true>>>>{{InternalName<Name>::QR_2, std::move(free_names_and_edges_2)}},
            reversed_set_2,
            empty_list<std::pair<Name, std::initializer_list<Name>>>(),
            std::move(result_name_2),
            false,
            empty_list<Name>(),
            use_qr_not_lq ? pmr::set<Name>{} : pmr::set<Name>{common_name_q},
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<std::pair<Name, std::initializer_list<std::pair<Symmetry, Size>>>>());
      return {std::move(use_qr_not_lq ? new_tensor_1 : new_tensor_2), std::move(use_qr_not_lq ? new_tensor_2 : new_tensor_1)};
   }
} // namespace TAT
#endif
