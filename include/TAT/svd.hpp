/**
 * \file svd.hpp
 *
 * Copyright (C) 2019  Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_SVD_HPP
#define TAT_SVD_HPP

#include "tensor.hpp"

#define LAPACK_COMPLEX_CUSTOM
using lapack_complex_float = std::complex<float>;
using lapack_complex_double = std::complex<double>;
//#include "lapacke.h"

extern "C" {
void sgesvd_(
      const char* jobu,
      const char* jobvt,
      const int* m,
      const int* n,
      const float* a,
      const int* lda,
      float* s,
      float* u,
      const int* ldu,
      float* vt,
      const int* ldvt,
      float* work,
      const int* lwork,
      int* info);
void dgesvd_(
      const char* jobu,
      const char* jobvt,
      const int* m,
      const int* n,
      const double* a,
      const int* lda,
      double* s,
      double* u,
      const int* ldu,
      double* vt,
      const int* ldvt,
      double* work,
      const int* lwork,
      int* info);
void cgesvd_(
      const char* jobu,
      const char* jobvt,
      const int* m,
      const int* n,
      const std::complex<float>* a,
      const int* lda,
      float* s,
      std::complex<float>* u,
      const int* ldu,
      std::complex<float>* vt,
      const int* ldvt,
      std::complex<float>* work,
      const int* lwork,
      float* rwork,
      int* info);
void zgesvd_(
      const char* jobu,
      const char* jobvt,
      const int* m,
      const int* n,
      const std::complex<double>* a,
      const int* lda,
      double* s,
      std::complex<double>* u,
      const int* ldu,
      std::complex<double>* vt,
      const int* ldvt,
      std::complex<double>* work,
      const int* lwork,
      double* rwork,
      int* info);
}

namespace TAT {
   template<class ScalarType>
   void calculate_svd(const int& m, const int& n, const int& min, const ScalarType* a, ScalarType* u, real_base_t<ScalarType>* s, ScalarType* vt);

   template<>
   void calculate_svd<float>(const int& m, const int& n, const int& min, const float* a, float* u, float* s, float* vt) {
      int result;
      int max = m > n ? m : n;
      int lwork = 2 * (5 * min + max);
      auto work = vector<float>(lwork);
      sgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &lwork, &result);
      if (result != 0) {
         TAT_WARNING("Error in GESVD");
      }
   }
   template<>
   void calculate_svd<double>(const int& m, const int& n, const int& min, const double* a, double* u, double* s, double* vt) {
      int result;
      int max = m > n ? m : n;
      int lwork = 2 * (5 * min + max);
      auto work = vector<double>(lwork);
      dgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &lwork, &result);
      if (result != 0) {
         TAT_WARNING("Error in GESVD");
      }
   }
   template<>
   void calculate_svd<std::complex<float>>(
         const int& m,
         const int& n,
         const int& min,
         const std::complex<float>* a,
         std::complex<float>* u,
         float* s,
         std::complex<float>* vt) {
      int result;
      int max = m > n ? m : n;
      int lwork = 2 * (5 * min + max);
      auto work = vector<std::complex<float>>(lwork);
      auto rwork = vector<float>(5 * min);
      cgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &lwork, rwork.data(), &result);
      if (result != 0) {
         TAT_WARNING("Error in GESVD");
      }
   }
   template<>
   void calculate_svd<std::complex<double>>(
         const int& m,
         const int& n,
         const int& min,
         const std::complex<double>* a,
         std::complex<double>* u,
         double* s,
         std::complex<double>* vt) {
      int result;
      int max = m > n ? m : n;
      int lwork = 2 * (5 * min + max);
      auto work = vector<std::complex<double>>(lwork);
      auto rwork = vector<double>(5 * min);
      zgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &lwork, rwork.data(), &result);
      if (result != 0) {
         TAT_WARNING("Error in GESVD");
      }
   }

   template<class ScalarType, class Symmetry>
   typename Tensor<ScalarType, Symmetry>::svd_result
   Tensor<ScalarType, Symmetry>::svd(const std::set<Name>& free_name_set_u, Name common_name_u, Name common_name_v, Size cut) const {
      constexpr bool is_fermi = is_fermi_symmetry_v<Symmetry>;
      // merge
      auto free_name_u = vector<Name>();
      auto free_name_v = vector<Name>();
      auto reversed_set_u = std::set<Name>();
      auto reversed_set_v = std::set<Name>();
      auto reversed_set_origin = std::set<Name>();
      auto result_name_u = vector<Name>();
      auto result_name_v = vector<Name>();
      auto free_names_and_edges_u = vector<std::tuple<Name, BoseEdge<Symmetry>>>();
      auto free_names_and_edges_v = vector<std::tuple<Name, BoseEdge<Symmetry>>>();
      result_name_v.push_back(common_name_v);
      for (auto i = 0; i < names.size(); i++) {
         const auto& n = names[i];
         if (free_name_set_u.find(n) != free_name_set_u.end()) {
            free_name_u.push_back(n);
            result_name_u.push_back(n);
            free_names_and_edges_u.push_back({n, {core->edges[i].map}});
            if constexpr (is_fermi) {
               if (core->edges[i].arrow) {
                  reversed_set_u.insert(n);
                  reversed_set_origin.insert(n);
               }
            }
         } else {
            free_name_v.push_back(n);
            result_name_v.push_back(n);
            free_names_and_edges_v.push_back({n, {core->edges[i].map}});
            if constexpr (is_fermi) {
               if (core->edges[i].arrow) {
                  reversed_set_v.insert(n);
                  reversed_set_origin.insert(n);
               }
            }
         }
      }
      result_name_u.push_back(common_name_u);
      const bool put_v_right = free_name_v.back() == names.back();
      auto tensor_merged = edge_operator(
            {},
            {},
            reversed_set_origin,
            {{SVD1, free_name_u}, {SVD2, free_name_v}},
            put_v_right ? vector<Name>{SVD1, SVD2} : vector<Name>{SVD2, SVD1});
      // gesvd
      auto common_edge_1 = Edge<Symmetry>();
      auto common_edge_2 = Edge<Symmetry>();
      for (const auto& [sym, _] : tensor_merged.core->blocks) {
         auto m = tensor_merged.core->edges[0].map.at(sym[0]);
         auto n = tensor_merged.core->edges[1].map.at(sym[1]);
         auto k = m > n ? n : m;
         common_edge_1.map[sym[1]] = k;
         common_edge_2.map[sym[0]] = k;
      }
      auto tensor_1 = Tensor<ScalarType, Symmetry>{put_v_right ? vector<Name>{SVD1, SVD2} : vector<Name>{SVD2, SVD1},
                                                   {std::move(tensor_merged.core->edges[0]), std::move(common_edge_1)}};
      auto tensor_2 = Tensor<ScalarType, Symmetry>{put_v_right ? vector<Name>{SVD1, SVD2} : vector<Name>{SVD2, SVD1},
                                                   {std::move(common_edge_2), std::move(tensor_merged.core->edges[1])}};
      auto result_s = std::map<Symmetry, vector<real_base_t<ScalarType>>>();
      for (const auto& [symmetries, block] : tensor_merged.core->blocks) {
         auto* data_u = tensor_1.core->blocks.at(symmetries).data();
         auto* data_v = tensor_2.core->blocks.at(symmetries).data();
         const auto* data = block.data();
         const int m = tensor_1.core->edges[0].map.at(symmetries[0]);
         const int n = tensor_2.core->edges[1].map.at(symmetries[1]);
         const int k = m > n ? n : m;
         auto s = vector<real_base_t<ScalarType>>(k);
         auto* s_data = s.data();
         calculate_svd<ScalarType>(m, n, k, data, data_u, s_data, data_v);
         result_s[symmetries[put_v_right]] = std::move(s);
      }
      const auto& tensor_u = put_v_right ? tensor_1 : tensor_2;
      const auto& tensor_v = put_v_right ? tensor_2 : tensor_1;
      reversed_set_u.insert(common_name_u);
      auto u = tensor_u.edge_operator({{SVD2, common_name_u}}, {{SVD1, free_names_and_edges_u}}, reversed_set_u, {}, result_name_u);
      auto v = tensor_v.edge_operator({{SVD1, common_name_v}}, {{SVD2, free_names_and_edges_v}}, reversed_set_v, {}, result_name_v);
      return {std::move(u), std::move(result_s), std::move(v)};
   }
} // namespace TAT
#endif
