/**
 * \file TAT.hpp
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
#ifndef TAT_HPP
#define TAT_HPP

#ifndef __cplusplus
#error only work for c++
#endif

#if __cplusplus < 201703L
#error require c++17 or later
#endif

#if __cplusplus < 202002L
#define TAT_USE_CXX20 false
#else
#define TAT_USE_CXX20 true
#endif

// Macros options
// - TAT_USE_MPI: define to enable mpi support, cmake can configure it
// - TAT_USE_MKL_TRANSPOSE: define to use mkl for matrix transpose, cmake can configure it, TODO optimize
// - TAT_USE_MKL_GEMM_BATCH: define to use mkl ?gemm_batch, cmake can configure it
// - TAT_USE_SIMPLE_NAME: define to use raw std::string as default name
// - TAT_USE_VALID_DEFAULT_TENSOR: define to construct valid tensor when no argument given, default behavior constructs an invalid tensor
// - TAT_USE_TIMER: define to add timers for some common operator
// - TAT_ERROR_BITS: throw exception for different situations, rather than print warning
// - TAT_NOTHING_BITS: keep silent for different situations, rather than print warning
// - TAT_L3_CACHE, TAT_L2_CACHE, TAT_L1_CACHE: cache size, used in transpose

/**
 * TAT is A Tensor library
 */
namespace TAT {
#ifdef TAT_VERSION
#define TAT_VERSION "0.1.4"
#endif

   /**
    * TAT version
    */
   inline const char* version = TAT_VERSION;

   /**
    * TAT informations about compiler and license
    */
   inline const char* information = "TAT " TAT_VERSION " ("
#ifdef TAT_BUILD_TYPE
                                    "" TAT_BUILD_TYPE ", "
#endif
                                    "" __DATE__ ", " __TIME__
#ifdef TAT_COMPILER_INFORMATION
                                    ", " TAT_COMPILER_INFORMATION
#endif
                                    ")\n"
                                    "Copyright (C) 2019-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>\n"
                                    "This is free software; see the source for copying conditions.  There is NO\n"
                                    "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.";

   // evil
   namespace detail {
      /**
       * Singleton, print a tips when program exits in debug mode, and control color ansi in windows
       */
      struct evil_t {
         evil_t() noexcept;
         ~evil_t();
      };
      inline const detail::evil_t evil;
   } // namespace detail

   // log and warning
   namespace detail {
      /**
       * Print log for TAT
       */
      inline void log(const char* message);

      /**
       * Print warning
       *
       * Share the same interface with `nothing` and `error`.
       * Used for different situation, control by macro TAT_NOTHING_BITS and TAT_ERROR_BITS
       *
       * \param message warning message
       *
       * \see nothing, error
       */
      inline void warning(const char* message);

      /**
       * Do nothing
       *
       * \see warning, error
       */
      inline void nothing(const char*) {}

      /**
       * Throw runtime exception
       *
       * \param message exception message
       *
       * \see warning, nothing
       */
      inline void error(const char* message);

#ifndef TAT_ERROR_BITS
#define TAT_ERROR_BITS 0
#endif
#ifndef TAT_NOTHING_BITS
#define TAT_NOTHING_BITS 0
#endif

      constexpr auto what_if_lapack_error = TAT_ERROR_BITS & 1 ? error : TAT_NOTHING_BITS & 1 ? nothing : warning;
      constexpr auto what_if_name_missing = TAT_ERROR_BITS & 2 ? error : TAT_NOTHING_BITS & 2 ? nothing : warning;
      constexpr auto what_if_copy_shared = TAT_ERROR_BITS & 4 ? error : TAT_NOTHING_BITS & 4 ? nothing : warning;
   } // namespace detail
} // namespace TAT

#include <cstdint>

namespace TAT {
   // type alias

   // The most common used integral type `short`, `int`, `long` have different size in different platform
   // in linux, they are 16, 32, 64(lp64)
   // in windows, they are 16, 32, 32(llp64)
   // So use uintxx_t explicitly to avoid it incompatible when import data exported in another platform
   // In TAT, there is also `int` type common used, especially when calling blas or lapack function
   // It is all 32 bit in most platform currently.
   // But it is 64bit in ilp64 and 16bit in lp32
   // So please do not link blas lapack using ilp64 or lp32
   /**
    * Tensor rank type
    */
   using Rank = std::uint16_t;
   /**
    * Tensor block number, or dimension segment number type
    */
   using Nums = std::uint32_t;
   /**
    * Tensor content data size, or dimension size type``
    */
   using Size = std::uint64_t;

   /**
    * Fermi arrow type, `false` and `true` for out and in
    */
   using Arrow = bool;
} // namespace TAT

#include <complex>
#include <type_traits>

namespace TAT {
   // traits about scalar
   template<typename T>
   concept is_real = std::is_scalar_v<T>;

   template<typename T>
   concept is_complex = is_real<typename T::value_type> && std::is_same_v<T, std::complex<typename T::value_type>>;

   template<typename T>
   concept is_scalar = is_real<T> || is_complex<T>;

   namespace detail {
      template<typename T>
      struct real_scalar_helper : std::conditional<is_real<T>, T, void> {};

      template<typename T>
      struct real_scalar_helper<std::complex<T>> : std::conditional<is_real<T>, T, void> {};
   } // namespace detail
   /**
    * Get corresponding real type, used in svd and norm
    *
    * \tparam T if T is complex type, return corresponding basic real type, if it is real type, return itself, otherwise, return void
    */
   template<typename T>
   using real_scalar = typename detail::real_scalar_helper<T>::type;
} // namespace TAT

#include "structure/tensor.hpp"

#include "miscellaneous/io.hpp"
#include "miscellaneous/mpi.hpp"
#include "miscellaneous/scalar.hpp"

#include "implement/contract.hpp"
#include "implement/edge_miscellaneous.hpp"
#include "implement/edge_operator.hpp"
#include "implement/exponential.hpp"
#include "implement/get_item.hpp"
#include "implement/identity_and_conjugate.hpp"
#include "implement/multiple.hpp"
#include "implement/qr.hpp"
#include "implement/shrink_and_expand.hpp"
#include "implement/svd.hpp"
#include "implement/trace.hpp"
#include "implement/transpose.hpp"

#endif
