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

#ifdef _MSVC_LANG
#if _MSVC_LANG < 201703L
#error require c++17 or later
#endif
#else
#if __cplusplus < 201703L
#error require c++17 or later
#endif
#endif

// 开关说明
// TAT_USE_MPI 定义以开启MPI支持, cmake可对此进行定义
// TAT_USE_MKL_TRANSPOSE 定义以使用mkl加速转置, cmake可对此进行定义 TODO 进一步优化
// TAT_USE_MKL_GEMM_BATCH 定义以使用mkl的?gemm_batch, cmake可对此进行定义
// TAT_USE_SINGULAR_MATRIX svd出来的奇异值使用矩阵表示
// TAT_USE_SIMPLE_NAME 定义以使用原始字符串作为name
// TAT_USE_VALID_DEFAULT_TENSOR 默认tensor初始化会产生一个合法的tensor, 默认不合法
// TAT_USE_TIMER 对常见操作进行计时
// TAT_USE_RESTRICT_SMALL_ALLOCATOR 使用严格的小分配器
// TAT_SMALL_ALLOCATOR_SIZE 小分配器的大小
// TAT_ERROR_BITS 将各类警告转换为异常
// TAT_NOTHING_BITS 将各类警告转换为静默
// TAT_L3_CACHE, TAT_L2_CACHE, TAT_L1_CACHE 在转置中会使用
// TAT_USE_L3_CACHE 转置中默认不使用l3_cache, 设置以使用之

/**
 * TAT is A Tensor library
 */
namespace TAT {
   /**
    * \defgroup Miscellaneous
    * @{
    */

   /**
    * TAT的版本号
    */
   inline const char* version =
#ifdef TAT_VERSION
         TAT_VERSION
#else
         "0.2.0"
#endif
         ;

   /**
    * 编译与license的相关信息
    */
   inline const char* information = "TAT"
#ifdef TAT_VERSION
                                    " " TAT_VERSION
#endif
                                    " ("
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

   /// \private
   struct evil_t {
      evil_t() noexcept;
      ~evil_t();
   };
   /**
    * Debug模式中, 将在程序末尾打印一行友情提示, 过早的优化是万恶之源, 同时此对象也控制windows下终端的色彩模式
    */
   inline const evil_t evil;

   /**
    * TAT使用的日志打印
    */
   inline void TAT_log(const char* message);

   /**
    * 什么事情也不做
    *
    * 接口和`TAT_warning`, `TAT_error`一致, 可供各种细分的警告或错误使用, 通过设置他们为这三个中的一个的指针来选择对错误的容忍程度
    *
    * \see TAT_warning, TAT_error
    */
   inline void TAT_nothing(const char*) {}

   /**
    * TAT使用的打印警告
    *
    * \param message 待打印的内容
    *
    * \see TAT_nothing, TAT_error
    */
   inline void TAT_warning(const char* message);

   /**
    * TAT使用的抛出运行时异常
    *
    * \param message 异常说明
    *
    * \see TAT_nothing, TAT_warning
    */
   inline void TAT_error(const char* message);

   // 下面这一段使用两个BITS来选择一些情况下是否警告，是否静默或者是否直接报错退出

#ifndef TAT_DOXYGEN_SHOULD_SKIP_THIS

#ifndef TAT_ERROR_BITS
#define TAT_ERROR_BITS 0
#endif
#ifndef TAT_NOTHING_BITS
#define TAT_NOTHING_BITS 0
#endif

   constexpr auto TAT_warning_or_error_when_lapack_error = TAT_ERROR_BITS & 1 ? TAT_error : TAT_NOTHING_BITS & 1 ? TAT_nothing : TAT_warning;
   constexpr auto TAT_warning_or_error_when_name_missing = TAT_ERROR_BITS & 2 ? TAT_error : TAT_NOTHING_BITS & 2 ? TAT_nothing : TAT_warning;
   constexpr auto TAT_warning_or_error_when_copy_shared = TAT_ERROR_BITS & 4 ? TAT_error : TAT_NOTHING_BITS & 4 ? TAT_nothing : TAT_warning;
#endif
   /**@}*/
} // namespace TAT

#include <cstdint>

namespace TAT {
   // 下面三个类型原本是short, int, long
   // 在linux(lp64)下分别是16, 32, 64
   // 但是windows(llp64)中是16, 32, 32
   // 在io中会出现windows和linux的输入输出互相不可读的问题
   // 所以显式写成uintxx_t的格式
   // TAT中还有一些地方会出现int, 一般为调用blas和lapack的地方
   // 为32位, 在64位系统中(ilp64)和32位系统中(lp32)分别是64为和16位
   // 所以底层的blas和lapack库不可以是ilp64或者lp32版本
   /**
    * 张量的秩的类型
    */
   using Rank = std::uint16_t;
   /**
    * 张量分块数目和一个边上对称性数目的类型
    */
   using Nums = std::uint32_t;
   /**
    * 张量数据维度大小和数据本身大小的类型
    */
   using Size = std::uint64_t;

   /**
    * 费米箭头方向的类型, `false`和`true`分别表示出入
    */
   using Arrow = bool;
} // namespace TAT

#include <complex>
#include <type_traits>

namespace TAT {
   template<typename T>
   concept is_real = std::is_scalar_v<T>;

   template<typename T>
   concept is_complex = std::is_same_v<T, std::complex<typename T::value_type>>;

   template<typename T>
   concept is_scalar = is_real<T> || is_complex<T>;

   template<typename T>
   struct real_scalar_helper : std::conditional<std::is_scalar_v<T>, T, void> {};
   template<typename T>
   struct real_scalar_helper<std::complex<T>> : std::conditional<std::is_scalar_v<T>, T, void> {};

   /**
    * 取对应的实数类型, 在svd, norm等地方会用到
    * \tparam T 如果`T`是`std::complex<S>`, 则为`S`, 若`T`为其他标量类型, 则为`T`本身, 否则为`void`
    */
   template<typename T>
   using real_scalar = typename real_scalar_helper<T>::type;
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
#include "implement/identity.hpp"
#include "implement/multiple.hpp"
#include "implement/qr.hpp"
#include "implement/shrink_and_expand.hpp"
#include "implement/svd.hpp"
#include "implement/trace.hpp"
#include "implement/transpose.hpp"

namespace TAT {
   /**
    * Z2对称性的类型
    */
   using Z2 = bool;
   /**
    * U1对称性的类型
    */
   using U1 = std::int32_t;

   using NoSymmetry = Symmetry<>;
   using Z2Symmetry = Symmetry<Z2>;
   using U1Symmetry = Symmetry<U1>;
   using FermiSymmetry = Symmetry<fermi_wrap<U1>>;
   using FermiZ2Symmetry = Symmetry<fermi_wrap<U1>, Z2>;
   using FermiU1Symmetry = Symmetry<fermi_wrap<U1>, U1>;
} // namespace TAT

#endif
