/**
 * \file name.hpp
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
#ifndef TAT_NAME_HPP
#define TAT_NAME_HPP

#include <concepts>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace TAT {
   /**
    * \defgroup Name
    * 张量的边的名称
    *
    * 张量的边的名称可以是用户自定义的类型, 但是需要满足一些条件
    * 1. 拥有比较函数, 从而使他可以作为`std::map`的`Key`
    * 2. 设置了InternalName, 这会在张量的一些函数中使用, 如svd, contract
    * 3. 在NameTraits中设置了输入输出函数, 会在io中使用
    *
    * \see NameTraits, InternalName
    * @{
    */
   /**
    * FastName使用的映射表类型
    * \see fastname_dataset
    */
   struct fastname_dataset_t {
      /**
       * Name中用于标号的类型
       */
      using fast_name_id_t = std::uint32_t;

      /**
       * Name的全局计数, 每当新建一个Name都会是指递增并获取一个关于Name的字符串唯一的标号
       */
      fast_name_id_t fastname_number = 1;

      /**
       * Name的字符串到标号的映射表
       *
       * \note 这个参数放在Name类外面, 是为了在gdb中显示得比较好看
       */
      std::map<std::string, fast_name_id_t> name_to_id = {{"", 0}};
      /**
       * 标号到Name的字符串的映射表
       */
      std::vector<std::string> id_to_name = {""};
   };
   /**
    * FastName使用的映射表包含数到字符串和字符串到数两个映射
    *
    * \see FastName
    */
   inline auto fastname_dataset = fastname_dataset_t();

   /**
    * 用于给张量的边命名的类型FastName, 新建FastName的时候可以选定标号, 也可以选定字符串作为名称, FastName将自动保证标号和名称的一一对应
    * \note 一个FastName拥有一个标号, 而每个标号对应一个双向唯一的字符串作为名字, 有fastname_dataset.fastname_number维护目前已分配的标号量,
    * 新建一个字符串的FastName时将递增fastname_dataset.fastname_number并获取一个唯一的标号
    * \see fastname_dataset
    */
   struct FastName {
      /**
       * FastName的标号
       */
      fastname_dataset_t::fast_name_id_t id = 0; // 默认为空串, 行为和std::string一致

      FastName() = default;

      FastName(const fastname_dataset_t::fast_name_id_t id) : id(id) {}

      template<typename String>
         requires(std::is_convertible_v<String, std::string> && !std::is_same_v<std::remove_cvref_t<String>, FastName>)
      FastName(String&& name) {
         // use template to avoid type convension here
         if (const auto found = fastname_dataset.name_to_id.find(name); found == fastname_dataset.name_to_id.end()) [[unlikely]] {
            fastname_dataset.id_to_name.emplace_back(name);
            id = fastname_dataset.name_to_id[name] = fastname_dataset.fastname_number++;
         } else [[likely]] {
            id = found->second;
         }
      }

      operator const std::string&() const {
         return fastname_dataset.id_to_name[id];
      }

      auto operator<=>(const FastName&) const = default;
   };

   /**
    * `Tensor`默认使用的`Name`, 如果没有定义宏`TAT_USE_SIMPLE_NAME`则为`FastName`, 否则为`std::string`
    *
    * \see Tensor
    */
   using DefaultName =
#ifdef TAT_USE_SIMPLE_NAME
         std::string
#else
         FastName
#endif
         ;

   /**
    * 对于每个将要被使用做Name的类型, 需要设置一部分保留类型
    *
    * 推荐将除了`Default_x`外的所有名称都进行定义, 但也可以只定义三个`Default_x`, 如果都定义了讲忽略`Default_x`
    * \tparam Name 将要被当作张量边名称的类型
    */
   template<typename Name>
   struct InternalName {
#define TAT_DEFINE_ALL_INTERNAL_NAME(x) static const Name x;
      TAT_DEFINE_ALL_INTERNAL_NAME(Default_0)
      TAT_DEFINE_ALL_INTERNAL_NAME(Default_1)
      TAT_DEFINE_ALL_INTERNAL_NAME(Default_2)
      TAT_DEFINE_ALL_INTERNAL_NAME(Contract_0)
      TAT_DEFINE_ALL_INTERNAL_NAME(Contract_1)
      TAT_DEFINE_ALL_INTERNAL_NAME(Contract_2)
      TAT_DEFINE_ALL_INTERNAL_NAME(SVD_U)
      TAT_DEFINE_ALL_INTERNAL_NAME(SVD_V)
      TAT_DEFINE_ALL_INTERNAL_NAME(QR_1)
      TAT_DEFINE_ALL_INTERNAL_NAME(QR_2)
      TAT_DEFINE_ALL_INTERNAL_NAME(Trace_1)
      TAT_DEFINE_ALL_INTERNAL_NAME(Trace_2)
      TAT_DEFINE_ALL_INTERNAL_NAME(Trace_3)
      TAT_DEFINE_ALL_INTERNAL_NAME(No_Old_Name)
      TAT_DEFINE_ALL_INTERNAL_NAME(No_New_Name)
      TAT_DEFINE_ALL_INTERNAL_NAME(Exp_1)
      TAT_DEFINE_ALL_INTERNAL_NAME(Exp_2)
#undef TAT_DEFINE_ALL_INTERNAL_NAME
   };
#ifndef TAT_DOXYGEN_SHOULD_SKIP_THIS
#define TAT_DEFINE_DEFAULT_INTERNAL_NAME(x, n) \
   template<typename Name> \
   const Name InternalName<Name>::x = InternalName<Name>::Default_##n;
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Contract_0, 0)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Contract_1, 1)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Contract_2, 2)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(SVD_U, 1)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(SVD_V, 2)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(QR_1, 1)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(QR_2, 2)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Trace_1, 0)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Trace_2, 1)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Trace_3, 2)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(No_Old_Name, 0)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(No_New_Name, 0)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Exp_1, 1)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Exp_2, 2)
#undef TAT_DEFINE_DEFAULT_INTERNAL_NAME
#define TAT_DEFINE_INTERNAL_NAME(x) \
   template<> \
   inline const FastName InternalName<FastName>::x = "," #x; \
   template<> \
   inline const std::string InternalName<std::string>::x = "," #x;
   TAT_DEFINE_INTERNAL_NAME(Contract_0)
   TAT_DEFINE_INTERNAL_NAME(Contract_1)
   TAT_DEFINE_INTERNAL_NAME(Contract_2)
   TAT_DEFINE_INTERNAL_NAME(SVD_U)
   TAT_DEFINE_INTERNAL_NAME(SVD_V)
   TAT_DEFINE_INTERNAL_NAME(QR_1)
   TAT_DEFINE_INTERNAL_NAME(QR_2)
   TAT_DEFINE_INTERNAL_NAME(Trace_1)
   TAT_DEFINE_INTERNAL_NAME(Trace_2)
   TAT_DEFINE_INTERNAL_NAME(Trace_3)
   TAT_DEFINE_INTERNAL_NAME(No_Old_Name)
   TAT_DEFINE_INTERNAL_NAME(No_New_Name)
   TAT_DEFINE_INTERNAL_NAME(Exp_1)
   TAT_DEFINE_INTERNAL_NAME(Exp_2)
#undef TAT_DEFINE_INTERNAL_NAME
#endif

   template<typename Name>
   using name_out_operator_t = std::ostream& (*)(std::ostream&, const Name&);
   template<typename Name>
   using name_in_operator_t = std::istream& (*)(std::istream&, Name&);

   /**
    * 对于每个将要被使用做Name的类型, 需要设置其输入输出方式
    *
    * 需要特化本类型, 定义本类型的write, read, print, scan四个函数, 类型为name_out_operator<Name>和name_in_operator<Name>'
    *
    * \tparam Name 将要被当作张量边名称的类型
    *
    * \see name_out_operator, name_in_operator
    */
   template<typename Name>
   struct NameTraits {};

   /**
    * 判断一个类型是否可以作为Name
    *
    * \tparam T 如果定义了NameTraits四个函数中的至少一个, 则认为用户是希望将Name作为名称的, 将`value`设置为`true`, 否则为`false`
    * \see is_name_v
    */
   template<typename Name>
   concept is_name = (requires(std::ostream & o, const Name& n) { NameTraits<Name>::write(o, n); }) ||
                     (requires(std::istream & i, Name& n) { &NameTraits<Name>::read(i, n); }) ||
                     (requires(std::ostream & o, const Name& n) { NameTraits<Name>::print(o, n); }) ||
                     (requires(std::istream & i, Name& n) { &NameTraits<Name>::scan(i, n); });
   /**@}*/
} // namespace TAT
#endif
