/**
 * \file tensor.hpp
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
#ifndef TAT_TENSOR_HPP
#define TAT_TENSOR_HPP

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <tuple>

#include "../utility/allocator.hpp"
#include "../utility/propagate_const.hpp"
#include "core.hpp"
#include "edge.hpp"
#include "name.hpp"
#include "symmetry.hpp"

namespace TAT {
   auto construct_name_to_index(const std::ranges::range auto& names) {
      using Name = std::remove_cvref_t<std::ranges::range_value_t<decltype(names)>>;
      std::vector<std::pair<std::remove_cvref_t<Name>, Rank>> result;
      result.reserve(names.size());
      for (Rank name_index = 0; name_index < names.size(); name_index++) {
         result.emplace_back(names[name_index], name_index);
      }
      do_sort(result);
      return result;
   }

   bool check_valid_name(const std::ranges::range auto& names, const Rank& rank) {
      if (names.size() != rank) [[unlikely]] {
         detail::error("Wrong name list length which no equals to expected length");
         return false;
      }
      for (auto i = names.begin(); i != names.end(); ++i) {
         for (auto j = std::next(i); j != names.end(); ++j) {
            if (*i == *j) [[unlikely]] {
               detail::error("Duplicated names in name list");
               return false;
            }
         }
      }
      return true;
   }

   // (Name, EdgeSegment)
   template<typename T, typename Name, typename Symmetry>
   concept name_edge_pair = requires(const T pair) {
      Name(std::get<0>(pair));
      edge_segment_t<Symmetry>(std::get<1>(pair));
   };

   // [(Name, EdgeSegment)]
   template<typename T, typename Name, typename Symmetry>
   concept name_edge_pair_list = std::ranges::range<T> && name_edge_pair<std::ranges::range_value_t<T>, Name, Symmetry>;

   // Name -> [(Name, EdgeSegment)]
   template<typename T, typename Name, typename Symmetry>
   concept split_configuration = requires(T split_map, Name name) {
      requires std::ranges::range<T>;
      { map_at(split_map, name) } -> name_edge_pair_list<Name, Symmetry>;
   };

   // Name -> [Name]
   template<typename T, typename Name, typename Symmetry>
   concept merge_configuration = requires(T merge_map, Name name) {
      requires std::ranges::range<T>;
      { map_at(merge_map, name) } -> range_of<Name>;
   };

   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   struct TensorShape;

   /**
    * Tensor type
    *
    * tensor type contains edge name, edge shape, and tensor content.
    * every edge has a Name as its name, for nom-symmetric tensor, an edge is
    * just a number describing its dimension.
    * for symmetric tensor, an edge is a segment like structure, describing
    * each symmetry's dimension.
    * tensor content is represented as several blocks, for non-symmetric tensor,
    * there is only one block
    *
    * \tparam ScalarType scalar type of tensor content
    * \tparam Symmetry tensor's symmetry
    * \tparam Name name type to distinguish different edge
    */
   template<is_scalar ScalarType = double, is_symmetry Symmetry = Symmetry<>, is_name Name = DefaultName>
   struct Tensor {
      // common used type alias
    public:
      using scalar_t = ScalarType;
      using symmetry_t = Symmetry;
      using name_t = Name;
      using edge_t = Edge<Symmetry>;
      using core_t = Core<ScalarType, Symmetry>;

      // tensor data
    public:
      /**
       * name of tensor's edge
       * \see Name
       */
      std::vector<Name> names;
      /**
       * a map from name to its edge's index
       * \note it is useful when the tensor rank is very high
       */
      std::vector<std::pair<Name, Rank>> name_to_index;
      /**
       * tensor data except name, including edge and block
       * \see Core
       * \note bacause edge rename is very common operation, to avoid copy data, put the remaining data into shared pointer
       */
      detail::propagate_const_shared_ptr<core_t> core;

    public:
      TensorShape<ScalarType, Symmetry, Name> shape() {
         return {this};
      }

      // constructors
    public:
      // There are many method to construct edge, so it is not proper to use initializer list
      /**
       * Initialize tensor with tensor edge name and tensor edge shape, blocks will be generated by edges
       *
       * \param names_init edge name
       * \param edges_init edge shape
       * \see Core
       */
      template<range_of<Name> VectorName = std::vector<Name>, range_of<Edge<Symmetry>> VectorEdge = std::vector<Edge<Symmetry>>>
      Tensor(VectorName&& names_init, VectorEdge&& edges_init) :
            names(forward_vector<std::vector<Name>>(std::forward<VectorName>(names_init))),
            name_to_index(construct_name_to_index(names)),
            core(std::make_shared<core_t>(std::forward<VectorEdge>(edges_init))) {
         check_valid_name(names, core->edges.size());
      }

      /**
       * Tensor deep copy, default copy will share the common data, i.e. the same core
       * \see core
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> copy() const {
         auto result = Tensor<ScalarType, Symmetry, Name>(names, core->edges);
         std::copy(storage().begin(), storage().end(), result.storage().begin());
         return result;
      }

      Tensor() : Tensor(1){};
      Tensor(const Tensor& other) = default;
      Tensor(Tensor&& other) noexcept = default;
      Tensor& operator=(const Tensor& other) = default;
      Tensor& operator=(Tensor&& other) noexcept = default;
      ~Tensor() = default;

      /**
       * create a rank-0 tensor
       * \param number the only element of this tensor
       */
      explicit Tensor(ScalarType number) : Tensor({}, {}) {
         storage().front() = number;
      }

    private:
      [[nodiscard]] static auto
      get_edge_from_edge_symmetry_and_arrow(const range_of<Symmetry> auto& edge_symmetry, const range_of<Arrow> auto& edge_arrow, Rank rank) {
         // used in one
         if constexpr (Symmetry::length == 0) {
            return std::vector<Edge<Symmetry>>(rank, {1});
         } else {
            auto result = std::vector<Edge<Symmetry>>();
            result.reserve(rank);
            for (auto [symmetry, arrow] = std::tuple{edge_symmetry.begin(), edge_arrow.begin()}; symmetry < edge_symmetry.end();
                 ++symmetry, ++arrow) {
               if constexpr (Symmetry::is_fermi_symmetry) {
                  result.push_back({{{*symmetry, 1}}, *arrow});
               } else {
                  result.push_back({{{*symmetry, 1}}});
               }
            }
            return result;
         }
      }

    public:
      /**
       * Create a high rank tensor but which only contains one element
       *
       * \note Tensor::one(a, {}, {}, {}) is equivilent to Tensor(a)
       * \param number the only element
       * \param names_init edge name
       * \param edge_symmetry the symmetry for every edge, if valid
       * \param edge_arrow the fermi arrow for every edge, if valid
       */
      template<
            range_of<Name> NameList = std::initializer_list<Name>,
            range_of<Symmetry> SymmetryList = std::initializer_list<Symmetry>,
            range_of<Arrow> ArrowList = std::initializer_list<Arrow>>
      [[nodiscard]] static Tensor<ScalarType, Symmetry, Name>
      one(ScalarType number, const NameList& names_init, const SymmetryList& edge_symmetry = {}, const ArrowList& edge_arrow = {}) {
         const auto rank = names_init.size();
         auto result = Tensor(names_init, get_edge_from_edge_symmetry_and_arrow(edge_symmetry, edge_arrow, rank));
         result.storage().front() = number;
         return result;
      }

      [[nodiscard]] bool scalar_like() const {
         return storage().size() == 1;
      }

      /**
       * Get the only element from a tensor which contains only one element
       */
      explicit operator ScalarType() const {
         if (!scalar_like()) [[unlikely]] {
            detail::error("Try to get the only element of the tensor which contains more than one element");
         }
         return storage().front();
      }

      // elementwise operators
    public:
      /**
       * 产生一个与自己形状一样的张量
       * \return 一个未初始化数据内容的张量
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> same_shape() const {
         return Tensor<ScalarType, Symmetry, Name>(names, core->edges);
      }
      /**
       * 对张量的每个数据元素做同样的非原地的变换
       * \param function 变换的函数
       * \return 张量自身
       * \note 参见std::transform
       * \see transform
       */
      template<typename Function>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> map(Function&& function) const {
         auto result = same_shape();
         std::transform(storage().begin(), storage().end(), result.storage().begin(), function);
         return result;
      }

      /**
       * 对张量的每个数据元素做同样的原地的变换
       * \param function 变换的函数
       * \return 张量自身
       * \note 参见std::transform
       * \see map
       */
      template<typename Function>
      Tensor<ScalarType, Symmetry, Name>& transform(Function&& function) & {
         if (core.use_count() != 1) [[unlikely]] {
            core = std::make_shared<Core<ScalarType, Symmetry>>(*core);
            detail::what_if_copy_shared("Set tensor shared, copy happened here");
         }
         std::transform(storage().begin(), storage().end(), storage().begin(), function);
         return *this;
      }
      template<typename Function>
      Tensor<ScalarType, Symmetry, Name>&& transform(Function&& function) && {
         return std::move(transform(function));
      }

      /**
       * 通过一个生成器设置一个张量内的数据
       * \param generator 生成器, 一般来说是一个无参数的函数, 返回值为标量, 多次调用填充张量
       * \return 张量自身
       * \see transform
       */
      template<typename Generator>
      Tensor<ScalarType, Symmetry, Name>& set(Generator&& generator) & {
         if (core.use_count() != 1) [[unlikely]] {
            core = std::make_shared<Core<ScalarType, Symmetry>>(*core);
            detail::what_if_copy_shared("Set tensor shared, copy happened here");
         }
         std::generate(storage().begin(), storage().end(), generator);
         return *this;
      }
      template<typename Generator>
      Tensor<ScalarType, Symmetry, Name>&& set(Generator&& generator) && {
         return std::move(set(generator));
      }

      /**
       * 将张量内的数据全部设置为零
       * \return 张量自身
       * \see set
       */
      Tensor<ScalarType, Symmetry, Name>& zero() & {
         return transform([](ScalarType) {
            return 0;
         });
      }
      Tensor<ScalarType, Symmetry, Name>&& zero() && {
         return std::move(zero());
      }

      /**
       * 将张量内的数据设置为便于测试的值
       * \return 张量自身
       * \see set
       */
      Tensor<ScalarType, Symmetry, Name>& range(ScalarType first = 0, ScalarType step = 1) & {
         return set([&first, step]() {
            auto result = first;
            first += step;
            return result;
         });
      }
      Tensor<ScalarType, Symmetry, Name>&& range(ScalarType first = 0, ScalarType step = 1) && {
         return std::move(range(first, step));
      }

      // get element
    public:
      template<pair_range_of<Name, std::pair<Symmetry, Size>> MapNamePoint = std::initializer_list<std::pair<Name, std::pair<Symmetry, Size>>>>
      [[nodiscard]] const ScalarType& at(MapNamePoint&& position) const& {
         return const_at(std::forward<MapNamePoint>(position));
      }

      template<pair_range_of<Name, Size> MapNameIndex = std::initializer_list<std::pair<Name, Size>>>
      [[nodiscard]] const ScalarType& at(MapNameIndex&& position) const& {
         return const_at(std::forward<MapNameIndex>(position));
      }

      [[nodiscard]] const ScalarType& at() const& {
         return at(empty_list<std::pair<Name, std::pair<Symmetry, Size>>>());
      }

      template<pair_range_of<Name, std::pair<Symmetry, Size>> MapNamePoint = std::initializer_list<std::pair<Name, std::pair<Symmetry, Size>>>>
      [[nodiscard]] ScalarType& at(MapNamePoint&& position) & {
         if (core.use_count() != 1) [[unlikely]] {
            core = std::make_shared<Core<ScalarType, Symmetry>>(*core);
            detail::what_if_copy_shared("Get reference which may change of shared tensor, copy happened here, use const_at to get const reference");
         }
         return const_cast<ScalarType&>(const_at(std::forward<MapNamePoint>(position)));
      }

      template<pair_range_of<Name, Size> MapNameIndex = std::initializer_list<std::pair<Name, Size>>>
      [[nodiscard]] ScalarType& at(MapNameIndex&& position) & {
         if (core.use_count() != 1) [[unlikely]] {
            core = std::make_shared<Core<ScalarType, Symmetry>>(*core);
            detail::what_if_copy_shared("Get reference which may change of shared tensor, copy happened here, use const_at to get const reference");
         }
         return const_cast<ScalarType&>(const_at(std::forward<MapNameIndex>(position)));
      }

      [[nodiscard]] ScalarType& at() & {
         return at(empty_list<std::pair<Name, std::pair<Symmetry, Size>>>());
      }

      template<pair_range_of<Name, std::pair<Symmetry, Size>> MapNamePoint = std::initializer_list<std::pair<Name, std::pair<Symmetry, Size>>>>
      [[nodiscard]] const ScalarType& const_at(MapNamePoint&& position) const& {
         auto pmr_guard = scope_resource(default_buffer_size);
         return get_item(may_need_sort<std::vector<std::pair<Name, std::pair<Symmetry, Size>>>>(position));
      }

      template<pair_range_of<Name, Size> MapNameIndex = std::initializer_list<std::pair<Name, Size>>>
      [[nodiscard]] const ScalarType& const_at(MapNameIndex&& position) const& {
         auto pmr_guard = scope_resource(default_buffer_size);
         return get_item(may_need_sort<std::vector<std::pair<Name, Size>>>(position));
      }

      [[nodiscard]] const ScalarType& const_at() const& {
         return const_at(empty_list<std::pair<Name, std::pair<Symmetry, Size>>>());
      }

    private:
      [[nodiscard]] const ScalarType& get_item(const auto& position) const&;

    public:
      /**
       * 不同标量类型的张量之间的转换函数
       * \tparam OtherScalarType 目标张量的基础标量类型
       * \return 转换后的张量
       */
      template<typename OtherScalarType>
         requires is_scalar<OtherScalarType>
      [[nodiscard]] Tensor<OtherScalarType, Symmetry, Name> to() const {
         if constexpr (std::is_same_v<ScalarType, OtherScalarType>) {
            return *this;
         } else {
            auto result = Tensor<OtherScalarType, Symmetry, Name>{};
            result.names = names;
            result.name_to_index = name_to_index;
            result.core = std::make_shared<Core<OtherScalarType, Symmetry>>(core->edges);
            std::transform(storage().begin(), storage().end(), result.storage().begin(), [](ScalarType input) -> OtherScalarType {
               if constexpr (is_complex<ScalarType> && is_real<OtherScalarType>) {
                  return OtherScalarType(input.real());
               } else {
                  return OtherScalarType(input);
               }
            });
            return result;
         }
      }

      /**
       * 求张量的模, 是拉平看作向量的模, 并不是矩阵模之类的东西
       * \tparam p 所求的模是张量的p-模, 如果p=-1, 则意味着最大模即p=inf
       * \return 标量类型的模
       */
      template<int p = 2>
      [[nodiscard]] real_scalar<ScalarType> norm() const {
         real_scalar<ScalarType> result = 0;
         if constexpr (p == -1) {
            // max abs
            for (const auto& number : storage()) {
               if (auto absolute_value = std::abs(number); absolute_value > result) {
                  result = absolute_value;
               }
            }
         } else if constexpr (p == 0) {
            result += real_scalar<ScalarType>(storage().size());
         } else {
            for (const auto& number : storage()) {
               if constexpr (p == 1) {
                  result += std::abs(number);
               } else if constexpr (p == 2) {
                  result += std::norm(number);
               } else {
                  if constexpr (p % 2 == 0 && is_real<ScalarType>) {
                     result += std::pow(number, p);
                  } else {
                     result += std::pow(std::abs(number), p);
                  }
               }
            }
            result = std::pow(result, 1. / p);
         }
         return result;
      }

      // get core element

      template<pair_range_of<Name, Symmetry> MapNameSymmetry = std::initializer_list<std::pair<Name, Symmetry>>>
      const auto& const_block(MapNameSymmetry&& map) const& {
         auto pmr_guard = scope_resource(default_buffer_size);
         return get_block(may_need_sort<pmr::vector<std::pair<Name, Symmetry>>>(std::forward<MapNameSymmetry>(map)));
      }
      template<pair_range_of<Name, Symmetry> MapNameSymmetry = std::initializer_list<std::pair<Name, Symmetry>>>
      const auto& block(MapNameSymmetry&& map) const& {
         return const_block(std::forward<MapNameSymmetry>(map));
      }
      template<pair_range_of<Name, Symmetry> MapNameSymmetry = std::initializer_list<std::pair<Name, Symmetry>>>
      auto& block(MapNameSymmetry&& map) & {
         return const_cast<no_initialize::pmr::vector<ScalarType>&>(const_block(std::forward<MapNameSymmetry>(map)));
      }

    private:
      const auto& get_block(const auto& map) const&;

    public:
      const auto& storage() const {
         return core->storage;
      }
      auto& storage() {
         return core->storage;
      }

      const Edge<Symmetry>& edges(Rank r) const {
         return core->edges[r];
      }
      Edge<Symmetry>& edges(Rank r) {
         return core->edges[r];
      }
      const Edge<Symmetry>& edges(const Name& name) const {
         return edges(map_at(name_to_index, name));
      }
      Edge<Symmetry>& edges(const Name& name) {
         return edges(map_at(name_to_index, name));
      }

      /**
       * 对张量的边进行操作的中枢函数, 对边依次做重命名, 分裂, 费米箭头取反, 合并, 转置的操作,
       * \param rename_map 重命名边的名称的映射表
       * \param split_map 分裂一些边的数据, 需要包含分裂后边的形状, 不然分裂不唯一
       * \param reversed_name 将要取反费米箭头的边的名称列表
       * \param merge_map 合并一些边的名称列表
       * \param new_names 最后进行的转置操作后的边的名称顺序列表
       * \param apply_parity 控制费米对称性中费米性质产生的符号是否应用在结果张量上的默认行为
       * \param parity_exclude_name 是否产生符号这个问题上行为与默认行为相反的操作的边的名称, 四部分分别是split, reverse, reverse_before_merge, merge
       * \return 进行了一系列操作后的结果张量
       * \note 反转不满足和合并操作的条件时, 将在合并前再次反转需要反转的边, 方向对齐第一个有方向的边
       * \note 因为费米箭头在反转和合并分裂时会产生半个符号, 所以需要扔给一方张量, 另一方张量不变号
       * \note 但是转置部分时产生一个符号的, 所以这一部分无视apply_parity
       * \note 本函数对转置外不标准的腿的输入是脆弱的
       */
      template<
            split_configuration<Name, Symmetry> SplitMap =
                  std::initializer_list<std::pair<Name, std::initializer_list<std::pair<Name, edge_segment_t<Symmetry>>>>>,
            range_of<Name> ReversedName = std::initializer_list<Name>,
            merge_configuration<Name, Symmetry> MergeMap = std::initializer_list<std::pair<Name, std::initializer_list<Name>>>,
            range_of<Name> NewNames = std::initializer_list<Name>,
            range_of<Name> ParityExcludeNameSplit = std::initializer_list<Name>,
            range_of<Name> ParityExcludeNameBeforeTranspose = std::initializer_list<Name>,
            range_of<Name> ParityExcludeNameAfterTranspose = std::initializer_list<Name>,
            range_of<Name> ParityExcludeNameMerge = std::initializer_list<Name>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> edge_operator(
            SplitMap&& split_map,         // Name -> [(Name, Edge)]
            ReversedName&& reversed_name, // {Name}
            MergeMap&& merge_map,         // Name -> ([Name], Edge)
            NewNames&& new_names,         // [Name]
            const bool apply_parity = false,
            ParityExcludeNameSplit&& parity_exclude_name_split = {},
            ParityExcludeNameBeforeTranspose&& parity_exclude_name_reversed_before_transpose = {},
            ParityExcludeNameAfterTranspose&& parity_exclude_name_reversed_after_transpose = {},
            ParityExcludeNameMerge&& parity_exclude_name_merge = {}) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         return edge_operator_implement(
               may_need_sort<pmr::vector<std::pair<Name, typename std::ranges::range_value_t<SplitMap>::second_type>>>(
                     std::forward<SplitMap>(split_map)),
               may_need_sort<pmr::vector<Name>>(std::forward<ReversedName>(reversed_name)),
               may_need_sort<pmr::vector<std::pair<Name, typename std::ranges::range_value_t<MergeMap>::second_type>>>(
                     std::forward<MergeMap>(merge_map)),
               forward_vector<std::vector<Name>>(std::forward<NewNames>(new_names)), // need std::vector
               apply_parity,
               may_need_sort<pmr::vector<Name>>(std::forward<ParityExcludeNameSplit>(parity_exclude_name_split)),
               may_need_sort<pmr::vector<Name>>(std::forward<ParityExcludeNameBeforeTranspose>(parity_exclude_name_reversed_before_transpose)),
               may_need_sort<pmr::vector<Name>>(std::forward<ParityExcludeNameAfterTranspose>(parity_exclude_name_reversed_after_transpose)),
               may_need_sort<pmr::vector<Name>>(std::forward<ParityExcludeNameMerge>(parity_exclude_name_merge)),
               std::initializer_list<std::pair<Name, std::initializer_list<std::pair<Symmetry, Size>>>>()); // last argument only used in svd
         // last arg: Name -> Symmetry -> Size
      }

      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> edge_operator_implement(
            const auto& split_map,
            const auto& reversed_name,
            const auto& merge_map,
            std::vector<Name> new_names,
            const bool apply_parity,
            const auto& parity_exclude_name_split,
            const auto& parity_exclude_name_reversed_before_transpose,
            const auto& parity_exclude_name_reversed_after_transpose,
            const auto& parity_exclude_name_merge,
            const auto& edge_and_symmetries_to_cut_before_all = {}) const;

    public:
      /**
       * 对张量边的名称进行重命名
       * \param dictionary 重命名方案的映射表
       * \return 仅仅改变了边的名称的张量, 与原张量共享Core
       * \note 虽然功能蕴含于edge_operator中, 但是edge_rename操作很常用, 所以并没有调用会稍微慢的edge_operator, 而是实现一个小功能的edge_rename
       */
      template<typename MapNameName = std::initializer_list<std::pair<Name, Name>>>
         requires std::same_as<std::remove_cvref_t<typename std::ranges::range_value_t<MapNameName>::first_type>, Name> &&
               is_name<typename std::ranges::range_value_t<MapNameName>::second_type>
      [[nodiscard]] auto edge_rename(MapNameName&& dictionary) const {
         using ResultName = typename std::ranges::range_value_t<MapNameName>::second_type;
         return edge_rename_implement<ResultName>(may_need_sort<pmr::vector<std::pair<Name, ResultName>>>(std::forward<MapNameName>(dictionary)));
      }

    private:
      template<typename ResultName>
      auto edge_rename_implement(const auto& dictionary) const;

    public:
      /**
       * 对张量进行转置
       * \param target_names 转置后的目标边的名称顺序
       * \return 转置后的结果张量
       */
      template<range_of<Name> VectorName = std::initializer_list<Name>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> transpose(VectorName&& target_names) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         return edge_operator_implement(
               empty_list<std::pair<Name, std::initializer_list<std::pair<Name, edge_segment_t<Symmetry>>>>>(),
               empty_list<Name>(),
               empty_list<std::pair<Name, std::initializer_list<Name>>>(),
               forward_vector<std::vector<Name>>(std::forward<VectorName>(target_names)),
               false,
               empty_list<Name>(),
               empty_list<Name>(),
               empty_list<Name>(),
               empty_list<Name>(),
               empty_list<std::pair<Name, std::initializer_list<std::pair<Symmetry, Size>>>>());
      }

      /**
       * 将费米张量的一些边进行反转
       * \param reversed_name 反转的边的集合
       * \param apply_parity 是否应用反转产生的符号
       * \param parity_exclude_name 与apply_parity行为相反的边名集合
       * \return 反转后的结果张量
       */
      template<range_of<Name> ReversedName = std::initializer_list<Name>, range_of<Name> ExcludeName = std::initializer_list<Name>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      reverse_edge(ReversedName&& reversed_name, bool apply_parity = false, ExcludeName&& parity_exclude_name = {}) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         return edge_operator_implement(
               empty_list<std::pair<Name, std::initializer_list<std::pair<Name, edge_segment_t<Symmetry>>>>>(),
               may_need_sort<pmr::vector<Name>>(std::forward<ReversedName>(reversed_name)),
               empty_list<std::pair<Name, std::initializer_list<Name>>>(),
               names,
               apply_parity,
               empty_list<Name>(),
               may_need_sort<pmr::vector<Name>>(std::forward<ExcludeName>(parity_exclude_name)),
               empty_list<Name>(),
               empty_list<Name>(),
               empty_list<std::pair<Name, std::initializer_list<std::pair<Symmetry, Size>>>>());
      }

      /**
       * 合并张量的一些边
       * \param merge 合并的边的名称的映射表
       * \param apply_parity 是否应用合并边产生的符号
       * \param parity_exclude_name_merge merge过程中与apply_parity不符的例外
       * \param parity_exclude_name_reverse merge前不得不做的reverse过程中与apply_parity不符的例外
       * \return 合并边后的结果张量
       * \note 合并前转置的策略是将一组合并的边按照合并时的顺序移动到这组合并边中最后的一个边前, 其他边位置不变
       */
      template<
            merge_configuration<Name, Symmetry> MergeMap = std::initializer_list<std::pair<Name, std::vector<Name>>>,
            range_of<Name> ParityExcludeNameAfterTranspose = std::initializer_list<Name>,
            range_of<Name> ParityExcludeNameMerge = std::initializer_list<Name>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> merge_edge(
            MergeMap&& merge,
            bool apply_parity = false,
            ParityExcludeNameMerge&& parity_exclude_name_merge = {},
            ParityExcludeNameAfterTranspose&& parity_exclude_name_reverse = {}) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         return merge_edge_implement(
               may_need_sort<pmr::vector<std::pair<Name, typename std::ranges::range_value_t<MergeMap>::second_type>>>(std::forward<MergeMap>(merge)),
               apply_parity,
               may_need_sort<pmr::vector<Name>>(std::forward<ParityExcludeNameMerge>(parity_exclude_name_merge)),
               may_need_sort<pmr::vector<Name>>(std::forward<ParityExcludeNameAfterTranspose>(parity_exclude_name_reverse)));
      }

    private:
      Tensor<ScalarType, Symmetry, Name>
      merge_edge_implement(auto merge, const bool apply_parity, const auto& parity_exclude_name_merge, const auto& parity_exclude_name_reverse) const;

    public:
      /**
       * 分裂张量的一些边
       * \param split 分裂的边的名称的映射表
       * \param apply_parity 是否应用分裂边产生的符号
       * \param parity_exclude_name_split split过程中与apply_parity不符的例外
       * \return 分裂边后的结果张量
       */
      template<
            split_configuration<Name, Symmetry> SplitMap =
                  std::initializer_list<std::pair<Name, std::initializer_list<std::pair<Name, edge_segment_t<Symmetry>>>>>,
            range_of<Name> ParityExcludeNameSplit = std::initializer_list<Name>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      split_edge(SplitMap&& split, bool apply_parity = false, ParityExcludeNameSplit&& parity_exclude_name_split = {}) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         return split_edge_implement(
               may_need_sort<pmr::vector<std::pair<Name, typename std::ranges::range_value_t<SplitMap>::second_type>>>(std::forward<SplitMap>(split)),
               apply_parity,
               may_need_sort<pmr::vector<Name>>(std::forward<ParityExcludeNameSplit>(parity_exclude_name_split)));
      }

    private:
      Tensor<ScalarType, Symmetry, Name> split_edge_implement(auto split, const bool apply_parity, const auto& parity_exclude_name_split) const;

    public:
      // 可以考虑不转置成矩阵直接乘积的可能, 但这个最多优化N^2的常数次, 只需要转置不调用多次就不会产生太大的问题
      /**
       * 两个张量的缩并运算
       * \param tensor_1 参与缩并的第一个张量
       * \param tensor_2 参与缩并的第二个张量
       * \param contract_names 两个张量将要缩并掉的边的名称
       * \return 缩并后的张量
       */
      template<same_pair_range_of<Name> SetNameAndName = std::initializer_list<std::pair<Name, Name>>>
      [[nodiscard]] static Tensor<ScalarType, Symmetry, Name> contract(
            const Tensor<ScalarType, Symmetry, Name>& tensor_1,
            const Tensor<ScalarType, Symmetry, Name>& tensor_2,
            SetNameAndName&& contract_names);

      template<is_scalar ScalarType1, is_scalar ScalarType2, same_pair_range_of<Name> SetNameAndName = std::initializer_list<std::pair<Name, Name>>>
      [[nodiscard]] static auto contract(
            const Tensor<ScalarType1, Symmetry, Name>& tensor_1,
            const Tensor<ScalarType2, Symmetry, Name>& tensor_2,
            SetNameAndName&& contract_names) {
         using ResultScalarType = std::common_type_t<ScalarType1, ScalarType2>;
         using ResultTensor = Tensor<ResultScalarType, Symmetry, Name>;
         if constexpr (std::is_same_v<ResultScalarType, ScalarType1>) {
            if constexpr (std::is_same_v<ResultScalarType, ScalarType2>) {
               return ResultTensor::contract(tensor_1, tensor_2, std::forward<SetNameAndName>(contract_names));
            } else {
               return ResultTensor::contract(tensor_1, tensor_2.template to<ResultScalarType>(), std::forward<SetNameAndName>(contract_names));
            }
         } else {
            if constexpr (std::is_same_v<ResultScalarType, ScalarType2>) {
               return ResultTensor::contract(tensor_1.template to<ResultScalarType>(), tensor_2, std::forward<SetNameAndName>(contract_names));
            } else {
               return ResultTensor::contract(
                     tensor_1.template to<ResultScalarType>(),
                     tensor_2.template to<ResultScalarType>(),
                     std::forward<SetNameAndName>(contract_names));
            }
         }
      }

      template<is_scalar OtherScalarType, same_pair_range_of<Name> SetNameAndName = std::initializer_list<std::pair<Name, Name>>>
      [[nodiscard]] auto contract(const Tensor<OtherScalarType, Symmetry, Name>& tensor_2, SetNameAndName&& contract_names) const {
         return contract(*this, tensor_2, std::forward<SetNameAndName>(contract_names));
      }

      /**
       * 将一个张量与另一个张量的所有相同名称的边进行缩并
       * \param other 另一个张量
       * \return 缩并后的结果
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> contract_all_edge(const Tensor<ScalarType, Symmetry, Name>& other) const {
         // other不含有的边会在contract中自动删除
         auto contract_names = std::vector<std::pair<Name, Name>>();
         for (const auto& i : names) {
            contract_names.push_back({i, i});
         }
         return contract(other, std::move(contract_names));
      }

      /**
       * 张量与自己的共轭进行尽可能的缩并
       * \return 缩并后的结果
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> contract_all_edge() const {
         return contract_all_edge(conjugate());
      }

      /**
       * 生成相同形状的单位张量
       * \param pairs 看作矩阵时边的配对方案
       */
      template<same_pair_range_of<Name> SetNameAndName = std::initializer_list<std::tuple<Name, Name>>>
      Tensor<ScalarType, Symmetry, Name>& identity(SetNameAndName&& pairs) & {
         auto pmr_guard = scope_resource(default_buffer_size);
         return identity_implement(may_need_sort<pmr::vector<std::ranges::range_value_t<SetNameAndName>>>(std::forward<SetNameAndName>(pairs)));
      }

      template<same_pair_range_of<Name> SetNameAndName = std::initializer_list<std::tuple<Name, Name>>>
      Tensor<ScalarType, Symmetry, Name>&& identity(SetNameAndName&& pairs) && {
         return std::move(identity(std::forward<SetNameAndName>(pairs)));
      }

    private:
      Tensor<ScalarType, Symmetry, Name>& identity_implement(const auto& pairs) &;

    public:
      /**
       * 看作矩阵后求出矩阵指数
       * \param pairs 边的配对方案
       * \param step 迭代步数
       */
      template<same_pair_range_of<Name> SetNameAndName = std::initializer_list<std::pair<Name, Name>>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> exponential(SetNameAndName&& pairs, int step = 2) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         return exponential_implement(
               may_need_sort<pmr::vector<std::ranges::range_value_t<SetNameAndName>>>(std::forward<SetNameAndName>(pairs)),
               step);
      }

    private:
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> exponential_implement(const auto& pairs, int step) const;

    public:
      /**
       * 生成张量的共轭张量
       * \note 如果为对称性张量, 量子数取反, 如果为费米张量, 箭头取反, 如果为复张量, 元素取共轭
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> conjugate() const;

      template<same_pair_range_of<Name> SetNameAndName = std::initializer_list<std::pair<Name, Name>>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> trace(SetNameAndName&& trace_names) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         return trace_implement(may_need_sort<pmr::vector<std::ranges::range_value_t<SetNameAndName>>>(std::forward<SetNameAndName>(trace_names)));
      }

    private:
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> trace_implement(const auto& trace_names) const;

    public:
      using SingularType = Tensor<ScalarType, Symmetry, Name>;
      /**
       * 张量svd的结果类型
       * \note S的的对称性是有方向的, 用来标注如何对齐, 向U对齐
       */
      struct svd_result {
         Tensor<ScalarType, Symmetry, Name> U;
         SingularType S;
         Tensor<ScalarType, Symmetry, Name> V;
      };

      /**
       * 张量qr的结果类型
       */
      struct qr_result {
         Tensor<ScalarType, Symmetry, Name> Q;
         Tensor<ScalarType, Symmetry, Name> R;
      };

      /**
       * 张量缩并上SVD产生的奇异值数据, 就地操作
       * \param S 奇异值
       * \param name 张量与奇异值缩并的边名
       * \param direction 奇异值是含有一个方向的, SVD的结果中U还是V将与S相乘在这里被选定
       * \param division 如果为真, 则进行除法而不是乘法
       * \return 缩并的结果
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> multiple(const SingularType& S, const Name& name, char direction, bool division = false) const;

      /**
       * 对张量进行svd分解
       * \param free_name_set_u svd分解中u的边的名称集合
       * \param common_name_u 分解后u新产生的边的名称
       * \param common_name_v 分解后v新产生的边的名称
       * \param cut 需要截断的维度数目
       * \return svd的结果
       * \see svd_result
       * \note 对于对称性张量, S需要有对称性, S对称性与V的公共边配对, 与U的公共边相同
       */
      template<range_of<Name> SetName = std::initializer_list<Name>>
      [[nodiscard]] svd_result
      svd(SetName&& free_name_set_u,
          const Name& common_name_u,
          const Name& common_name_v,
          Size cut = Size(-1),
          const Name& singular_name_u = InternalName<Name>::SVD_U,
          const Name& singular_name_v = InternalName<Name>::SVD_V) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         return svd_implement(
               may_need_sort<pmr::vector<Name>>(std::forward<SetName>(free_name_set_u)),
               common_name_u,
               common_name_v,
               cut,
               singular_name_u,
               singular_name_v);
      }

    private:
      [[nodiscard]] svd_result svd_implement(
            const auto& free_name_set_u,
            const Name& common_name_u,
            const Name& common_name_v,
            Size cut,
            const Name& singular_name_u,
            const Name& singular_name_v) const;

    public:
      /**
       * 对张量进行qr分解
       * \param free_name_direction free_name_set取的方向, 为'Q'或'R'
       * \param free_name_set qr分解中某一侧的边的名称集合
       * \param common_name_q 分解后q新产生的边的名称
       * \param common_name_r 分解后r新产生的边的名称
       * \return qr的结果
       * \see qr_result
       */
      template<range_of<Name> SetName = std::initializer_list<Name>>
      [[nodiscard]] qr_result qr(char free_name_direction, SetName&& free_name_set, const Name& common_name_q, const Name& common_name_r) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         return qr_implement(
               free_name_direction,
               may_need_sort<pmr::vector<Name>>(std::forward<SetName>(free_name_set)),
               common_name_q,
               common_name_r);
      }

    private:
      [[nodiscard]] qr_result
      qr_implement(char free_name_direction, const auto& free_name_set, const Name& common_name_q, const Name& common_name_r) const;

    public:
      using EdgePointShrink = std::conditional_t<Symmetry::length == 0, Size, std::tuple<Symmetry, Size>>;
      using EdgePointExpand = std::conditional_t<
            Symmetry::length == 0,
            std::tuple<Size, Size>,
            std::conditional_t<Symmetry::is_fermi_symmetry, std::tuple<Arrow, Symmetry, Size, Size>, std::tuple<Symmetry, Size, Size>>>;
      // index, dim

      template<pair_range_of<Name, EdgePointExpand> ExpandConfigure = std::vector<std::pair<Name, EdgePointExpand>>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      expand(ExpandConfigure&& configure, const Name& old_name = InternalName<Name>::No_Old_Name) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         return expand_implement(may_need_sort<pmr::vector<std::pair<Name, EdgePointExpand>>>(std::forward<ExpandConfigure>(configure)), old_name);
      }

      template<pair_range_of<Name, EdgePointShrink> ShrinkConfigure = std::vector<std::pair<Name, EdgePointShrink>>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      shrink(ShrinkConfigure&& configure, const Name& new_name = InternalName<Name>::No_New_Name, Arrow arrow = false) const {
         auto pmr_guard = scope_resource(default_buffer_size);
         return shrink_implement(
               may_need_sort<pmr::vector<std::pair<Name, EdgePointShrink>>>(std::forward<ShrinkConfigure>(configure)),
               new_name,
               arrow);
      }

    private:
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      expand_implement(const auto& configure, const Name& old_name = InternalName<Name>::No_Old_Name) const;

      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      shrink_implement(const auto& configure, const Name& new_name = InternalName<Name>::No_New_Name, Arrow arrow = false) const;

    public:
      const Tensor<ScalarType, Symmetry, Name>& meta_put(std::ostream&) const;
      const Tensor<ScalarType, Symmetry, Name>& data_put(std::ostream&) const;
      Tensor<ScalarType, Symmetry, Name>& meta_get(std::istream&);
      Tensor<ScalarType, Symmetry, Name>& data_get(std::istream&);

      [[nodiscard]] std::string show() const;
      [[nodiscard]] std::string dump() const;
      Tensor<ScalarType, Symmetry, Name>& load(const std::string&) &;
      Tensor<ScalarType, Symmetry, Name>&& load(const std::string& string) && {
         return std::move(load(string));
      };

      using i_am_a_tensor = void;
   };
   template<typename T>
   concept is_tensor = requires {
      typename T::i_am_a_tensor;
   };

   template<
         is_tensor Tensor1,
         is_tensor Tensor2,
         std::ranges::range SetNameAndName = std::initializer_list<std::tuple<typename Tensor1::name_t, typename Tensor2::name_t>>>
   [[nodiscard]] auto contract(const Tensor1& tensor_1, const Tensor2& tensor_2, SetNameAndName&& contract_names) {
      return tensor_1.contract(tensor_2, std::forward<SetNameAndName>(contract_names));
   }

   /// \private
   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   struct TensorShape {
      Tensor<ScalarType, Symmetry, Name>* owner;
   };

   // TODO: middle 用edge operator表示一个待计算的张量, 在contract中用到
   // 因为contract的操作是这样的
   // merge gemm split
   // 上一次split可以和下一次的merge合并
   // 比较重要， 可以大幅减少对称性张量的分块
   /*
   template<is_scalar ScalarType, is_symmetry Symmetry, is_name Name>
   struct QuasiTensor {
      Tensor<ScalarType, Symmetry, Name> tensor;
      std::map<Name, std::vector<std::tuple<Name, edge_segment_t<Symmetry>>>> split_map;
      std::set<Name> reversed_set;
      std::vector<Name> res_name;

      QuasiTensor

      operator Tensor<ScalarType, Symmetry, Name>() && {
         return tensor.edge_operator({}, split_map, reversed_set, {}, std::move(res_name));
      }
      operator Tensor<ScalarType, Symmetry, Name>() const& {
         return tensor.edge_operator({}, split_map, reversed_set, {}, res_name);
      }

      Tensor<ScalarType, Symmetry, Name> merge_again(
            const std::set<Name>& merge_reversed_set,
            const std::map<Name, std::vector<Name>>& merge_map,
            std::vector<Name>&& merge_res_name,
            std::set<Name>& split_parity_mark,
            std::set<Name>& merge_parity_mark) {
         auto total_reversed_set = reversed_set; // merge_reversed_set
         return tensor.edge_operator(
               {},
               split_map,
               total_reversed_set,
               merge_map,
               merge_res_name,
               false,
               {{{}, split_parity_mark, {}, merge_parity_mark}});
      }
      QuasiTensor<ScalarType, Symmetry, Name>
   };
   */

   // TODO: lazy framework
   // 看一下idris是如何做的
   // 需要考虑深搜不可行的问题
   // 支持inplace操作

} // namespace TAT
#endif
