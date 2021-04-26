/**
 * Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#ifndef PYTAT_HPP
#define PYTAT_HPP

#include <random>
#include <sstream>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// use simple no symmetry to avoid load problem
#define TAT_USE_SIMPLE_NOSYMMETRY
#include <TAT/TAT.hpp>

#define TAT_SINGLE_SYMMETRY_ALL_SCALAR(SYM) \
   TAT_SINGLE_SCALAR_SYMMETRY(S, float, SYM) \
   TAT_SINGLE_SCALAR_SYMMETRY(D, double, SYM) \
   TAT_SINGLE_SCALAR_SYMMETRY(C, std::complex<float>, SYM) \
   TAT_SINGLE_SCALAR_SYMMETRY(Z, std::complex<double>, SYM)

#define TAT_LOOP_ALL_SCALAR_SYMMETRY \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(No) \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(Z2) \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(U1) \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(Fermi) \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(FermiZ2) \
   TAT_SINGLE_SYMMETRY_ALL_SCALAR(FermiU1)

namespace TAT {
   namespace py = pybind11;

   inline auto random_engine = std::default_random_engine(std::random_device()());
   inline void set_random_seed(unsigned int seed) {
      random_engine.seed(seed);
   }

   template<typename T>
   std::unique_ptr<T> to_unique(T&& object) {
      return std::make_unique<T>(std::move(object));
   }

   struct AtExit {
      std::vector<std::function<void()>> function_list;
      void operator()(std::function<void()>&& function) {
         function_list.push_back(std::move(function));
      }
      void release() {
         for (auto& function : function_list) {
            function();
         }
         function_list.resize(0);
      }
   };
   inline AtExit at_exit;

   template<typename Type, typename Args>
   auto implicit_init() {
      at_exit([]() {
         py::implicitly_convertible<Args, Type>();
      });
      return py::init<Args>();
   }

   template<typename Type, typename Args, typename Func>
   auto implicit_init(Func&& func) {
      at_exit([]() {
         py::implicitly_convertible<Args, Type>();
      });
      return py::init(func);
   }

   template<typename ScalarType, typename Symmetry>
   struct blocks_of_tensor {
      py::object tensor;
   };

   template<typename ScalarType, typename Symmetry>
   struct unordered_block_of_tensor {
      py::object tensor;
      std::map<DefaultName, Symmetry> position;
   };

   template<typename ScalarType, typename Symmetry>
   struct ordered_block_of_tensor {
      py::object tensor;
      std::vector<std::tuple<DefaultName, Symmetry>> position;
   };

   template<typename Symmetry>
   auto generate_vector_of_name_and_symmetry(const std::vector<DefaultName>& position) {
      auto real_position = std::vector<std::tuple<DefaultName, Symmetry>>();
      for (const auto& n : position) {
         real_position.push_back({n, Symmetry()});
      }
      return real_position;
   }

   template<typename Block>
   auto try_get_numpy_array(Block& block) {
      auto result = py::cast(block, py::return_value_policy::move);
      try {
         return py::module_::import("numpy").attr("array")(result, py::arg("copy") = false);
      } catch (const py::error_already_set&) {
         return result;
      }
   }

   template<typename Block>
   auto try_set_numpy_array(Block& block, const py::object& object) {
      auto result = py::cast(block, py::return_value_policy::move);
      try {
         result = py::module_::import("numpy").attr("array")(result, py::arg("copy") = false);
      } catch (const py::error_already_set&) {
         throw std::runtime_error("Cannot import numpy but setting block of tensor need numpy");
      }
      result.attr("__setitem__")(py::ellipsis(), object);
   }

   template<typename Tensor, typename Symmetry>
   auto& find_block(Tensor& tensor, const std::map<DefaultName, Symmetry>& map) {
      if constexpr (Symmetry::length == 0) {
         return tensor.core->blocks.front().second;
      } else {
         std::vector<Symmetry> symmetries;
         symmetries.reserve(tensor.names.size());
         for (const auto& i : tensor.names) {
            symmetries.push_back(map.at(i));
         }
         return map_at(tensor.core->blocks, symmetries);
      }
   }

   template<typename ScalarType, typename Symmetry>
   void declare_tensor(
         py::module_& symmetry_m,
         const std::string& scalar_short_name,
         const std::string& scalar_name,
         const std::string& symmetry_short_name) {
      auto self_m = symmetry_m.def_submodule(scalar_short_name.c_str());
      auto block_m = self_m.def_submodule("Block");
      using T = Tensor<ScalarType, Symmetry>;
      using E = Edge<Symmetry>;
      using BS = blocks_of_tensor<ScalarType, Symmetry>;
      using B1 = unordered_block_of_tensor<ScalarType, Symmetry>;
      using B2 = ordered_block_of_tensor<ScalarType, Symmetry>;
      std::string tensor_name = scalar_short_name + symmetry_short_name;
      py::class_<BS>(
            block_m,
            "Blocks",
            ("Blocks of a tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_short_name + "Symmetry").c_str())
            .def("__getitem__",
                 [](const BS& bs, std::map<DefaultName, Symmetry> position) {
                    auto block = B1{bs.tensor, std::move(position)};
                    return try_get_numpy_array(block);
                 })
            .def("__setitem__",
                 [](BS& bs, std::map<DefaultName, Symmetry> position, const py::object& object) {
                    auto block = B1{bs.tensor, std::move(position)};
                    try_set_numpy_array(block, object);
                 })
            .def("__getitem__",
                 [](const BS& bs, std::vector<std::tuple<DefaultName, Symmetry>> position) {
                    auto block = B2{bs.tensor, std::move(position)};
                    return try_get_numpy_array(block);
                 })
            .def("__setitem__",
                 [](BS& bs, std::vector<std::tuple<DefaultName, Symmetry>> position, const py::object& object) {
                    auto block = B2{bs.tensor, std::move(position)};
                    try_set_numpy_array(block, object);
                 })
            .def("__getitem__",
                 [](const BS& bs, const std::vector<DefaultName>& position) {
                    auto block = B2{bs.tensor, generate_vector_of_name_and_symmetry<Symmetry>(position)};
                    return try_get_numpy_array(block);
                 })
            .def("__setitem__", [](BS& bs, const std::vector<DefaultName>& position, const py::object& object) {
               auto block = B2{bs.tensor, generate_vector_of_name_and_symmetry<Symmetry>(position)};
               try_set_numpy_array(block, object);
            });
      py::class_<B1>(
            block_m,
            "UnorderedBlock",
            ("Unordered Block of a tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_short_name + "Symmetry").c_str(),
            py::buffer_protocol())
            .def_buffer([](B1& b) {
               // 返回的buffer是可变的
               auto& tensor = py::cast<T&>(b.tensor);
               auto& block = find_block(tensor, b.position);
               const Rank rank = tensor.names.size();
               auto dimensions = std::vector<Size>(rank);
               auto leadings = std::vector<Size>(rank);
               for (auto i = 0; i < rank; i++) {
                  dimensions[i] = map_at(tensor.core->edges[i].map, b.position[tensor.names[i]]);
                  // 使用operator[]在NoSymmetry时获得默认对称性, 从而得到仅有的维度
               }
               for (auto i = rank; i-- > 0;) {
                  if (i == rank - 1) {
                     leadings[i] = sizeof(ScalarType);
                  } else {
                     leadings[i] = leadings[i + 1] * dimensions[i + 1];
                  }
               }
               return py::buffer_info{
                     block.data(),
                     sizeof(ScalarType),
                     py::format_descriptor<ScalarType>::format(),
                     rank,
                     std::move(dimensions),
                     std::move(leadings)};
            });
      py::class_<B2>(
            block_m,
            "OrderedBlock",
            ("Ordered Block of a tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_short_name + "Symmetry").c_str(),
            py::buffer_protocol())
            .def_buffer([](B2& b) {
               // 返回的buffer是可变的
               auto& tensor = py::cast<T&>(b.tensor);
               auto position_map = std::map<DefaultName, Symmetry>();
               for (const auto& [name, symmetry] : b.position) {
                  position_map[name] = symmetry;
               }
               auto& block = find_block(tensor, position_map);
               const Rank rank = tensor.names.size();
               auto dimensions = std::vector<Size>(rank);
               auto leadings = std::vector<Size>(rank);
               for (auto i = 0; i < rank; i++) {
                  dimensions[i] = map_at(tensor.core->edges[i].map, position_map[tensor.names[i]]);
                  // 使用operator[]在NoSymmetry时获得默认对称性, 从而得到仅有的维度
               }
               for (auto i = rank; i-- > 0;) {
                  if (i == rank - 1) {
                     leadings[i] = sizeof(ScalarType);
                  } else {
                     leadings[i] = leadings[i + 1] * dimensions[i + 1];
                  }
               }
               auto real_dimensions = std::vector<Size>(rank);
               auto real_leadings = std::vector<Size>(rank);
               for (auto i = 0; i < rank; i++) {
                  auto j = map_find(tensor.name_to_index, std::get<0>(b.position[i]))->second;
                  real_dimensions[i] = dimensions[j];
                  real_leadings[i] = leadings[j];
               }
               return py::buffer_info{
                     block.data(),
                     sizeof(ScalarType),
                     py::format_descriptor<ScalarType>::format(),
                     rank,
                     std::move(real_dimensions),
                     std::move(real_leadings)};
            });
      ScalarType one = 1;
      if constexpr (is_complex<ScalarType>) {
         one = ScalarType(1, 1);
      }
      auto tensor_t = py::class_<T>(
                            self_m,
                            "Tensor",
                            ("Tensor with scalar type as " + scalar_name + " and symmetry type " + symmetry_short_name + "Symmetry").c_str())
                            .def_readonly("name", &T::names, "Names of all edge of the tensor")
                            .def_property_readonly(
                                  "edge",
                                  [](T& tensor) -> std::vector<E>& {
                                     return tensor.core->edges;
                                  },
                                  "Edges of tensor")
                            .def_property_readonly(
                                  "data",
                                  [](T & tensor) -> auto& { return tensor.core->blocks; },
                                  "All block data of the tensor")
                            .def(ScalarType() + py::self)
                            .def(py::self + ScalarType())
                            .def(py::self += ScalarType())
                            .def(ScalarType() - py::self)
                            .def(py::self - ScalarType())
                            .def(py::self -= ScalarType())
                            .def(ScalarType() * py::self)
                            .def(py::self * ScalarType())
                            .def(py::self *= ScalarType())
                            .def(ScalarType() / py::self)
                            .def(py::self / ScalarType())
                            .def(py::self /= ScalarType())
#define TAT_LOOP_OPERATOR(ANOTHERSCALAR) \
   def(py::self + Tensor<ANOTHERSCALAR, Symmetry>()) \
         .def(py::self - Tensor<ANOTHERSCALAR, Symmetry>()) \
         .def(py::self* Tensor<ANOTHERSCALAR, Symmetry>()) \
         .def(py::self / Tensor<ANOTHERSCALAR, Symmetry>())
                            .TAT_LOOP_OPERATOR(float)
                            .TAT_LOOP_OPERATOR(double)
                            .TAT_LOOP_OPERATOR(std::complex<float>)
                            .TAT_LOOP_OPERATOR(std::complex<double>)
#undef TAT_LOOP_OPERATOR
#define TAT_LOOP_OPERATOR(ANOTHERSCALAR) \
   def(py::self += Tensor<ANOTHERSCALAR, Symmetry>()) \
         .def(py::self -= Tensor<ANOTHERSCALAR, Symmetry>()) \
         .def(py::self *= Tensor<ANOTHERSCALAR, Symmetry>()) \
         .def(py::self /= Tensor<ANOTHERSCALAR, Symmetry>())
                            .TAT_LOOP_OPERATOR(float)
                            .TAT_LOOP_OPERATOR(double);
      if constexpr (is_complex<ScalarType>) {
         tensor_t.TAT_LOOP_OPERATOR(std::complex<float>).TAT_LOOP_OPERATOR(std::complex<double>);
         tensor_t.def("__complex__", [](const T& tensor) {
            return ScalarType(tensor);
         });
      } else {
         tensor_t.def("__float__", [](const T& tensor) {
            return ScalarType(tensor);
         });
         tensor_t.def("__complex__", [](const T& tensor) {
            return std::complex<ScalarType>(ScalarType(tensor));
         });
      }
      tensor_t
#undef TAT_LOOP_OPERATOR
            .def("__str__",
                 [](const T& tensor) {
                    if (tensor.core) {
                       return tensor.show();
                    } else {
                       return std::string("{}");
                    }
                 })
            .def("__repr__",
                 [tensor_name](const T& tensor) {
                    auto out = std::stringstream();
                    out << tensor_name << "Tensor";
                    out << '{';
                    if (tensor.core) {
                       out << console_green << "names" << console_origin << ':';
                       out << tensor.names << ',';
                       out << console_green << "edges" << console_origin << ':';
                       out << tensor.core->edges;
                    }
                    out << '}';
                    return out.str();
                 })
            .def(py::init<>(), "Default Constructor")
            .def(py::init<std::vector<DefaultName>, std::vector<E>, bool>(),
                 py::arg("names"),
                 py::arg("edges"),
                 py::arg("auto_reverse") = false,
                 "Construct tensor with edge names and edge shapes")
            .def(py::init<ScalarType>(), py::arg("number"), "Create rank 0 tensor with only one element")
            .def(py::init<>([](const std::string& string) {
                    auto ss = std::stringstream(string);
                    auto result = T();
                    ss >> result;
                    return result;
                 }),
                 "Read tensor from text string")
            .def_static(
                  "one",
                  &T::one,
                  py::arg("number"),
                  py::arg("names"),
                  py::arg("edge_symmetry") = py::list(),
                  py::arg("edge_arrow") = py::list(),
                  "Create tensor with high rank but containing only one element")
            .def("copy", &T::copy, "Deep copy a tensor")
            .def("__copy__", &T::copy)
            .def("__deepcopy__", &T::copy)
            .def("same_shape", &T::same_shape, "Create a tensor with same shape")
            .def("map", &T::template map<std::function<ScalarType(ScalarType)>>, py::arg("function"), "Out-place map every element of a tensor")
            .def(
                  "transform",
                  [](T& tensor, std::function<ScalarType(ScalarType)>& function) -> T& {
                     return tensor.transform(function);
                  },
                  py::arg("function"),
                  "In-place map every element of a tensor",
                  py::return_value_policy::reference_internal)
            .def(
                  "sqrt",
                  [](const T& tensor) {
                     return tensor.map([](ScalarType value) {
                        return std::sqrt(value);
                     });
                  },
                  "Get elementwise square root")
            .def(
                  "set",
                  [](T& tensor, std::function<ScalarType()>& function) -> T& {
                     return tensor.set(function);
                  },
                  py::arg("function"),
                  "Set every element of a tensor by a function",
                  py::return_value_policy::reference_internal)
            .def(
                  "zero",
                  [](T& tensor) -> T& {
                     return tensor.zero();
                  },
                  "Set all element zero",
                  py::return_value_policy::reference_internal)
            .def(
                  "range",
                  [](T& tensor, ScalarType first, ScalarType step) -> T& {
                     return tensor.range(first, step);
                  },
                  py::arg("first") = 0,
                  py::arg("step") = 1,
                  "Useful function generate simple data in tensor element for test",
                  py::return_value_policy::reference_internal)
            .def_property_readonly(
                  "block",
                  [](const py::object& tensor) {
                     return blocks_of_tensor<ScalarType, Symmetry>{tensor};
                  })
            .def(
                  "__getitem__",
                  [](const T& tensor, const std::map<DefaultName, typename T::EdgePoint>& position) {
                     return tensor.at(position);
                  },
                  py::arg("dictionary_from_name_to_symmetry_and_dimension"))
            .def(
                  "__setitem__",
                  [](T& tensor, const std::map<DefaultName, typename T::EdgePoint>& position, const ScalarType& value) {
                     tensor.at(position) = value;
                  },
                  py::arg("dictionary_from_name_to_symmetry_and_dimension"),
                  py::arg("value"))
            .def("shrink",
                 &T::template shrink<std::map<DefaultName, typename T::EdgePoint>>,
                 "Shrink Edge of tensor",
                 py::arg("configure"),
                 py::arg("new_name") = ",No_New_Name",
                 py::arg("arrow") = false)
            .def("expand",
                 &T::template expand<std::map<DefaultName, typename T::EdgePointWithArrow>>,
                 "Expand Edge of tensor",
                 py::arg("configure"),
                 py::arg("old_name") = ",No_Old_Name")
            .def(
                  "to",
                  [](const T& tensor, const py::object& object) -> py::object {
                     auto string = py::str(object);
                     auto contain = [&string](const char* other) {
                        return py::cast<bool>(string.attr("__contains__")(other));
                     };
                     if (contain("float32")) {
                        return py::cast(tensor.template to<float>(), py::return_value_policy::move);
                     } else if (contain("complex32")) {
                        return py::cast(tensor.template to<std::complex<float>>(), py::return_value_policy::move);
                     } else if (contain("float")) {
                        return py::cast(tensor.template to<double>(), py::return_value_policy::move);
                     } else if (contain("complex")) {
                        return py::cast(tensor.template to<std::complex<double>>(), py::return_value_policy::move);
                     } else if (contain("S")) {
                        return py::cast(tensor.template to<float>(), py::return_value_policy::move);
                     } else if (contain("D")) {
                        return py::cast(tensor.template to<double>(), py::return_value_policy::move);
                     } else if (contain("C")) {
                        return py::cast(tensor.template to<std::complex<float>>(), py::return_value_policy::move);
                     } else if (contain("Z")) {
                        return py::cast(tensor.template to<std::complex<double>>(), py::return_value_policy::move);
                     } else {
                        throw std::runtime_error("Invalid scalar type in type conversion");
                     }
                  },
                  "Convert to other scalar type tensor")
            .def("norm_max", &T::template norm<-1>, "Get -1 norm, namely max absolute value")
            .def("norm_num", &T::template norm<0>, "Get 0 norm, namely number of element, note: not check whether equal to 0")
            .def("norm_sum", &T::template norm<1>, "Get 1 norm, namely summation of all element absolute value")
            .def("norm_2", &T::template norm<2>, "Get 2 norm")
            .def(
                  "edge_rename",
                  [](const T& tensor, const std::map<DefaultName, DefaultName>& dictionary) {
                     return tensor.edge_rename(dictionary);
                  },
                  py::arg("name_dictionary"),
                  "Rename names of edges, which will not copy data")
            .def("transpose",
                 &T::template transpose<std::vector<DefaultName>>,
                 py::arg("new_names"),
                 "Transpose the tensor to the order of new names")
            .def("reverse_edge",
                 &T::template reverse_edge<std::set<DefaultName>, std::set<DefaultName>>,
                 py::arg("reversed_name_set"),
                 py::arg("apply_parity") = false,
                 py::arg("parity_exclude_name_set") = py::set(),
                 "Reverse fermi arrow of several edge")
            .def("merge_edge",
                 &T::template merge_edge<std::map<DefaultName, std::vector<DefaultName>>, std::set<DefaultName>, std::set<DefaultName>>,
                 py::arg("merge_map"),
                 py::arg("apply_parity") = false,
                 py::arg("parity_exclude_name_merge_set") = py::set(),
                 py::arg("parity_exclude_name_reverse_set") = py::set(),
                 "Merge several edges of the tensor into ones")
            .def("split_edge",
                 &T::template split_edge<std::map<DefaultName, std::vector<std::tuple<DefaultName, edge_map_t<Symmetry>>>>, std::set<DefaultName>>,
                 py::arg("split_map"),
                 py::arg("apply_parity") = false,
                 py::arg("parity_exclude_name_split_set") = py::set(),
                 "Split edges of a tensor to many edges")
            .def(
                  "edge_operator",
                  [](const T& tensor,
                     const std::map<DefaultName, DefaultName>& rename_map,
                     const std::map<DefaultName, std::vector<std::tuple<DefaultName, edge_map_t<Symmetry>>>>& split_map,
                     const std::set<DefaultName>& reversed_name,
                     const std::map<DefaultName, std::vector<DefaultName>>& merge_map,
                     const std::vector<DefaultName>& new_names,
                     const bool apply_parity,
                     std::set<DefaultName> parity_exclude_name_split_set,
                     std::set<DefaultName> parity_exclude_name_reverse_set,
                     std::set<DefaultName> parity_exclude_name_reverse_before_merge_set,
                     std::set<DefaultName> parity_exclude_name_merge_set) {
                     return tensor.edge_operator(
                           rename_map,
                           split_map,
                           reversed_name,
                           merge_map,
                           new_names,
                           apply_parity,
                           parity_exclude_name_split_set,
                           parity_exclude_name_reverse_set,
                           parity_exclude_name_reverse_before_merge_set,
                           parity_exclude_name_merge_set);
                  },
                  py::arg("rename_map"),
                  py::arg("split_map"),
                  py::arg("reversed_name"),
                  py::arg("merge_map"),
                  py::arg("new_names"),
                  py::arg("apply_parity") = false,
                  py::arg("parity_exclude_name_split_set") = py::set(),
                  py::arg("parity_exclude_name_reverse_set") = py::set(),
                  py::arg("parity_exclude_name_reverse_before_merge_set") = py::set(),
                  py::arg("parity_exclude_name_merge_set") = py::set(),
                  "Tensor Edge Operator")
#define TAT_LOOP_CONTRACT(ANOTHERSCALAR) \
   def( \
         "contract", \
         [](const T& tensor_1, const Tensor<ANOTHERSCALAR, Symmetry>& tensor_2, std::set<std::tuple<DefaultName, DefaultName>> contract_names) { \
            return tensor_1.contract(tensor_2, std::move(contract_names)); \
         }, \
         py::arg("another_tensor"), \
         py::arg("contract_names"), \
         "Contract two tensors")
            .TAT_LOOP_CONTRACT(float)
            .TAT_LOOP_CONTRACT(double)
            .TAT_LOOP_CONTRACT(std::complex<float>)
            .TAT_LOOP_CONTRACT(std::complex<double>)
#undef TAT_LOOP_CONTRACT
            .def(
                  "contract_all_edge",
                  [](const T& tensor) {
                     return tensor.contract_all_edge();
                  },
                  "Contract all edge with conjugate tensor")
            .def(
                  "contract_all_edge",
                  [](const T& tensor, const T& other) {
                     return tensor.contract_all_edge(other);
                  },
                  py::arg("another_tensor"),
                  "Contract as much as possible with another tensor on same name edges")
            .def(
                  "identity",
                  [](T& tensor, std::set<std::tuple<DefaultName, DefaultName>>& pairs) -> T& {
                     return tensor.identity(pairs);
                  },
                  py::arg("pairs"),
                  "Get a identity tensor with same shape")
            .def("exponential",
                 &T::template exponential<std::set<std::tuple<DefaultName, DefaultName>>>,
                 py::arg("pairs"),
                 py::arg("step") = 2,
                 "Calculate exponential like matrix")
            .def("conjugate", &T::conjugate, "Get the conjugate Tensor")
            .def("trace", &T::template trace<std::set<std::tuple<DefaultName, DefaultName>>>)
            .def(
                  "svd",
                  [](const T& tensor,
                     const std::set<DefaultName>& free_name_set_u,
                     const DefaultName& common_name_u,
                     const DefaultName& common_name_v,
                     Size cut,
                     const DefaultName& singular_name_u,
                     const DefaultName& singular_name_v) {
                     auto result = tensor.svd(free_name_set_u, common_name_u, common_name_v, cut, singular_name_u, singular_name_v);
                     return py::make_tuple(std::move(result.U), std::move(result.S), std::move(result.V));
                  },
                  py::arg("free_name_set_u"),
                  py::arg("common_name_u"),
                  py::arg("common_name_v"),
                  py::arg("cut") = Size(-1),
                  py::arg("singular_name_u") = DefaultName("DefaultSingularNameU"),
                  py::arg("singular_name_v") = DefaultName("DefaultSingularNameV"),
                  "Singular value decomposition")
            .def(
                  "qr",
                  [](const T& tensor,
                     char free_name_direction,
                     const std::set<DefaultName>& free_name_set,
                     const DefaultName& common_name_q,
                     const DefaultName& common_name_r) {
                     auto result = tensor.qr(free_name_direction, free_name_set, common_name_q, common_name_r);
                     return py::make_tuple(std::move(result.Q), std::move(result.R));
                  },
                  py::arg("free_name_direction"),
                  py::arg("free_name_set"),
                  py::arg("common_name_q"),
                  py::arg("common_name_r"),
                  "QR decomposition")
            .def(
                  "multiple",
                  [](T& tensor, const typename T::SingularType& s, const DefaultName& name, char direction, bool division) {
                     return tensor.multiple(s, name, direction, division);
                  },
                  py::arg("singular"),
                  py::arg("name"),
                  py::arg("direction"),
                  py::arg("division") = false,
                  "Multiple with singular generated by svd")
            .def(
                  "dump",
                  [](T& tensor) {
                     return py::bytes(tensor.dump());
                  },
                  "dump Tensor to bytes")
            .def(
                  "load",
                  [](T& tensor, const py::bytes& bytes) -> T& {
                     return tensor.load(std::string(bytes));
                  },
                  "Load Tensor from bytes",
                  py::return_value_policy::reference_internal)
            .def(py::pickle(
                  [](const T& tensor) {
                     return py::bytes(tensor.dump());
                  },
                  // 这里必须是make_unique, 很奇怪, 可能是pybind11的bug
                  [](const py::bytes& bytes) {
                     return to_unique(T().load(std::string(bytes)));
                  }))
            .def(
                  "rand",
                  [](T& tensor, ScalarType min, ScalarType max) -> T& {
                     if constexpr (is_complex<ScalarType>) {
                        auto distribution_real = std::uniform_real_distribution<real_scalar<ScalarType>>(min.real(), max.real());
                        auto distribution_imag = std::uniform_real_distribution<real_scalar<ScalarType>>(min.imag(), max.imag());
                        return tensor.set([&distribution_real, &distribution_imag]() -> ScalarType {
                           return {distribution_real(random_engine), distribution_imag(random_engine)};
                        });
                     } else {
                        auto distribution = std::uniform_real_distribution<real_scalar<ScalarType>>(min, max);
                        return tensor.set([&distribution]() {
                           return distribution(random_engine);
                        });
                     }
                  },
                  py::arg("min") = 0,
                  py::arg("max") = one,
                  "Set Uniform Random Number into Tensor",
                  py::return_value_policy::reference_internal)
            .def(
                  "randn",
                  [](T& tensor, ScalarType mean, ScalarType stddev) -> T& {
                     if constexpr (is_complex<ScalarType>) {
                        auto distribution_real = std::normal_distribution<real_scalar<ScalarType>>(mean.real(), stddev.real());
                        auto distribution_imag = std::normal_distribution<real_scalar<ScalarType>>(mean.imag(), stddev.imag());
                        return tensor.set([&distribution_real, &distribution_imag]() -> ScalarType {
                           return {distribution_real(random_engine), distribution_imag(random_engine)};
                        });
                     } else {
                        auto distribution = std::normal_distribution<real_scalar<ScalarType>>(mean, stddev);
                        return tensor.set([&distribution]() {
                           return distribution(random_engine);
                        });
                     }
                  },
                  py::arg("mean") = 0,
                  py::arg("stddev") = one,
                  "Set Normal Distribution Random Number into Tensor",
                  py::return_value_policy::reference_internal);
   }

   template<typename Symmetry, typename Element, bool IsTuple, template<typename, bool = false> class EdgeType = Edge>
   auto declare_edge(py::module_& symmetry_m, const char* name) {
      auto result = py::class_<EdgeType<Symmetry>>(
                          symmetry_m,
                          is_edge<EdgeType<Symmetry>> ? "Edge" : "EdgeMap",
                          ("Edge with symmetry type as " + std::string(name) + "Symmetry").c_str())
                          .def_readonly("map", &EdgeType<Symmetry>::map)
                          .def(implicit_init<EdgeType<Symmetry>, Size>(), py::arg("dimension"), "Edge with only one symmetry")
                          .def(implicit_init<EdgeType<Symmetry>, std::map<Symmetry, Size>>(),
                               py::arg("dictionary_from_symmetry_to_dimension"),
                               "Create Edge with dictionary from symmetry to dimension")
                          .def(implicit_init<EdgeType<Symmetry>, const std::set<Symmetry>&>(),
                               py::arg("set_of_symmetry"),
                               "Edge with several symmetries which dimensions are all one");
      py::implicitly_convertible<py::dict, EdgeType<Symmetry>>();
      py::implicitly_convertible<py::set, EdgeType<Symmetry>>();
      if constexpr (is_edge<EdgeType<Symmetry>>) {
         result = result.def("__str__",
                             [](const EdgeType<Symmetry>& edge) {
                                auto out = std::stringstream();
                                out << edge;
                                return out.str();
                             })
                        .def("__repr__", [name](const EdgeType<Symmetry>& edge) {
                           auto out = std::stringstream();
                           out << name << "Edge";
                           if constexpr (Symmetry::length == 0) {
                              out << "[";
                           }
                           out << edge;
                           if constexpr (Symmetry::length == 0) {
                              out << "]";
                           }
                           return out.str();
                        });
      }
      if constexpr (!std::is_same_v<Element, void>) {
         // is not no symmetry
         result = result.def(implicit_init<EdgeType<Symmetry>, std::map<Element, Size>>([](const std::map<Element, Size>& element_map) {
                                auto symmetry_map = std::map<Symmetry, Size>();
                                for (const auto& [key, value] : element_map) {
                                   if constexpr (IsTuple) {
                                      symmetry_map[std::make_from_tuple<Symmetry>(key)] = value;
                                   } else {
                                      symmetry_map[Symmetry(key)] = value;
                                   }
                                }
                                return EdgeType<Symmetry>(std::move(symmetry_map));
                             }),
                             py::arg("dictionary_from_symmetry_to_dimension"),
                             "Create Edge with dictionary from symmetry to dimension")
                        .def(implicit_init<EdgeType<Symmetry>, const std::set<Element>&>([](const std::set<Element>& element_set) {
                                auto symmetry_set = std::set<Symmetry>();
                                for (const auto& element : element_set) {
                                   if constexpr (IsTuple) {
                                      symmetry_set.insert(std::make_from_tuple<Symmetry>(element));
                                   } else {
                                      symmetry_set.insert(Symmetry(element));
                                   }
                                }
                                return EdgeType<Symmetry>(std::move(symmetry_set));
                             }),
                             py::arg("set_of_symmetry"),
                             "Edge with several symmetries which dimensions are all one");
      }
      if constexpr (Symmetry::is_fermi_symmetry && is_edge<EdgeType<Symmetry>>) {
         // is fermi symmetry 且没有强制设置为BoseEdge
         result = result.def_readonly("arrow", &EdgeType<Symmetry>::arrow, "Fermi Arrow of the edge")
                        .def(py::init<Arrow, std::map<Symmetry, Size>>(),
                             py::arg("arrow"),
                             py::arg("dictionary_from_symmetry_to_dimension"),
                             "Fermi Edge created from arrow and dictionary")
                        .def(implicit_init<EdgeType<Symmetry>, std::tuple<Arrow, std::map<Symmetry, Size>>>(
                                   [](std::tuple<Arrow, std::map<Symmetry, Size>> p) {
                                      return std::make_from_tuple<EdgeType<Symmetry>>(std::move(p));
                                   }),
                             py::arg("tuple_of_arrow_and_dictionary"),
                             "Fermi Edge created from arrow and dictionary");
         if constexpr (!std::is_same_v<Element, void>) {
            // true if not FermiSymmetry
            result = result.def(py::init([](Arrow arrow, const std::map<Element, Size>& element_map) {
                                   auto symmetry_map = std::map<Symmetry, Size>();
                                   for (const auto& [key, value] : element_map) {
                                      if constexpr (IsTuple) {
                                         symmetry_map[std::make_from_tuple<Symmetry>(key)] = value;
                                      } else {
                                         symmetry_map[Symmetry(key)] = value;
                                      }
                                   }
                                   return EdgeType<Symmetry>(arrow, std::move(symmetry_map));
                                }),
                                py::arg("arrow"),
                                py::arg("dictionary_from_symmetry_to_dimension"),
                                "Fermi Edge created from arrow and dictionary")
                           .def(implicit_init<EdgeType<Symmetry>, std::tuple<Arrow, std::map<Element, Size>>>(
                                      [](const std::tuple<Arrow, std::map<Element, Size>>& p) {
                                         const auto& [arrow, element_map] = p;
                                         auto symmetry_map = std::map<Symmetry, Size>();
                                         for (const auto& [key, value] : element_map) {
                                            if constexpr (IsTuple) {
                                               symmetry_map[std::make_from_tuple<Symmetry>(key)] = value;
                                            } else {
                                               symmetry_map[Symmetry(key)] = value;
                                            }
                                         }
                                         return EdgeType<Symmetry>(arrow, std::move(symmetry_map));
                                      }),
                                py::arg("tuple_of_arrow_and_dictionary"),
                                "Fermi Edge created from arrow and dictionary");
         }
      }
      return result;
   }

   template<typename Symmetry>
   auto declare_symmetry(py::module_& symmetry_m, const char* name) {
      return py::class_<Symmetry>(symmetry_m, "Symmetry", (std::string(name) + "Symmetry").c_str())
            .def(py::self < py::self)
            .def(py::self > py::self)
            .def(py::self <= py::self)
            .def(py::self >= py::self)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def(-py::self)
            .def("__hash__",
                 [](const Symmetry& symmetry) {
                    return py::hash(py::cast(static_cast<const typename Symmetry::base_tuple&>(symmetry)));
                 })
            .def("__repr__",
                 [=](const Symmetry& symmetry) {
                    auto out = std::stringstream();
                    out << name;
                    out << "Symmetry";
                    out << "[";
                    out << symmetry;
                    out << "]";
                    return out.str();
                 })
            .def("__str__", [=](const Symmetry& symmetry) {
               auto out = std::stringstream();
               out << symmetry;
               return out.str();
            });
   }
} // namespace TAT

#endif
