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

#include "PyTAT.hpp"

namespace TAT {
   void set_navigator(py::module_&);
   void set_random(py::module_&);
   py::class_<mpi_t> set_mpi(py::module_&);
   void set_name(py::module_&);
#define TAT_SINGLE_SCALAR_SYMMETRY(SCALARSHORT, SCALAR, SYM) \
   void dealing_Tensor_##SCALARSHORT##SYM( \
         py::module_& symmetry_m, \
         const std::string& scalar_short_name, \
         const std::string& scalar_name, \
         const std::string& symmetry_short_name); \
   void dealing_MPI_##SCALARSHORT##SYM(py::class_<mpi_t>& py_mpi_t);
   TAT_LOOP_ALL_SCALAR_SYMMETRY
#undef TAT_SINGLE_SCALAR_SYMMETRY

#ifndef TAT_PYTHON_MODULE
#define TAT_PYTHON_MODULE TAT
#endif

   PYBIND11_MODULE(TAT_PYTHON_MODULE, tat_m) {
      tat_m.doc() = "TAT is A Tensor library!";
      tat_m.attr("version") = version;
      tat_m.attr("information") = information;
      // random
      set_random(tat_m);
      // name
      set_name(tat_m);
      // symmetry and edge
      auto No_m = tat_m.def_submodule("No");
      declare_symmetry<NoSymmetry>(No_m, "No").def(py::init<>());
      declare_edge<NoSymmetry, void, false>(No_m, "No");
      declare_edge<NoSymmetry, void, false, edge_map_t>(No_m, "No");
      auto Z2_m = tat_m.def_submodule("Z2");
      declare_symmetry<Z2Symmetry>(Z2_m, "Z2")
            .def(implicit_init<Z2Symmetry, Z2>(), py::arg("z2"))
            .def_property_readonly("z2", [](const Z2Symmetry& symmetry) {
               return std::get<0>(symmetry);
            });
      declare_edge<Z2Symmetry, Z2, false>(Z2_m, "Z2");
      declare_edge<Z2Symmetry, Z2, false, edge_map_t>(Z2_m, "Z2");
      auto U1_m = tat_m.def_submodule("U1");
      declare_symmetry<U1Symmetry>(U1_m, "U1")
            .def(implicit_init<U1Symmetry, U1>(), py::arg("u1"))
            .def_property_readonly("u1", [](const U1Symmetry& symmetry) {
               return std::get<0>(symmetry);
            });
      declare_edge<U1Symmetry, U1, false>(U1_m, "U1");
      declare_edge<U1Symmetry, U1, false, edge_map_t>(U1_m, "U1");
      auto Fermi_m = tat_m.def_submodule("Fermi");
      declare_symmetry<FermiSymmetry>(Fermi_m, "Fermi")
            .def(implicit_init<FermiSymmetry, U1>(), py::arg("fermi"))
            .def_property_readonly("fermi", [](const FermiSymmetry& symmetry) {
               return std::get<0>(symmetry);
            });
      declare_edge<FermiSymmetry, U1, false>(Fermi_m, "Fermi");
      declare_edge<FermiSymmetry, U1, false, edge_map_t>(Fermi_m, "Fermi");
      auto FermiZ2_m = tat_m.def_submodule("FermiZ2");
      declare_symmetry<FermiZ2Symmetry>(FermiZ2_m, "FermiZ2")
            .def(py::init<U1, Z2>(), py::arg("fermi"), py::arg("z2"))
            .def(implicit_init<FermiZ2Symmetry, const std::tuple<U1, Z2>&>([](const std::tuple<U1, Z2>& p) {
                    return std::make_from_tuple<FermiZ2Symmetry>(p);
                 }),
                 py::arg("tuple_of_fermi_z2"))
            .def_property_readonly(
                  "fermi",
                  [](const FermiZ2Symmetry& symmetry) {
                     return std::get<0>(symmetry);
                  })
            .def_property_readonly("z2", [](const FermiZ2Symmetry& symmetry) {
               return std::get<1>(symmetry);
            });
      declare_edge<FermiZ2Symmetry, std::tuple<U1, Z2>, true>(FermiZ2_m, "FermiZ2");
      declare_edge<FermiZ2Symmetry, std::tuple<U1, Z2>, true, edge_map_t>(FermiZ2_m, "FermiZ2");
      auto FermiU1_m = tat_m.def_submodule("FermiU1");
      declare_symmetry<FermiU1Symmetry>(FermiU1_m, "FermiU1")
            .def(py::init<U1, U1>(), py::arg("fermi"), py::arg("u1"))
            .def(implicit_init<FermiU1Symmetry, const std::tuple<U1, U1>&>([](const std::tuple<U1, U1>& p) {
                    return std::make_from_tuple<FermiU1Symmetry>(p);
                 }),
                 py::arg("tuple_of_fermi_u1"))
            .def_property_readonly(
                  "fermi",
                  [](const FermiU1Symmetry& symmetry) {
                     return std::get<0>(symmetry);
                  })
            .def_property_readonly("u1", [](const FermiU1Symmetry& symmetry) {
               return std::get<1>(symmetry);
            });
      declare_edge<FermiU1Symmetry, std::tuple<U1, U1>, true>(FermiU1_m, "FermiU1");
      declare_edge<FermiU1Symmetry, std::tuple<U1, U1>, true, edge_map_t>(FermiU1_m, "FermiU1");
      // tensor
#define TAT_SINGLE_SCALAR_SYMMETRY(SCALARSHORT, SCALAR, SYM) dealing_Tensor_##SCALARSHORT##SYM(SYM##_m, #SCALARSHORT, #SCALAR, #SYM);
      TAT_LOOP_ALL_SCALAR_SYMMETRY
#undef TAT_SINGLE_SCALAR_SYMMETRY
      // get tensor
      set_navigator(tat_m);
      // normal = no
      tat_m.attr("Normal") = tat_m.attr("No");
      at_exit.release();
   }
} // namespace TAT
