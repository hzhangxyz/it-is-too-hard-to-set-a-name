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
   void set_navigator(py::module_& tat_m) {
      tat_m.def("navigator", [tat_m](const py::args& args, const py::kwargs& kwargs) -> py::object {
         if (py::len(args) == 0 && py::len(kwargs) == 0) {
            tat_m.attr("mpi").attr("print")(tat_m.attr("information"));
            return py::none();
         }
         auto text = py::str(py::make_tuple(args, kwargs));
         auto contain = [&text](const char* string) {
            return py::cast<bool>(text.attr("__contains__")(string));
         };
         if (contain("mpi") || contain("MPI")) {
            return tat_m.attr("mpi");
         }
         std::string scalar = "";
         std::string fermi = "";
         std::string symmetry = "";
         if (contain("Fermi")) {
            fermi = "Fermi";
         }
         if (contain("Bose")) {
            if (fermi != "") {
               throw std::runtime_error("Fermi Ambiguous");
            }
         }
         if (contain("U1")) {
            symmetry = "U1";
         }
         if (contain("Z2")) {
            if (symmetry == "") {
               symmetry = "Z2";
            } else {
               throw std::runtime_error("Symmetry Ambiguous");
            }
         }
         if (contain("No")) {
            if (symmetry != "") {
               throw std::runtime_error("Symmetry Ambiguous");
            }
         }
         if (symmetry == "" && fermi == "") {
            symmetry = "No";
         }
         if (contain("complex")) {
            scalar = "Z";
         }
         if (contain("complex32")) {
            scalar = "C";
         }
         if (contain("float")) {
            if (scalar == "") {
               scalar = "D";
            } else {
               throw std::runtime_error("Scalar Ambiguous");
            }
         }
         if (contain("float32")) {
            if (scalar == "" || scalar == "D") {
               scalar = "S";
            } else {
               throw std::runtime_error("Scalar Ambiguous");
            }
         }
         if (scalar == "") {
            throw std::runtime_error("Scalar Ambiguous");
         }
         return tat_m.attr((fermi + symmetry).c_str()).attr(scalar.c_str()).attr("Tensor");
      });
      auto py_type = py::module_::import("builtins").attr("type");
      py::dict callable_type_dict;
      callable_type_dict["__call__"] = tat_m.attr("navigator");
      py::list base_types;
      base_types.append(py::type::of(tat_m));
      tat_m.attr("__class__") = py_type("CallableModule", py::tuple(base_types), callable_type_dict);
   }
} // namespace TAT
