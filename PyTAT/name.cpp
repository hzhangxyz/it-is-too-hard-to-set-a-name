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
   void set_name(py::module_& tat_m) {
#ifndef TAT_USE_SIMPLE_NAME
      py::class_<DefaultName>(tat_m, "Name", "Name used in edge of tensor, which is just a string but stored by identical integer")
            .def(py::self < py::self)
            .def(py::self > py::self)
            .def(py::self <= py::self)
            .def(py::self >= py::self)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::init<fastname_dataset_t::fast_name_id_t>(), py::arg("id"), "Name with specified id directly")
            .def_readonly("id", &DefaultName::id)
            .def("__hash__",
                 [](const DefaultName& name) {
                    return py::hash(py::cast(name.id));
                 })
            .def_static(
                  "load",
                  [](const py::bytes& bytes) {
                     return load_fastname_dataset(std::string(bytes));
                  })
            .def_static(
                  "dump",
                  []() {
                     return py::bytes(dump_fastname_dataset());
                  })
            .def_property_readonly(
                  "name",
                  [](const DefaultName& name) {
                     return static_cast<const std::string&>(name);
                  })
            .def(implicit_init<DefaultName, const char*>(), py::arg("name"), "Name with specified name")
            .def("__repr__",
                 [](const DefaultName& name) {
                    return "Name[" + static_cast<const std::string&>(name) + "]";
                 })
            .def("__str__", [](const DefaultName& name) {
               return static_cast<const std::string&>(name);
            });
#endif
   }
} // namespace TAT
