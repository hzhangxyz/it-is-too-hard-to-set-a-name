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

#include <iostream>
#include <lazy.hpp>

#include "run_test.hpp"

void run_test() {
   auto a = lazy::Root(1);
   auto b = lazy::Root(2);
   std::cout << a->get() << "\n";
   std::cout << b->get() << "\n";
   auto c = lazy::Path(
         [](int a, int b) {
            return a + b;
         },
         a,
         b);
   auto d = lazy::Node(
         [](int c, int a) {
            return c * a;
         },
         c,
         a);
   std::cout << d->get() << "\n";
   a->set(233);
   std::cout << d->get() << "\n";
   auto snap = lazy::default_graph.dump();
   b->set(666);
   std::cout << d->get() << "\n";
   lazy::default_graph.load(snap);
   std::cout << d->get() << "\n";
}
