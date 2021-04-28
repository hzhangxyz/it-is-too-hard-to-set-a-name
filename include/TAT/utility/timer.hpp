/**
 * \file timer.hpp
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
#ifndef TAT_TIMER_HPP
#define TAT_TIMER_HPP

#include <chrono>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "../TAT.hpp"

namespace TAT {
#ifdef TAT_USE_TIMER
   /**
    * Timer type
    *
    * Use single timer stack to count time cost different function concerned, including itself and total.
    *
    * To use it, define a global timer `timer xxx_guard("xxx")`, and call `auto timer_guard = xxx_guard();`
    * at the begin of the function
    */
   struct timer {
    private:
      using time_point = std::chrono::high_resolution_clock::time_point;
      using time_duration = std::chrono::high_resolution_clock::duration;
      using time_pair = std::pair<time_point, time_duration>;

    private:
      static auto get_current_time() {
         return std::chrono::high_resolution_clock::now();
      }
      static auto duration_to_second(const time_duration& count) {
         return std::chrono::duration<double>(count).count();
      }

    private:
      struct timer_stack_t {
         std::stack<time_pair, std::vector<time_pair>> stack;
         timer_stack_t() {
            stack.push({get_current_time(), time_duration::zero()});
         }
         ~timer_stack_t() {
            const auto& [start_point, children_time] = stack.top();
            auto program_total_time = get_current_time() - start_point;
            auto program_misc_time = program_total_time - children_time;
            detail::log(
                  ("total : " + std::to_string(duration_to_second(program_total_time)) + ", " + std::to_string(duration_to_second(program_misc_time)))
                        .c_str());
         }
      };
      inline static auto timer_stack = timer_stack_t();

      std::string timer_name;
      /**
       * count total time, will be printed when destructing the timer
       */
      time_duration timer_total_count;
      /**
       * count self time, will be printed when destructing the timer
       */
      time_duration timer_self_count;

    public:
      /**
       * Create a timer with given name
       */
      timer(const char* name) : timer_name(name), timer_total_count(time_duration::zero()), timer_self_count(time_duration::zero()) {}

      ~timer() {
         if (timer_total_count.count() != 0) {
            auto self_count_in_second = std::chrono::duration<float>(timer_self_count).count();
            auto total_count_in_second = std::chrono::duration<float>(timer_total_count).count();
            detail::log((timer_name + " : " + std::to_string(total_count_in_second) + ", " + std::to_string(self_count_in_second)).c_str());
         }
      }

      struct timer_guard {
       private:
         timer* owner;

       public:
         timer_guard(timer* owner) : owner(owner) {
            timer_stack.stack.push({get_current_time(), time_duration::zero()});
         }

         ~timer_guard() {
            const auto& [start_point, children_time] = timer_stack.stack.top();
            auto this_guard_time = get_current_time() - start_point;
            owner->timer_total_count += this_guard_time;
            owner->timer_self_count += this_guard_time - children_time;
            timer_stack.stack.pop();
            timer_stack.stack.top().second += this_guard_time;
         }
      };

      /**
       * Start to count time for the timer in current scope
       */
      auto operator()() {
         return timer_guard(this);
      }
   };
#else
   struct timer {
      struct timer_guard {};
      auto operator()() {
         return timer_guard{};
      }
      timer(const char*) {}
   };
#endif
} // namespace TAT
#endif
