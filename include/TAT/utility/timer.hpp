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
#include <vector>

namespace TAT {
   /**
    * \defgroup Timer
    * @{
    */
#ifdef TAT_USE_TIMER
   using time_point = std::chrono::high_resolution_clock::time_point;
   using time_duration = std::chrono::high_resolution_clock::duration;
   using time_pair = std::tuple<time_point, time_duration>;

   /**
    * 获取当前时间点
    */
   inline auto get_current_time() {
      return std::chrono::high_resolution_clock::now();
   }
   /**
    * 将`std::chrono::high_resolution_clock::duration`转化为秒数
    */
   inline auto count_to_second(const time_duration& count) {
      return std::chrono::duration<double>(count).count();
   }

   /// \private
   struct timer_stack_t {
      std::stack<time_pair, std::vector<time_pair>> stack;
      timer_stack_t() {
         stack.push({get_current_time(), time_duration::zero()});
      }
      ~timer_stack_t() {
         const auto& [start_point, children_time] = stack.top();
         auto program_total_time = get_current_time() - start_point;
         auto program_misc_time = program_total_time - children_time;
         TAT_log(("total : " + std::to_string(count_to_second(program_total_time)) + ", " + std::to_string(count_to_second(program_misc_time)))
                       .c_str());
      }
   };
   /**
    * 统计各个函数时间用的栈
    */
   inline auto timer_stack = timer_stack_t();

   /**
    * 计时器类型
    *
    * 使用timer_stack对各个关注的函数进行计时，将统计调用其他函数的时间和自身的时间
    *
    * 如果希望增加自定义的计时器, 在全局空间定义`timer some_function_guard("some function_name");`,
    * 在计时处添加`auto timer_guard = some_function_guard();`即可, 计时器会统计自构建到析构间的时间
    *
    * \see timer_stack
    */
   struct timer {
      std::string timer_name;
      /**
       * 计时器统计的全部时间, 将在计时器销毁时打印出来
       */
      time_duration timer_total_count;
      /**
       * 计时器统计的自身时间, 将在计时器销毁时打印出来
       */
      time_duration timer_self_count;

      timer(const char* name) : timer_name(name), timer_total_count(time_duration::zero()), timer_self_count(time_duration::zero()) {}

      ~timer() {
         if (timer_total_count.count() != 0) {
            auto self_count_in_second = std::chrono::duration<float>(timer_self_count).count();
            auto total_count_in_second = std::chrono::duration<float>(timer_total_count).count();
            TAT_log((timer_name + " : " + std::to_string(total_count_in_second) + ", " + std::to_string(self_count_in_second)).c_str());
         }
      }

      struct timer_guard {
         timer* owner;

         timer_guard(timer* owner) : owner(owner) {
            timer_stack.stack.push({get_current_time(), time_duration::zero()});
         }

         ~timer_guard() {
            const auto& [start_point, children_time] = timer_stack.stack.top();
            auto this_guard_time = get_current_time() - start_point;
            owner->timer_total_count += this_guard_time;
            owner->timer_self_count += this_guard_time - children_time;
            timer_stack.stack.pop();
            std::get<1>(timer_stack.stack.top()) += this_guard_time;
         }
      };

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
   /**@}*/
} // namespace TAT
#endif
