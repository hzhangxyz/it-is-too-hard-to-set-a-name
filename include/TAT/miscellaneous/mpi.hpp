/**
 * \file mpi.hpp
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
#ifndef TAT_MPI_HPP
#define TAT_MPI_HPP

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "../utility/timer.hpp"
#include "io.hpp"

#ifdef TAT_USE_MPI
// 不可以extern "C"，因为mpi.h发现不可以被暂时屏蔽的宏__cplusplus后申明一些cpp的函数
#include <mpi.h>
#endif

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace TAT {
   /**
    * \defgroup MPI
    * @{
    */

   /**
    * 对流进行包装, 包装后流只会根据创建时指定的有效性决定是否输出
    */
   struct mpi_one_output_stream {
      std::ostream& out;
      bool valid;
      std::ostringstream string;
      ~mpi_one_output_stream() {
         out << string.str() << std::flush;
      }
      mpi_one_output_stream(std::ostream& out, bool valid) : out(out), valid(valid) {}

      template<typename Type>
      mpi_one_output_stream& operator<<(const Type& value) & {
         if (valid) [[likely]] {
            string << value;
         }
         return *this;
      }

      template<typename Type>
      mpi_one_output_stream&& operator<<(const Type& value) && {
         if (valid) [[likely]] {
            string << value;
         }
         return std::move(*this);
      }
   };

   /**
    * 对流进行包装, 每次输出之前打印当前rank
    */
   struct mpi_rank_output_stream {
      std::ostream& out;
      std::ostringstream string;
      ~mpi_rank_output_stream() {
         out << string.str() << std::flush;
      }
      mpi_rank_output_stream(std::ostream& out, int rank) : out(out) {
         if (rank != -1) [[unlikely]] {
            string << "[rank " << rank << "] ";
         }
      }

      template<typename Type>
      mpi_rank_output_stream& operator<<(const Type& value) & {
         string << value;
         return *this;
      }

      template<typename Type>
      mpi_rank_output_stream&& operator<<(const Type& value) && {
         string << value;
         return std::move(*this);
      }
   };

   // TODO: 使用类似std::format一样的公用的序列化方式
   template<typename T>
   concept serializable = requires(std::ostream& o, std::istream& i, const T& u, T& v) {
      o < u;
      i > v;
   };

   inline timer mpi_send_guard("mpi_send");
   inline timer mpi_receive_guard("mpi_receive");
   inline timer mpi_broadcast_guard("mpi_broadcast");
   inline timer mpi_reduce_guard("mpi_reduce");

   /**
    * 一个mpi handler类型, 会在构造和析构时自动调用MPI_Init和MPI_Finalize, 且会获取Size和Rank信息, 同时提供只在某个rank下有效的输出流
    *
    * 创建多个mpi_t不会产生冲突
    */
   struct mpi_t {
      int size = 1;
      int rank = 0;
#ifdef TAT_USE_MPI
      static constexpr bool enabled = true;
#else
      static constexpr bool enabled = false;
#endif
#ifdef TAT_USE_MPI
      static bool initialized() noexcept {
         int result;
         MPI_Initialized(&result);
         return result;
      }
      static bool finalized() noexcept {
         int result;
         MPI_Finalized(&result);
         return result;
      }
      mpi_t() {
         if (!initialized()) [[unlikely]] {
            MPI_Init(nullptr, nullptr);
         }
         MPI_Comm_size(MPI_COMM_WORLD, &size);
         MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      }
      ~mpi_t() {
         if (!finalized()) [[unlikely]] {
            MPI_Finalize();
         }
      }

      static void barrier() {
         MPI_Barrier(MPI_COMM_WORLD);
      }

      static constexpr int mpi_tag = 0;

      template<serializable Type>
      static void send(const Type& value, const int destination) {
         auto timer_guard = mpi_send_guard();
         std::ostringstream stream;
         stream < value;
         auto data = stream.str(); // TODO: 也许可以不需复制, 但这个在mpi框架内可能不是很方便
         // TODO 是不是可以立即返回?
         MPI_Send(data.data(), data.length(), MPI_BYTE, destination, mpi_tag, MPI_COMM_WORLD);
      }

      // TODO: 异步的处理, 这个优先级很低, 也许以后将和gpu中做svd, gemm一起做成异步
      template<serializable Type>
      static Type receive(const int source) {
         auto timer_guard = mpi_receive_guard();
         auto status = MPI_Status();
         MPI_Probe(source, mpi_tag, MPI_COMM_WORLD, &status);
         int length;
         MPI_Get_count(&status, MPI_BYTE, &length);
         auto data = std::basic_string<char, std::char_traits<char>, no_initialize_allocator<char>>();
         data.resize(length);
         MPI_Recv(data.data(), length, MPI_BYTE, source, mpi_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         std::basic_istringstream<char, std::char_traits<char>, no_initialize_allocator<char>> stream(data);
         auto result = Type();
         stream > result;
         return result;
      }

      template<serializable Type>
      Type send_receive(const Type& value, const int source, const int destination) const {
         if (rank == source) [[unlikely]] {
            send(value, destination);
         }
         if (rank == destination) [[unlikely]] {
            return receive<Type>(source);
         }
         return Type();
      }

      template<serializable Type>
      Type broadcast(const Type& value, const int root) const {
         auto timer_guard = mpi_broadcast_guard();
         if (size == 1) [[unlikely]] {
            return value;
         }
         if (0 > root || root >= size) [[unlikely]] {
            TAT_error("Invalid root rank when mpi broadcast");
         }
         const auto this_fake_rank = (size + rank - root) % size;
         Type result;
         // get from father
         if (this_fake_rank != 0) [[likely]] {
            const auto father_fake_rank = (this_fake_rank - 1) / 2;
            const auto father_real_rank = (father_fake_rank + root) % size;
            result = receive<Type>(father_real_rank);
         } else [[unlikely]] {
            // 自己就是root的话, 会复制一次张量
            result = value;
         }
         // send to son
         const auto left_son_fake_rank = this_fake_rank * 2 + 1;
         const auto right_son_fake_rank = this_fake_rank * 2 + 2;
         if (left_son_fake_rank < size) [[likely]] {
            const auto left_son_real_rank = (left_son_fake_rank + root) % size;
            send(result, left_son_real_rank);
         }
         if (right_son_fake_rank < size) [[likely]] {
            const auto right_son_real_rank = (right_son_fake_rank + root) % size;
            send(result, right_son_real_rank);
         }
         return result;
      }

      template<serializable Type, typename Func>
         requires requires(const Type& a, const Type& b, Func&& f) {
            { f(a, b) } -> std::same_as<Type>;
         }
      Type reduce(const Type& value, const int root, Func&& function) const {
         auto timer_guard = mpi_reduce_guard();
         if (size == 1) [[unlikely]] {
            return value;
         }
         if (0 > root || root >= size) [[unlikely]] {
            TAT_error("Invalid root rank when mpi reduce");
         }
         const auto this_fake_rank = (size + rank - root) % size;
         Type result;
         // get from son
         const auto left_son_fake_rank = this_fake_rank * 2 + 1;
         const auto right_son_fake_rank = this_fake_rank * 2 + 2;
         if (left_son_fake_rank < size) [[likely]] {
            const auto left_son_real_rank = (left_son_fake_rank + root) % size;
            result = function(value, receive<Type>(left_son_real_rank));
         }
         if (right_son_fake_rank < size) [[likely]] {
            const auto right_son_real_rank = (right_son_fake_rank + root) % size;
            // 如果左儿子不存在, 那么右儿子一定不存在, 所以不必判断result是否有效
            result = function(result, receive<Type>(right_son_real_rank));
         }
         // pass to father
         if (this_fake_rank != 0) [[likely]] {
            const auto father_fake_rank = (this_fake_rank - 1) / 2;
            const auto father_real_rank = (father_fake_rank + root) % size;
            if (left_son_fake_rank < size) [[likely]] {
               send(result, father_real_rank);
            } else [[unlikely]] {
               send(value, father_real_rank);
            }
         }
         return result;
         // 子叶为空tensor, 每个非子叶节点为reduce了所有的后代的结果
      }
#endif
      auto out_one(int rank_specified = 0) {
         return mpi_one_output_stream(std::cout, rank_specified == rank);
      }
      auto log_one(int rank_specified = 0) {
         return mpi_one_output_stream(std::clog, rank_specified == rank);
      }
      auto err_one(int rank_specified = 0) {
         return mpi_one_output_stream(std::cerr, rank_specified == rank);
      }

      auto out_rank() {
         return mpi_rank_output_stream(std::cout, size == 1 ? -1 : rank);
      }
      auto log_rank() {
         return mpi_rank_output_stream(std::clog, size == 1 ? -1 : rank);
      }
      auto err_rank() {
         return mpi_rank_output_stream(std::cerr, size == 1 ? -1 : rank);
      }
   };
   /**
    * \see mpi_t
    */
   inline mpi_t mpi;
   /**@}*/

   inline evil_t::evil_t() noexcept {
#ifdef _WIN32
      HANDLE output_handle = GetStdHandle(STD_OUTPUT_HANDLE);
      DWORD output_mode = 0;
      GetConsoleMode(output_handle, &output_mode);
      SetConsoleMode(output_handle, output_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
      HANDLE error_handle = GetStdHandle(STD_ERROR_HANDLE);
      DWORD error_mode = 0;
      GetConsoleMode(error_handle, &error_mode);
      SetConsoleMode(error_handle, error_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
   }
   inline evil_t::~evil_t() {
#ifndef NDEBUG
      mpi.log_one() << console_blue << "\n\nPremature optimization is the root of all evil!\n"
                    << console_origin << "                                       --- Donald Knuth\n\n\n";
#endif
   }

   inline void TAT_log(const char* message) {
      mpi.log_rank() << console_yellow << message << console_origin << '\n';
   }

   inline void TAT_warning(const char* message) {
      mpi.err_rank() << console_red << message << console_origin << '\n';
   }

   inline void TAT_error(const char* message) {
      throw std::runtime_error(message);
   }
} // namespace TAT
#endif
