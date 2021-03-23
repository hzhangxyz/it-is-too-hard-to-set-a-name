cmake_minimum_required(VERSION 3.13)

# linux一般没问题, clang, gcc当然是都没问题的 我只使用过openmpi
# blas, lapack和mkl都没有问题
# windows的情况如下
# vs 2019 可以找到intel mpi, 使用clang和msvc都是可以的, 但是mkl需要手动设置MKLROOT的环境变量
# clion 无法找到intel mpi, 其他与vs 2019一致
# 自己的pc上linux中无法使用intel的编译器进行编译, 似乎与版本有关系, 在学校超算上配合合适的gcc版本是可以编译的
# windows中的intel情况懒得测试了

# 设置TAT library
add_library(TAT INTERFACE)

# 寻找TAT header path
find_path(TAT_INCLUDE_PATH NAMES "TAT/TAT.hpp" HINTS ${PROJECT_SOURCE_DIR}/include/ ${CMAKE_SOURCE_DIR}/include/ REQUIRED)
message("-- TAT headers found at ${TAT_INCLUDE_PATH}")
target_include_directories(TAT INTERFACE ${TAT_INCLUDE_PATH})

# 设置为c++17, 大多数超算上目前都有支持c++17的编译器, 故如此, c++20的话部分不支持, 所以本库也不使用
target_compile_features(TAT INTERFACE cxx_std_20)

# 常设置的参数有
# CMAKE_BUILD_TYPE, CMAKE_CXX_FLAGS
# TAT_USE_MPI, TAT_PYTHON_MODULE, TAT_FORCE_VERSION
# TAT_MATH_LIBRARIES, BLA_VENDOR, BLA_STATIC
# PYBIND11_PYTHON_VERSION, PYTHON_EXECUTABLE
option(TAT_USE_MPI "Use mpi for TAT" ON)
set(TAT_PYTHON_MODULE TAT CACHE STRING "Set python binding module name")
set(TAT_FORCE_VERSION 0.2.0 CACHE STRING "Force set TAT version")

# 下面几个宏仅仅用于记录编译信息
target_compile_definitions(TAT INTERFACE TAT_VERSION="${TAT_FORCE_VERSION}")
target_compile_definitions(TAT INTERFACE TAT_BUILD_TYPE="$<IF:$<STREQUAL:${CMAKE_BUILD_TYPE},>,Default,${CMAKE_BUILD_TYPE}>")
target_compile_definitions(TAT INTERFACE TAT_COMPILER_INFORMATION="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} on ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

# msvc必须加个utf8的参数
target_compile_options(TAT INTERFACE "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# 尝试启用mpi, 如果无法启用, 也没关系
if(CMAKE_CXX_COMPILER MATCHES "mpi")
   message("-- Using mpi compiler directly")
   target_compile_definitions(TAT INTERFACE TAT_USE_MPI)
   if(NOT TAT_USE_MPI)
      message("-- Ignore TAT_USE_MPI=OFF")
   endif()
else()
   if(TAT_USE_MPI)
         find_package(MPI QUIET)
         if(MPI_FOUND)
            target_include_directories(TAT INTERFACE ${MPI_INCLUDE_PATH})
            target_link_libraries(TAT INTERFACE ${MPI_LIBRARIES})
            message("-- Using mpi by linking mpi libraries")
            target_compile_definitions(TAT INTERFACE TAT_USE_MPI)
         else()
            message("-- No mpi support since mpi not found")
         endif()
   else()
      message("-- Disable mpi support since TAT_USE_MPI=OFF")
   endif()
endif()

# 链接blas和lapack, 尽量使用静态链接, 如果是emscripten则链接emlapack, 需要自行放入emscripten目录下
if(DEFINED TAT_MATH_LIBRARIES)
   message("-- Use customed math libraries")
   target_link_libraries(TAT INTERFACE ${TAT_MATH_LIBRARIES})
else()
   if(EMSCRIPTEN)
      message("-- Use emscripten blas and lapack")
      target_link_libraries(TAT INTERFACE ${CMAKE_SOURCE_DIR}/emscripten/liblapack.a)
      target_link_libraries(TAT INTERFACE ${CMAKE_SOURCE_DIR}/emscripten/libblas.a)
      target_link_libraries(TAT INTERFACE ${CMAKE_SOURCE_DIR}/emscripten/libf2c.a)
   else()
      find_package(BLAS REQUIRED)
      find_package(LAPACK REQUIRED)
      target_link_libraries(TAT INTERFACE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
      # 检查是否使用了mkl
      if(BLAS_LIBRARIES MATCHES mkl)
         message("-- Using mkl")
         target_compile_definitions(TAT INTERFACE TAT_USE_MKL_TRANSPOSE)
         target_compile_definitions(TAT INTERFACE TAT_USE_MKL_GEMM_BATCH)
      else()
         message("-- Not using mkl")
      endif()
   endif()
endif()

# 增加python binding目标
set(TAT_BUILD_PYTAT OFF)
if(NOT EMSCRIPTEN)
   find_package(pybind11 QUIET)
   if(pybind11_FOUND)
      message("-- Enable python(system)")
      set(TAT_BUILD_PYTAT ON)
   elseif(EXISTS ${PROJECT_SOURCE_DIR}/pybind11)
      add_subdirectory(pybind11)
      message("-- Enable python(local)")
      set(TAT_BUILD_PYTAT ON)
   else()
      message("-- Disable python since pybind11 not found, try install pybind11 or put it into TAT directory")
   endif()
   if(NOT EXISTS ${PROJECT_SOURCE_DIR}/PyTAT/PyTAT.cpp)
      set(TAT_BUILD_PYTAT OFF)
   endif()
endif()
if(TAT_BUILD_PYTAT)
   file(GLOB PYTAT_SRC ${PROJECT_SOURCE_DIR}/PyTAT/generated_code/*.cpp ${PROJECT_SOURCE_DIR}/PyTAT/*.cpp)
   pybind11_add_module(PyTAT ${PYTAT_SRC} EXCLUDE_FROM_ALL)
   target_link_libraries(PyTAT PRIVATE TAT)
   set_target_properties(PyTAT PROPERTIES OUTPUT_NAME ${TAT_PYTHON_MODULE})
   target_compile_definitions(PyTAT PRIVATE TAT_PYTHON_MODULE=${TAT_PYTHON_MODULE})
endif()
