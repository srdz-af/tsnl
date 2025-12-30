cmake_minimum_required(VERSION 3.15)
project(tsnl2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic)

find_package(Threads REQUIRED)
option(USE_BLAS "Link against BLAS for GEMV/GEMM" OFF)
if(USE_BLAS)
    find_package(BLAS)
endif()

set(SRC_FILES
    src/io.cpp
    src/csr.cpp
    src/rng.cpp
    src/threadpool.cpp
    src/sampler.cpp
    src/subgraph.cpp
    src/features.cpp
    src/encoder.cpp
    src/decoder.cpp
    src/loss.cpp
    src/optim.cpp
    src/metrics.cpp
    src/checkpoint.cpp
)

add_library(kgcore ${SRC_FILES})
target_include_directories(kgcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(kgcore PUBLIC Threads::Threads)
if(USE_BLAS AND BLAS_FOUND)
    target_link_libraries(kgcore PUBLIC BLAS::BLAS)
    target_compile_definitions(kgcore PUBLIC USE_BLAS)
endif()

add_executable(kg_train src/main_train.cpp)
target_link_libraries(kg_train PRIVATE kgcore)

add_executable(kg_infer src/main_infer.cpp)
target_link_libraries(kg_infer PRIVATE kgcore)

add_executable(kg_eval src/main_eval.cpp)
target_link_libraries(kg_eval PRIVATE kgcore)

add_executable(kg_build_reverse src/main_build_reverse.cpp)
target_link_libraries(kg_build_reverse PRIVATE kgcore)

enable_testing()
add_executable(small_sanity tests/small_sanity.cpp)
target_link_libraries(small_sanity PRIVATE kgcore)
add_test(NAME small_sanity COMMAND small_sanity)
