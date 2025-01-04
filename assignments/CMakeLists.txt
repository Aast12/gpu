cmake_minimum_required(VERSION 3.14)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75)  # Adjust for your GPU
project(assignments CUDA CXX)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES
        "*/src/*.cu"
        "*/src/*.cpp"
)
# Tell CMake to treat all source files as CUDA sources
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)

add_executable(hw2_ex1
        assignment2/src/hw2_ex1.cu
)
add_executable(hw2_ex2
        assignment2/src/hw2_ex2.cu
)
add_executable(hw4_ex3
        assignment4/src/hw4-ex3.cu
)

target_link_libraries(hw4_ex3 PRIVATE cublas
        cusparse)
set_target_properties(hw2_ex1 hw2_ex2 hw4_ex3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75"
)
