cmake_minimum_required(VERSION 3.18)  # Better to use newer CMake for modern CUDA
project(assignments CUDA C)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(hw2_ex1
        src/hw2_ex1.cu
)
add_executable(hw2_ex2
        src/hw2_ex2.cu
)
set_target_properties(hw2_ex1 hw2_ex2 PROPERTIES
        CUDA_ARCHITECTURES "86"
)