cmake_minimum_required(VERSION 3.25)
project(CUDA LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Tell nvcc to ignore that MSVC is "too new"
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>)

add_executable(CUDA
        Hello.cu)


set_target_properties(CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
