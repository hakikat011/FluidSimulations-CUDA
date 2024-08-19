# File: src/cuda_integration/CMakeLists.txt

cmake_minimum_required(VERSION 3.10)

project(cuda_integration LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Include Houdini and CUDA directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${HOUDINI_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

# Define source files
set(SOURCES
    SOP_CudaIntegration.cpp
    kernels/advection.cu
)

# Create a shared library for the Houdini SOP node
add_library(cuda_integration SHARED ${SOURCES})
target_link_libraries(cuda_integration ${CUDA_LIBRARIES})
