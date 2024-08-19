cmake_minimum_required(VERSION 3.10)
project(HoudiniExtensionProject)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Include directories
include_directories(src/common)
include_directories(src/amr)
include_directories(src/vdb_management)
include_directories(src/cuda_integration)