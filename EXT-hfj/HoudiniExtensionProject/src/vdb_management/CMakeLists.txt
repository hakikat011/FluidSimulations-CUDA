# File: src/vdb_management/CMakeLists.txt

cmake_minimum_required(VERSION 3.10)

project(vdb_management LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Include Houdini and OpenVDB directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${HOUDINI_INCLUDE_DIR})
include_directories(${OPENVDB_INCLUDE_DIRS})

# Link with OpenVDB libraries
find_package(OpenVDB REQUIRED)

# Define source files
set(SOURCES
    SOP_VDBManagement.cpp
)

# Create a shared library for the Houdini SOP node
add_library(vdb_management SHARED ${SOURCES})
target_link_libraries(vdb_management ${HOUDINI_LIBRARIES} ${OPENVDB_LIBRARIES})
