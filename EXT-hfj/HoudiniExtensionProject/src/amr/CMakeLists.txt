# File: src/amr/CMakeLists.txt

cmake_minimum_required(VERSION 3.10)

project(amr LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Include Houdini directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${HOUDINI_INCLUDE_DIR})

# Define source files
set(SOURCES
    SOP_AMR.cpp
)

# Create a shared library for the Houdini SOP node
add_library(amr SHARED ${SOURCES})
target_link_libraries(amr ${HOUDINI_LIBRARIES})
