cmake_minimum_required(VERSION 3.10)
project(HoudiniPlugins)

# Add plugin source files
add_library(HoudiniPlugin SHARED HoudiniPlugin.cpp)

# Link necessary libraries
target_link_libraries(HoudiniPlugin HoudiniExtensionProject)