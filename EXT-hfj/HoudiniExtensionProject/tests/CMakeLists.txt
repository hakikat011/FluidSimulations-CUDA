cmake_minimum_required(VERSION 3.10)
project(Tests)

# Add test files
add_executable(test_amr src/amr/tests/test_amr.cpp)
add_executable(test_vdb src/vdb_management/tests/test_vdb.cpp)
add_executable(test_advection src/cuda_integration/tests/test_advection.cpp)
add_executable(integration_tests tests/integration_tests.cpp)

# Link necessary libraries
target_link_libraries(test_amr HoudiniExtensionProject)
target_link_libraries(test_vdb HoudiniExtensionProject)
target_link_libraries(test_advection HoudiniExtensionProject)
target_link_libraries(integration_tests HoudiniExtensionProject)