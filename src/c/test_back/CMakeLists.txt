cmake_minimum_required(VERSION 3.22)

add_executable(test_back test.cpp)

target_link_libraries(test_back PRIVATE HydraCudaStatic)
target_include_directories(test_back PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}../Cuda)
target_compile_features(test_back PRIVATE cxx_std_17)