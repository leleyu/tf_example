cmake_minimum_required(VERSION 3.12)
project(tf_example)

set(CMAKE_CXX_STANDARD 14)
set(TF_HOME /Users/leleyu/workspace/github/tf_env)

include_directories(${TF_HOME}/include)
include_directories(${TF_HOME}/include/eigen3)
link_directories(${TF_HOME}/lib)


add_executable(example examples/example.cpp)
target_link_libraries(example tensorflow_cc tensorflow_framework)

add_executable(matrix examples/matrix.cpp)
target_link_libraries(matrix tensorflow_cc tensorflow_framework)

add_executable(tf_tensor examples/tf_tensor.cc)
target_link_libraries(tf_tensor tensorflow_cc tensorflow_framework)

