cmake_minimum_required(VERSION 3.17)
project(cnn)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
#find_library(OpenCV)
find_package(OpenCV)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(cnn  main.cpp )

target_link_libraries(cnn ${OpenCV_LIBS})
