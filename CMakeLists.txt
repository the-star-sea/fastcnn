cmake_minimum_required(VERSION 3.17)
project(cnn)

set(CMAKE_CXX_STANDARD 14)

#find_library(OpenCV)
find_package(OpenCV)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(cnn main.cpp face_binary_cls.cpp)

target_link_libraries(cnn ${OpenCV_LIBS})
