cmake_minimum_required(VERSION 3.17)
project(cnn)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -O3")#arm上不能使用avx

set(OpenCV_DIR "C:/OpenCV/build/install")#arm上注释本局,win10修改路径
find_package(OpenCV REQUIRED)
IF(OPENCV_FOUND)
    message("Found OpenCV!")
    include_directories(
            ${LOCAL_INCLUDE_DIR}
            ${OpenCV_INCLUDE_DIRS}
    )
ELSE()
    message(FATAL_ERROR "Missed OpenCV! CMake will exit")
ENDIF()
add_executable(cnn  main.cpp)

target_link_libraries(cnn ${OpenCV_LIBS})