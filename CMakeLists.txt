cmake_minimum_required(VERSION 3.6)
project(Spline)

set(CMAKE_CXX_STANDARD 14)

include_directories(include lib/eigen)

add_library(${PROJECT_NAME} SHARED
        src/Spline.cpp)