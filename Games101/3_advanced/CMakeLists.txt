cmake_minimum_required(VERSION 3.10)
project(Rasterizer)

set(OpenCV_DIR C:/opencv/opencv/build/x64/vc16/lib/)
find_package(OpenCV REQUIRED)
set(EIGEN_DIR C:/eigen-3.4.0/)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include ${EIGEN_DIR})

add_executable(Rasterizer main.cpp rasterizer.hpp rasterizer.cpp global.hpp Triangle.hpp Triangle.cpp)
target_link_libraries(Rasterizer ${OpenCV_LIBRARIES})
