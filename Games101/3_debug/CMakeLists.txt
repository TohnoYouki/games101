cmake_minimum_required(VERSION 3.10)
project(Rasterizer)

set(OpenCV_DIR C:/opencv/opencv/build/x64/vc16/lib/)
find_package(OpenCV REQUIRED)
set(EIGEN_DIR C:/eigen-3.4.0/)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include ${EIGEN_DIR})

add_executable(Rasterizer main.cpp Rasterizer.hpp Rasterizer.cpp global.hpp JobThread.hpp 
                         JobThread.cpp OBJ_Loader.h Shader.hpp Shader.cpp SwapChain.hpp SwapChain.cpp 
                         Scene.hpp Scene.cpp Texture.hpp Texture.cpp Triangle.hpp Triangle.cpp Timer.hpp)
target_link_libraries(Rasterizer ${OpenCV_LIBRARIES})
