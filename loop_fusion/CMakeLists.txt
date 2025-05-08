cmake_minimum_required(VERSION 3.16)
project(loop_fusion)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV)
find_package(Ceres REQUIRED)
find_package(Eigen3)
find_package(PCL 1.7 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    camera_models
    cv_bridge
    roslib
    message_filters
)

include_directories(SYSTEM include ${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

set(Torch_DIR "$ENV{HOME}/source/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${active_loop_INCLUDE_DIRS})

add_message_files(
  FILES
  StereoImage.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

link_directories(${PCL_LIBRARY_DIRS})

add_executable(loop_fusion_node
  src/pose_graph_node.cpp
  src/pose_graph.cpp
  src/keyframe.cpp
  src/utility/CameraPoseVisualization.cpp
  src/CNN/superpoint_common.cpp 
  src/CNN/superpoint_onnx.cpp
)

target_link_libraries(loop_fusion_node 
  ${catkin_LIBRARIES}  
  ${OpenCV_LIBS} 
  ${CERES_LIBRARIES}
  ${PCL_LIBRARIES}
  ${TORCH_LIBRARIES} 
  onnxruntime
  faiss
  dw
) 
