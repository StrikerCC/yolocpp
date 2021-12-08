cmake_minimum_required(VERSION 3.20)
project(yolocpp)

set(CMAKE_CXX_STANDARD 11)

################################## opencv ##################################
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

################################## boost ##################################
#find_package(Boost REQUIRED COMPONENTS python_scripts)
#include_directories(${Boost_INCLUDE_DIRS})
#message(${Boost_INCLUDE_DIRS})
#message("${Boost_LIBRARIES}")
#message("${Boost_LIBS}")

################################## python ##################################
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
#include_directories(${Python3_INCLUDE_DIRS})
include_directories(/home/cheng/anaconda3/envs/yolov5/include/python3.8
        /home/cheng/anaconda3/envs/yolov5/lib/python3.8)
#link_libraries(/home/cheng/anaconda3/envs/yolov5/lib)

################################## qt5 ##################################
#find_package(Qt5Core REQUIRED)
##include_directories(${})

################################## exe ##################################
include_directories(./det)
add_executable(${PROJECT_NAME} main.cpp
        det/Detection.cpp det/Detection.h
        det/PyClass.cpp det/PyClass.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
#target_link_libraries(${PROJECT_NAME} libpython3.8.so)
target_link_libraries(${PROJECT_NAME} /home/cheng/anaconda3/envs/yolov5/lib/libpython3.8.so)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
#target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
