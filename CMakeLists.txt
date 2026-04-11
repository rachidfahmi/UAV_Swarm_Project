cmake_minimum_required(VERSION 3.1)
project(UAVSearchPath)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories($ENV{HOME}/cadmium_v2/include)
include_directories($ENV{HOME}/cadmium_v2/json/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/main/include)

add_executable(UAVSearch main/main.cpp)
