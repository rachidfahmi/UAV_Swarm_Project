cmake_minimum_required(VERSION 3.10)
project(UAVSearchPath)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(NOT DEFINED ENV{CADMIUM})
    message(FATAL_ERROR "CADMIUM environment variable is not set. Please run: echo $CADMIUM")
endif()

get_filename_component(CADMIUM_ROOT "$ENV{CADMIUM}/.." ABSOLUTE)

include_directories($ENV{CADMIUM})
include_directories(${CADMIUM_ROOT}/json/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/main/include)

add_executable(UAVSearch main/main.cpp)