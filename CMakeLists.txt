if(CMAKE_TOOLCHAIN_FILE)
    set(LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to")
    # get absolute path, but get_filename_component ABSOLUTE only refer with source dir, so find_file here :(
    get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
    find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
    message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

cmake_minimum_required(VERSION 2.8.12)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

project(ncnn)

option(NCNN_OPENMP "openmp support" ON)
option(NCNN_STDIO "load model from external file" ON)
option(NCNN_STRING "plain and verbose string" ON)
option(NCNN_INSTALL_SDK "install ncnn library and headers" ON)
option(NCNN_BENCHMARK "print benchmark information for every layer" OFF)
option(NCNN_PIXEL "convert and resize from/to image pixel" ON)
option(NCNN_PIXEL_ROTATE "rotate image pixel orientation" ON)
option(NCNN_CMAKE_VERBOSE "print verbose cmake messages" OFF)
option(NCNN_REQUANT "auto merge int8 quant and dequant" OFF)
option(NCNN_DISABLE_PIC "disable position-independent code" OFF)
option(NCNN_BUILD_BENCHMARK "build benchmark" ON)
option(NCNN_DISABLE_RTTI "disable rtti" ON)

##############################################

# set cmake default folder name
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

add_subdirectory(src)
if(NCNN_BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()
