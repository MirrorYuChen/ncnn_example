cmake_minimum_required(VERSION 3.1)

project(face)

find_package(OpenCV)
message("Opencv Version: ${OPENCV_VERSION}" )

file(GLOB SRC_FILES src/*.cpp src/*h)
file(GLOB LIBS lib/*.a)

set(INC_DIRS
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -std=c++11 -fopenmp")
include_directories(${INC_DIRS})
add_executable(main ${SRC_FILES})
target_link_libraries(main ${OpenCV_LIBS} ${LIBS} ${Vulkan_LIBRARY})