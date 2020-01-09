cmake_minimum_required(VERSION 3.1)

project(face)

find_package(OpenCV)
message("Opencv Version: ${OPENCV_VERSION}" )

file(GLOB SRC_FILES
    src/*.cpp
    src/align/*cpp
    src/common/*.cpp
    src/detect/*.cpp
    src/track/*.cpp
    src/database/*.cpp
    src/detect/retinaface/*.cpp
    src/detect/mtcnn/*.cpp
    src/detect/centerface/*.cpp
    src/landmark/*.cpp
    src/landmark/zqlandmark/*.cpp
    src/recognize/*.cpp
    src/recognize/mobilefacenet/*.cpp
)
file(GLOB LIBS lib/*.a)

set(INC_DIRS
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/align
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/database
    ${PROJECT_SOURCE_DIR}/src/detect
    ${PROJECT_SOURCE_DIR}/src/detect/retinaface
    ${PROJECT_SOURCE_DIR}/src/detect/mtcnn
    ${PROJECT_SOURCE_DIR}/src/detect/centerface
    ${PROJECT_SOURCE_DIR}/src/track
    ${PROJECT_SOURCE_DIR}/src/landmark
    ${PROJECT_SOURCE_DIR}/src/landmark/zqlandmark
    ${PROJECT_SOURCE_DIR}/src/recognize
    ${PROJECT_SOURCE_DIR}/src/recognize/mobilefacenet
)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -std=c++11 -fopenmp")
include_directories(${INC_DIRS})
add_executable(main ${SRC_FILES})
target_link_libraries(main ${OpenCV_LIBS} ${LIBS})