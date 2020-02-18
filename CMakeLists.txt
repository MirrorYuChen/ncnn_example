cmake_minimum_required(VERSION 3.1)

project(face)

find_package(OpenCV)
message("Opencv Version: ${OPENCV_VERSION}" )

file(GLOB SRC_FILES
    src/*.cpp
    src/aligner/*cpp
    src/common/*.cpp
    src/tracker/*.cpp
    src/database/*.cpp
    src/database/stream/*.cpp
    src/detecter/*.cpp
    src/detecter/retinaface/*.cpp
    src/detecter/mtcnn/*.cpp
    src/detecter/centerface/*.cpp
    src/landmarker/*.cpp
    src/landmarker/zqlandmark/*.cpp
    src/recognizer/*.cpp
    src/recognizer/mobilefacenet/*.cpp
)
file(GLOB LIBS lib/*.a)

set(INC_DIRS
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/align
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/database
    ${PROJECT_SOURCE_DIR}/src/database/stream
    ${PROJECT_SOURCE_DIR}/src/detecter
    ${PROJECT_SOURCE_DIR}/src/detecert/retinaface
    ${PROJECT_SOURCE_DIR}/src/detecter/mtcnn
    ${PROJECT_SOURCE_DIR}/src/detecter/centerface
    ${PROJECT_SOURCE_DIR}/src/tracker
    ${PROJECT_SOURCE_DIR}/src/landmarker
    ${PROJECT_SOURCE_DIR}/src/landmarker/zqlandmark
    ${PROJECT_SOURCE_DIR}/src/recognizer
    ${PROJECT_SOURCE_DIR}/src/recognizer/mobilefacenet
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