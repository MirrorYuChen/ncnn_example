cmake_minimum_required(VERSION 3.1)

set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/common/common.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/face/face_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/face/aligner/aligner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/face/database/face_database.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/face/database/stream/file_stream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/face/detecter/detecter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/face/detecter/centerface/centerface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/face/detecter/mtcnn/mtcnn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/face/detecter/retinaface/retinaface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker/landmarker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker/zqlandmarker/zqlandmarker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/face/recognizer/recognizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/face/recognizer/mobilefacenet/mobilefacenet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/face/tracker/tracker.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/object/object_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/object/object/mobilenetssd/mobilenetssd.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/classifier/classifier_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/classifier/classifier/mobilenet/mobilenet.cpp

)

# list(APPEND SRCS ${LAYER_ARCH_SRC})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -fPIC -std=c++11 -fopenmp")
add_library(VisionEngine STATIC ${SRC_FILES})

find_package(OpenCV)
message("Opencv Version: ${OPENCV_VERSION}" )
file(GLOB LIBS ${CMAKE_CURRENT_SOURCE_DIR}/../lib/*.a)
target_link_libraries(VisionEngine PUBLIC ${LIBS} ${OpenCV_LIBS})

if(MIRROR_OPENMP)
find_package(OpenMP)
if(NOT TARGET OpenMP::OpenMP_CXX AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
    target_compile_options(VisionEngine PRIVATE ${OpenMP_CXX_FLAGS})
endif()
endif()

if(MIRROR_OPENMP AND OpenMP_CXX_FOUND)
    message("Building with OpenMP")
    target_link_libraries(VisionEngine PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(VisionEngine
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include/vision>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>

        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/align>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/common>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/database>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/detecter>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/detecter/centerface>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/landmarker/zqlandmarker>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/recognizer>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/recognizer/mobilefacenet>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/face/tracker>

        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/object>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/object/object>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/object/common>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/object/object/mobilenetssd>

        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/classifier>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/classifier/classifier>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/classifier/classifier/mobilenet>
)

if(MIRROR_INSTALL_SDK)
    install(TARGETS VisionEngine EXPORT VisionEngine ARCHIVE DESTINATION ${PROJECT_BINARY_DIR}/lib)
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/common/common.h
        ${CMAKE_CURRENT_SOURCE_DIR}/face/face_engine.h
        ${CMAKE_CURRENT_SOURCE_DIR}/object/object_engine.h
        ${CMAKE_CURRENT_SOURCE_DIR}/classifier/classifier_engine.h
        DESTINATION ${PROJECT_BINARY_DIR}/include
    )
endif()


if(MIRROR_BUILD_FACE)
    add_executable(face ${CMAKE_CURRENT_SOURCE_DIR}/face/test_face.cpp)
    target_link_libraries(face PUBLIC VisionEngine)
endif()

if(MIRROR_BUILD_CLASSIFIER)
    add_executable(classifier ${CMAKE_CURRENT_SOURCE_DIR}/classifier/test_classifier.cpp)
    target_link_libraries(classifier PUBLIC VisionEngine)
endif()

if(MIRROR_BUILD_OBJECT)
    add_executable(object ${CMAKE_CURRENT_SOURCE_DIR}/object/test_object.cpp)
    target_link_libraries(object PUBLIC VisionEngine)
endif()