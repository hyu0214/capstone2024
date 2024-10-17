cmake_minimum_required(VERSION 3.10)  # CMake 최소 버전을 3.10으로 변경
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "build type" FORCE)
endif()

project(spherical-sfm)

set(CMAKE_CXX_STANDARD 14)  # C++ 표준을 C++14로 변경
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Ceres, OpenCV, Eigen, Threads 설정
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Threads REQUIRED)

# sphericalsfm 라이브러리 정의
include_directories(include)
add_library(sphericalsfm
    include/sphericalsfm/ray.h
    include/sphericalsfm/so3.h
    src/so3.cpp
    include/sphericalsfm/sparse.hpp
    include/sphericalsfm/sfm.h
    src/sfm.cpp
    include/sphericalsfm/sfm_types.h
    src/sfm_types.cpp
    include/sphericalsfm/estimator.h
    include/sphericalsfm/msac.h
    include/sphericalsfm/preemptive_ransac.h
    include/sphericalsfm/spherical_estimator.h
    src/spherical_estimator.cpp
    include/sphericalsfm/triangulation_estimator.h
    src/triangulation_estimator.cpp
    include/sphericalsfm/spherical_fast_estimator.h
    src/spherical_fast_estimator.cpp
    include/sphericalsfm/plane_estimator.h
    src/plane_estimator.cpp
    include/sphericalsfm/spherical_utils.h
    src/spherical_utils.cpp
    include/sphericalsfm/rotation_averaging.h
    src/rotation_averaging.cpp
)

# 라이브러리 링크
target_link_libraries(sphericalsfm ${CERES_LIBRARIES} ${OpenCV_LIBS} Threads::Threads)

# gflags 설정
find_package(gflags REQUIRED)

# 실행 파일 정의 및 링크
add_executable(run_spherical_sfm_realtime
    spherical_sfm_tools.h
    spherical_sfm_tools.cpp
    spherical_sfm_io.h
    spherical_sfm_io.cpp
    run_spherical_sfm_realtime.cpp
)

target_link_libraries(run_spherical_sfm_realtime sphericalsfm ${OpenCV_LIBS} gflags)
