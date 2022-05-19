cmake_minimum_required(VERSION 3.1)
project(hw3)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# libigl
option(LIBIGL_OPENGL             "Build target igl::opengl"         ON)
option(LIBIGL_GLFW               "Build target igl::glfw"           ON)
option(LIBIGL_IMGUI              "Use ImGui"                        ON)
option(LIBIGL_USE_STATIC_LIBRARY "Build LibIGL as a static library" OFF)

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../libigl)
    message("libigl directory exists")
    set(FETCHCONTENT_SOURCE_DIR_LIBIGL "${CMAKE_CURRENT_SOURCE_DIR}/../libigl" CACHE PATH "Path to local libigl copy (to avoid re-cloning for each build directory)" FORCE)
else()
    message("libigl directory doesn't exist")
endif()
include(libigl)

# Build mode that compiles with optimizations but leaves in assertions.
set(CMAKE_CXX_FLAGS_RELWITHASSERT        "-O3" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_C_FLAGS_RELWITHASSERT          "-O3" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT    "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_RELWITHASSERT
    CMAKE_C_FLAGS_RELWITHASSERT
    CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT
    CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT
    CMAKE_MODULE_LINKER_FLAGS_RELWITHASSERT
    CMAKE_STATIC_LINKER_FLAGS_RELWITHASSERT
)

# Enable address sanitizer in debug mode.
set (CMAKE_C_FLAGS_DEBUG      "${CMAKE_C_FLAGS_DEBUG}      -fno-omit-frame-pointer -fsanitize=address")
set (CMAKE_CXX_FLAGS_DEBUG    "${CMAKE_CXX_FLAGS_DEBUG}    -fno-omit-frame-pointer -fsanitize=address")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to RelWithAssert")
    set(CMAKE_BUILD_TYPE "RelWithAssert")
endif()

# We need -fPIC when compiling our libraries and our dependencies for
# the python bindings (shared libraries) to link.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile flags (used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Make sure warnings/errors are still colorized when using Ninja for building.
add_definitions(-fdiagnostics-color=always)

# Add your project files
if (NOT SOLUTION)
    add_library(TopOpt Viewer.hh GridGeneration.hh ElasticityTensor.hh SymmetricMatrix.hh RegionLasso.cc RegionLasso.hh VisualizationGeometry.hh TopologyOptimizer.hh TopologyOptimizer.cc DensityField.cc)
else()
    add_library(TopOpt Viewer.hh GridGeneration.hh ElasticityTensor.hh SymmetricMatrix.hh RegionLasso.cc RegionLasso.hh VisualizationGeometry.hh TopologyOptimizer.hh TopologyOptimizer_solution.cc DensityField_solution.cc)
endif()
target_link_libraries(TopOpt PUBLIC igl::core igl::opengl igl::glfw igl::imgui)

find_package(CHOLMOD QUIET)
if (CHOLMOD_FOUND)
    message("Using CHOLMOD")
    target_compile_definitions(TopOpt PUBLIC HAS_CHOLMOD)
    target_link_libraries(TopOpt PUBLIC cholmod::cholmod)
else()
    message("CHOMOD not found; using Eigen's Sparse Cholesky")
endif()

target_include_directories(TopOpt PUBLIC SYSTEM ${CMAKE_SOURCE_DIR}/3rdParty/LBFGSpp/include)

add_executable(${PROJECT_NAME} main.cc)
target_link_libraries(${PROJECT_NAME} TopOpt)

add_executable(tests tests.cc)
target_link_libraries(tests TopOpt)
