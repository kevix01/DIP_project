cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(D3D LANGUAGES C CXX VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Cython related dependencies
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${NumPy_INCLUDE_DIRS})

option(BUILD_WITH_CUDA "Build the library with CUDA support" ON)

if(BUILD_WITH_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")
    add_compile_definitions(BUILD_WITH_CUDA)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(CMAKE_CUDA_FLAGS_DEBUG "-G -g") # disable CUDA optimization when debug
    endif()
endif()
if(BUILD_WITH_CUDA)
    if(NOT TARGET CUDA::nvToolsExt)
        message(STATUS "CUDA::nvToolsExt not found, using NVTX3 shim")

        if(NOT TARGET CUDA::nvtx3)
            add_library(CUDA::nvtx3 INTERFACE IMPORTED)
            target_include_directories(CUDA::nvtx3 SYSTEM INTERFACE "${CUDAToolkit_INCLUDE_DIRS}")
            target_link_libraries(CUDA::nvtx3 INTERFACE ${CMAKE_DL_LIBS})
        endif()

        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
        target_compile_definitions(CUDA::nvToolsExt INTERFACE TORCH_CUDA_USE_NVTX3)
        target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
    endif()

    if(NOT DEFINED LIBNVTOOLSEXT OR LIBNVTOOLSEXT MATCHES "NOTFOUND")
        find_library(LIBDL dl)
        set(LIBNVTOOLSEXT ${LIBDL} CACHE FILEPATH "Compatibility shim for legacy nvToolsExt linkage" FORCE)
        message(STATUS "Setting LIBNVTOOLSEXT compatibility shim to: ${LIBNVTOOLSEXT}")
    endif()
endif()

if (BUILD_WITH_PYTORCH)
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

    if (TARGET torch_cpu)
        # tweak torch compiling flags, see https://github.com/pytorch/pytorch/pull/36021
        set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")

        # https://github.com/pytorch/pytorch/issues/38122
        find_library(TORCH_PYTHON torch_python)
        list(APPEND TORCH_LIBRARIES ${TORCH_PYTHON})

        if (BUILD_WITH_CUDA)
            set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")

            # tweak torch dependencies, see https://github.com/pytorch/pytorch/issues/33928
            get_target_property(TORCH_INTERFACE_LIB torch_cuda INTERFACE_LINK_LIBRARIES)
            string(REPLACE "/usr/local/cuda" ${CUDA_TOOLKIT_ROOT_DIR} TORCH_INTERFACE_LIB "${TORCH_INTERFACE_LIB}")
            set_target_properties(torch_cuda PROPERTIES INTERFACE_LINK_LIBRARIES "${TORCH_INTERFACE_LIB}")
        endif(BUILD_WITH_CUDA)
    else ()
        if (BUILD_WITH_CUDA)
            # this case is for pytorch 1.4
            get_target_property(TORCH_INTERFACE_LIB torch INTERFACE_LINK_LIBRARIES)
            string(REPLACE "/usr/local/cuda" ${CUDA_TOOLKIT_ROOT_DIR} TORCH_INTERFACE_LIB "${TORCH_INTERFACE_LIB}")
            set_target_properties(torch PROPERTIES INTERFACE_LINK_LIBRARIES "${TORCH_INTERFACE_LIB}")
        endif(BUILD_WITH_CUDA)
    endif ()
endif (BUILD_WITH_PYTORCH)

# include thirdparty libraries
include_directories(thirdparty)

add_subdirectory(d3d)
