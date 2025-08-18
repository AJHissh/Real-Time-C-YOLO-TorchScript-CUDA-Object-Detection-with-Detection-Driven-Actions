cmake_minimum_required(VERSION 3.15)
project(abCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WIN32_LEAN_AND_MEAN 1)
add_definitions(-DUNICODE -D_UNICODE)

set(WINDOWS_KIT_DIR "C:/Program Files (x86)/Windows Kits/10")
set(WINDOWS_KIT_INCLUDE_DIR "${WINDOWS_KIT_DIR}/Include/10.0.26100.0")

set(DIRECTX_INCLUDE_DIRS 
    "${WINDOWS_KIT_INCLUDE_DIR}/um"
    "${WINDOWS_KIT_INCLUDE_DIR}/shared"
    "${WINDOWS_KIT_INCLUDE_DIR}/winrt"
)

set(NVTX3_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include/nvtx3")
if(EXISTS "${NVTX3_DIR}/nvtx3.hpp")
    message(STATUS "Found NVTX3 at: ${NVTX3_DIR}")
    add_library(nvtx3 INTERFACE IMPORTED)
    set_target_properties(nvtx3 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${NVTX3_DIR}"
    )
    set(USE_NVTX ON)
else()
    message(WARNING "NVTX3 not found at: ${NVTX3_DIR}")
    set(USE_NVTX OFF)
endif()


set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
find_package(CUDAToolkit REQUIRED)


set(CAFFE2_STATIC_LINK_CUDA ON)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "Suppress warnings")
option(USE_NVTX "Enable NVTX support" OFF) 

find_package(Torch REQUIRED PATHS "C:/libtorch")

if(TARGET torch::nvtoolsext)
    set_target_properties(torch::nvtoolsext PROPERTIES 
        IMPORTED_CONFIGURATIONS ""
        IMPORTED_LOCATION ""
        INTERFACE_LINK_LIBRARIES ""
    )
endif()

find_package(OpenCV REQUIRED)


add_executable(ab main.cpp dxgi_capture.cpp dxgi_capture.h)

target_include_directories(ab PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${DIRECTX_INCLUDE_DIRS}
    ${NVTX3_DIR}
)

target_link_libraries(ab PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    d3d11.lib
    dxgi.lib
    dxguid.lib
    gdiplus.lib
    gdi32.lib
    user32.lib
    CUDA::cudart
    CUDA::cublas
)

if(USE_NVTX)
    message(STATUS "NVTX3 support enabled")
else()
    message(STATUS "NVTX3 support disabled")
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Torch CUDA available: ${TORCH_CUDA_AVAILABLE}")
message(STATUS "Windows SDK version: 10.0.26100.0")
message(STATUS "DirectX includes: ${DIRECTX_INCLUDE_DIRS}")
message(STATUS "Build configuration complete")
