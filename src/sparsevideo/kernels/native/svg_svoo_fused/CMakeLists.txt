cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18 AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures for SparseVideo native kernels")
endif()

project(_kernels LANGUAGES CUDA CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import importlib.util; spec=importlib.util.find_spec('flashinfer'); print(list(spec.submodule_search_locations)[0] if spec and spec.submodule_search_locations else '')"
  OUTPUT_VARIABLE FLASHINFER_PACKAGE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT FLASHINFER_PACKAGE_DIR)
  message(FATAL_ERROR "flashinfer-python is required to build SparseVideo native fused kernels")
endif()

set(FLASHINFER_INCLUDE_DIR "${FLASHINFER_PACKAGE_DIR}/data/include")
set(CUTLASS_INCLUDE_DIR "${FLASHINFER_PACKAGE_DIR}/data/cutlass/include")

foreach(REQUIRED_INCLUDE_DIR IN LISTS FLASHINFER_INCLUDE_DIR CUTLASS_INCLUDE_DIR)
  if(NOT EXISTS "${REQUIRED_INCLUDE_DIR}")
    message(FATAL_ERROR "Required include directory does not exist: ${REQUIRED_INCLUDE_DIR}")
  endif()
endforeach()

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
if(NOT TORCH_PYTHON_LIBRARY)
  find_library(TORCH_PYTHON_LIBRARY torch_python)
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
  OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

file(GLOB PYTORCH_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cu")

add_library(_kernels MODULE ${PYTORCH_SOURCES})
set_target_properties(_kernels PROPERTIES PREFIX "" SUFFIX "${PYTHON_EXTENSION_SUFFIX}")

target_include_directories(
  _kernels
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${FLASHINFER_INCLUDE_DIR}"
    "${CUTLASS_INCLUDE_DIR}"
)

target_compile_options(
  _kernels
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      -gencode=arch=compute_80,code=sm_80
      --expt-extended-lambda
      --expt-relaxed-constexpr
      --use_fast_math
      --disable-warnings
    >
    $<$<COMPILE_LANGUAGE:CXX>:-w>
)

target_link_libraries(_kernels PRIVATE ${TORCH_LIBRARIES} Python::Python ${TORCH_PYTHON_LIBRARY})
