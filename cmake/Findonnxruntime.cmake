#[=======================================================================[.rst:
Findonnxruntime
----------------

Locates ONNX Runtime and exposes it as the imported target
``onnxruntime::onnxruntime``.

This module does NOT download or install anything. It only *finds* an
already-provided ONNX Runtime, from either of:

  1. A CONFIG package (e.g. the MSYS2 / vcpkg / conda distribution, which
     ships ``onnxruntimeConfig.cmake``). Preferred when available.
  2. A plain include/lib layout, such as the official prebuilt archives
     from https://github.com/microsoft/onnxruntime/releases. Point CMake
     at it via ``-DCMAKE_PREFIX_PATH=<extracted-dir>`` or
     ``-DONNXRUNTIME_ROOT=<extracted-dir>``.

Result Variables
^^^^^^^^^^^^^^^^^
``onnxruntime_FOUND``
  True if ONNX Runtime was found.

Imported Targets
^^^^^^^^^^^^^^^^^
``onnxruntime::onnxruntime``
  The ONNX Runtime shared library, with include directories attached.
#]=======================================================================]

# 1. Prefer an upstream CONFIG package if one is installed (MSYS2/vcpkg/conda).
find_package(onnxruntime CONFIG QUIET)
if(onnxruntime_FOUND)
  return()
endif()

# 2. Fall back to a manual search across the plain include/lib layout used by
#    the official prebuilt archives. Layouts differ slightly between
#    distributions, hence the extra PATH_SUFFIXES.
find_path(ORT_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_ROOT}"
  PATH_SUFFIXES include onnxruntime include/onnxruntime
)
find_library(ORT_LIBRARY
  NAMES onnxruntime
  HINTS "${ONNXRUNTIME_ROOT}"
  PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(onnxruntime
  REQUIRED_VARS ORT_LIBRARY ORT_INCLUDE_DIR
)

if(onnxruntime_FOUND AND NOT TARGET onnxruntime::onnxruntime)
  add_library(onnxruntime::onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
  )
  if(WIN32)
    # On Windows, IMPORTED_LOCATION is the .dll; linking needs the .lib next
    # to it (same basename, same directory for the official prebuilt zips).
    get_filename_component(ORT_LIB_DIR "${ORT_LIBRARY}" DIRECTORY)
    get_filename_component(ORT_LIB_NAME "${ORT_LIBRARY}" NAME_WE)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
      IMPORTED_IMPLIB "${ORT_LIBRARY}"
      IMPORTED_LOCATION "${ORT_LIB_DIR}/${ORT_LIB_NAME}.dll"
    )
  else()
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
      IMPORTED_LOCATION "${ORT_LIBRARY}"
    )
  endif()
endif()

mark_as_advanced(ORT_INCLUDE_DIR ORT_LIBRARY)
