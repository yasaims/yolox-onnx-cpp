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
     at it via ``-DONNXRUNTIME_ROOT=<extracted-dir>``.

Note: the official prebuilt archive's own ``onnxruntimeConfig.cmake`` (route
1) is broken as of 1.20.1 -- its ``onnxruntimeTargets.cmake`` hardcodes a
``lib64`` path even though the archive ships the library under ``lib``, and
it raises a ``message(FATAL_ERROR)`` that ``find_package(... QUIET)`` cannot
suppress. Passing ``ONNXRUNTIME_ROOT`` therefore skips route 1 entirely and
goes straight to the manual search below, so pointing at a prebuilt archive
never risks tripping over its own broken CONFIG package.

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
#    Skipped when ONNXRUNTIME_ROOT is set explicitly: that's the caller
#    saying "use the plain include/lib layout at this path" (route 2 below),
#    and the official prebuilt archive's own CONFIG package is broken (see
#    docstring above), so we must not let find_package() go looking for it.
if(NOT ONNXRUNTIME_ROOT)
  find_package(onnxruntime CONFIG QUIET)
  if(onnxruntime_FOUND)
    return()
  endif()
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
