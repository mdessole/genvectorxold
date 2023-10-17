project(genvectorx)

cmake_minimum_required(VERSION 3.9)

if (oneapi)
   SET(_sycl_search_dirs ${SYCL_ROOT_DIR} /usr/lib /usr/local/lib /opt/intel/oneapi/compiler/latest/linux)
   find_program(SYCL_COMPILER
                NAMES icpx dpcpp clang++
                HINTS ${_sycl_search_dirs}
                PATH_SUFFIXES bin)
   find_path(SYCL_INCLUDE_DIR
             NAMES sycl/sycl.hpp
             HINTS ${_sycl_search_dirs}
             PATH_SUFFIXES include)

   if (SYCL_COMPILER)
     set(sycl ON)

     function(add_sycl_to_root_target)
       CMAKE_PARSE_ARGUMENTS(ARG "" "TARGET" "SOURCES" ${ARGN})
       get_target_property(_library_name ${ARG_TARGET} OUTPUT_NAME)
       get_target_property(_deps ${ARG_TARGET} LINK_LIBRARIES)

       message(STATUS "Output library name: ${_library_name}")
       target_include_directories(${ARG_TARGET} PUBLIC ${SYCL_INCLUDE_DIR} ${SYCL_INCLUDE_DIR}/sycl)

       # Get include directories for the SYCL target
       get_target_property(_inc_dirs ${ARG_TARGET} INCLUDE_DIRECTORIES)
       if (_inc_dirs)
         foreach(dir ${_inc_dirs})
           list(APPEND _inc_list -I${dir})
         endforeach()
         list(REMOVE_DUPLICATES _inc_list)
       endif()
       #list(APPEND _inc_list -I${SYCL_INCLUDE_DIR}/sycl)
       #list(APPEND _inc_list -I${SYCL_INCLUDE_DIR}/sycl/CL)
       message(STATUS "_inc_list: ${_inc_list}")
       message(STATUS "SYCL_INCLUDE_DIR: ${SYCL_INCLUDE_DIR}")

       # Compile the sycl source files with the found sycl compiler
       set(SYCL_FLAGS "-fsycl -fsycl-unnamed-lambda -fsycl-device-only -Xclang ") #-fsycl-targets=nvptx64-nvidia-cuda
       message(STATUS "SYCL flags: ${SYCL_FLAGS}")
       separate_arguments(SYCL_FLAGS NATIVE_COMMAND ${SYCL_FLAGS})
       foreach(src ${ARG_SOURCES})
         set(_output_path ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_library_name}.dir/${src}${CMAKE_CXX_OUTPUT_EXTENSION})
         add_custom_command(OUTPUT ${_output_path}
                            COMMAND ${SYCL_COMPILER} ${SYCL_FLAGS}  -c
                            ${_inc_list}
                            -o ${_output_path}
                            ${CMAKE_CURRENT_SOURCE_DIR}/${src}
                            DEPENDS ${_deps} ${ARG_TARGET}
                            COMMENT "Building SYCL object ${_output_path}"
                            MAIN_DEPENDENCY ${src})
        
       endforeach()
     endfunction()

     message(STATUS "Found Intel OneAPI SYCL: ${SYCL_INCLUDE_DIR}  (modify with: SYCL_ROOT_DIR)")
     message(STATUS "Using SYCL Host Compiler: ${SYCL_COMPILER}")
else()
    if(fail-on-missing)
       message(FATAL_ERROR "OpenAPI SYCL library not found")
     else()
       message(STATUS "OpenAPI SYCL library not found")
       set(sycl OFF CACHE BOOL "Disabled because no SYCL implementation is not found" FORCE)
     endif()
   endif()
 endif()

if (opensycl)
  if (oneapi)
    message(WARNING "Disable OneAPI to load OpenSYCL")
    set(sycl OFF CACHE BOOL "Disabled because OpenSYCL is enabled" FORCE)
  else()
    find_package(OpenSYCL)
    if (OpenSYCL_FOUND)
      set(sycl ON)
      function(add_sycl_to_root_target)
        CMAKE_PARSE_ARGUMENTS(ARG "" "TARGET" "SOURCES" ${ARGN})
        add_sycl_to_target(TARGET ${ARG_TARGET}  SOURCES ${ARG_SOURCES})
        target_include_directories(${ARG_TARGET} INTERFACE ${OpenSYCL_INCLUDE_DIRS})
        target_link_libraries(${ARG_TARGET} INTERFACE OpenSYCL::hipSYCL-rt)
      endfunction()
      message(STATUS "OpenSYCL sycl enabled")
    else()
      if(fail-on-missing)
        message(FATAL_ERROR "Open SYCL library not found")
      else()
        message(STATUS "Open SYCL library not found")
        set(sycl OFF CACHE BOOL "Disabled because no SYCL implementation is not found" FORCE)
      endif()
    endif()
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT ACPP_DEBUG_LEVEL)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(ACPP_DEBUG_LEVEL 3 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  else()
    set(ACPP_DEBUG_LEVEL 2 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  endif()
endif()

add_definitions(-DHIPSYCL_DEBUG_LEVEL=${ACPP_DEBUG_LEVEL})

if (nocompile)
  ROOT_LINKER_LIBRARY(GenVector
      src/3DConversions.cxx
      src/3DDistances.cxx
      src/AxisAngle.cxx
      src/AxisAngleXother.cxx
      src/BitReproducible.cxx
      src/Boost.cxx
      src/BoostX.cxx
      src/BoostY.cxx
      src/BoostZ.cxx
      src/EulerAngles.cxx
      src/LorentzRotation.cxx
      src/Quaternion.cxx
      src/QuaternionXaxial.cxx
      src/Rotation3D.cxx
      src/Rotation3DxAxial.cxx
      src/RotationZYX.cxx
      src/VectorUtil.cxx
    DEPENDENCIES
      Core
      MathCore
  )

  if (opensycl)
    #set_target_properties(GenVector PROPERTIES LINKER_LANGUAGE CXX)
    target_compile_definitions(GenVector INTERFACE CL_TARGET_OPENCL_VERSION=300)
    target_compile_definitions(GenVector PUBLIC ROOT_MATH_SYCL)
    target_compile_options(GenVector INTERFACE -O2)
    target_include_directories(GenVector INTERFACE ${OpenSYCL_INCLUDE_DIRS})
    target_link_libraries(GenVector INTERFACE OpenSYCL::hipSYCL-rt)
    add_sycl_to_target(TARGET GenVector inc/Math/GenVector/MathUtil.h)
  endif()

  if (oneapi)
    #set_target_properties(GenVector PROPERTIES LINKER_LANGUAGE CXX)
    target_compile_definitions(GenVector INTERFACE CL_TARGET_OPENCL_VERSION=300)
    target_compile_definitions(GenVector PUBLIC ROOT_MATH_SYCL)
    target_compile_options(GenVector INTERFACE -O2)
    target_include_directories(GenVector INTERFACE ${OpenSYCL_INCLUDE_DIRS})
    target_link_libraries(GenVector INTERFACE OpenSYCL::hipSYCL-rt)
    add_sycl_to_target(TARGET GenVector inc/Math/GenVector/MathUtil.h)
  endif()
endif()

add_library(GenVectorX SHARED
    inc/Math/AxisAngle.h
    inc/Math/Boost.h
    inc/Math/BoostX.h
    inc/Math/BoostY.h
    inc/Math/BoostZ.h
    inc/Math/Cartesian2D.h
    inc/Math/Cartesian3D.h
    inc/Math/Cylindrical3D.h
    inc/Math/CylindricalEta3D.h
    inc/Math/DisplacementVector2D.h
    inc/Math/DisplacementVector3D.h
    inc/Math/EulerAngles.h
    inc/Math/GenVector/3DConversions.h
    inc/Math/GenVector/3DDistances.h
    inc/Math/GenVector/AxisAnglefwd.h
    inc/Math/GenVector/AxisAngle.h
    inc/Math/GenVector/BitReproducible.h
    inc/Math/GenVector/Boostfwd.h
    inc/Math/GenVector/Boost.h
    inc/Math/GenVector/BoostXfwd.h
    inc/Math/GenVector/BoostX.h
    inc/Math/GenVector/BoostYfwd.h
    inc/Math/GenVector/BoostY.h
    inc/Math/GenVector/BoostZfwd.h
    inc/Math/GenVector/BoostZ.h
    inc/Math/GenVector/Cartesian2Dfwd.h
    inc/Math/GenVector/Cartesian2D.h
    inc/Math/GenVector/Cartesian3Dfwd.h
    inc/Math/GenVector/Cartesian3D.h
    inc/Math/GenVector/CoordinateSystemTags.h
    inc/Math/GenVector/Cylindrical3Dfwd.h
    inc/Math/GenVector/Cylindrical3D.h
    inc/Math/GenVector/CylindricalEta3Dfwd.h
    inc/Math/GenVector/CylindricalEta3D.h
    inc/Math/GenVector/DisplacementVector2Dfwd.h
    inc/Math/GenVector/DisplacementVector2D.h
    inc/Math/GenVector/DisplacementVector3Dfwd.h
    inc/Math/GenVector/DisplacementVector3D.h
    inc/Math/GenVector/eta.h
    inc/Math/GenVector/etaMax.h
    inc/Math/GenVector/EulerAnglesfwd.h
    inc/Math/GenVector/EulerAngles.h
    inc/Math/GenVector/GenVector_exception.h
    inc/Math/GenVector/GenVectorIO.h
    inc/Math/GenVector/LorentzRotationfwd.h
    inc/Math/GenVector/LorentzRotation.h
    inc/Math/GenVector/LorentzVectorfwd.h
    inc/Math/GenVector/LorentzVector.h
    inc/Math/GenVector/MathUtil.h
    inc/Math/GenVector/Plane3D.h
    inc/Math/GenVector/Polar2Dfwd.h
    inc/Math/GenVector/Polar2D.h
    inc/Math/GenVector/Polar3Dfwd.h
    inc/Math/GenVector/Polar3D.h
    inc/Math/GenVector/PositionVector2Dfwd.h
    inc/Math/GenVector/PositionVector2D.h
    inc/Math/GenVector/PositionVector3Dfwd.h
    inc/Math/GenVector/PositionVector3D.h
    inc/Math/GenVector/PtEtaPhiE4Dfwd.h
    inc/Math/GenVector/PtEtaPhiE4D.h
    inc/Math/GenVector/PtEtaPhiM4Dfwd.h
    inc/Math/GenVector/PtEtaPhiM4D.h
    inc/Math/GenVector/PxPyPzE4Dfwd.h
    inc/Math/GenVector/PxPyPzE4D.h
    inc/Math/GenVector/PxPyPzM4Dfwd.h
    inc/Math/GenVector/PxPyPzM4D.h
    inc/Math/GenVector/Quaternionfwd.h
    inc/Math/GenVector/Quaternion.h
    inc/Math/GenVector/Rotation3Dfwd.h
    inc/Math/GenVector/Rotation3D.h
    inc/Math/GenVector/RotationXfwd.h
    inc/Math/GenVector/RotationX.h
    inc/Math/GenVector/RotationYfwd.h
    inc/Math/GenVector/RotationY.h
    inc/Math/GenVector/RotationZfwd.h
    inc/Math/GenVector/RotationZ.h
    inc/Math/GenVector/RotationZYXfwd.h
    inc/Math/GenVector/RotationZYX.h
    inc/Math/GenVector/Transform3D.h
    inc/Math/GenVector/Translation3D.h
    inc/Math/GenVector/VectorUtil.h
    inc/Math/LorentzRotation.h
    inc/Math/LorentzVector.h
    inc/Math/Plane3D.h
    inc/Math/Point2Dfwd.h
    inc/Math/Point2D.h
    inc/Math/Point3Dfwd.h
    inc/Math/Point3D.h
    inc/Math/Polar2D.h
    inc/Math/Polar3D.h
    inc/Math/PositionVector2D.h
    inc/Math/PositionVector3D.h
    inc/Math/PtEtaPhiE4D.h
    inc/Math/PtEtaPhiM4D.h
    inc/Math/PxPyPzE4D.h
    inc/Math/PxPyPzM4D.h
    inc/Math/Quaternion.h
    inc/Math/Rotation3D.h
    inc/Math/RotationX.h
    inc/Math/RotationY.h
    inc/Math/RotationZ.h
    inc/Math/RotationZYX.h
    inc/Math/Transform3D.h
    inc/Math/Translation3D.h
    inc/Math/Vector2Dfwd.h
    inc/Math/Vector2D.h
    inc/Math/Vector3Dfwd.h
    inc/Math/Vector3D.h
    inc/Math/Vector4Dfwd.h
    inc/Math/Vector4D.h
    inc/Math/VectorUtil.h
)

set_target_properties(GenVectorX PROPERTIES LINKER_LANGUAGE CXX)

if (opensycl)
  #set_target_properties(GenVector PROPERTIES LINKER_LANGUAGE CXX)
  #target_compile_definitions(GenVectorX INTERFACE CL_TARGET_OPENCL_VERSION=300)
  target_compile_definitions(GenVectorX PUBLIC ROOT_MATH_SYCL)
  #target_compile_options(GenVectorX INTERFACE -O2)
  #target_include_directories(GenVectorX INTERFACE ${OpenSYCL_INCLUDE_DIRS})
  #target_link_libraries(GenVectorX INTERFACE OpenSYCL::hipSYCL-rt)
  #add_sycl_to_target(TARGET GenVectorX inc/Math/GenVector /MathUtil.h)
  add_sycl_to_root_target(TARGET GenVectorX inc/Math/GenVector/MathUtil.h)
endif()

install(TARGETS GenVectorX
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if (nocompile)
  ROOT_GENERATE_DICTIONARY(G__GenVector32
      Math/Point2D.h
      Math/Point3D.h
      Math/Vector2D.h
      Math/Vector3D.h
      Math/Vector4D.h
    MODULE
      GenVector
    MULTIDICT
    LINKDEF
      Math/LinkDef_GenVector32.h
    OPTIONS
      -writeEmptyRootPCM
    DEPENDENCIES
      Core
      MathCore
  )

  ROOT_ADD_TEST_SUBDIRECTORY(test)

  ROOT_INSTALL_HEADERS()
endif()