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
    add_sycl_to_target(TARGET GenVector inc/SYCLMath/GenVector/MathUtil.h)
  endif()

  if (oneapi)
    #set_target_properties(GenVector PROPERTIES LINKER_LANGUAGE CXX)
    target_compile_definitions(GenVector INTERFACE CL_TARGET_OPENCL_VERSION=300)
    target_compile_definitions(GenVector PUBLIC ROOT_MATH_SYCL)
    target_compile_options(GenVector INTERFACE -O2)
    target_include_directories(GenVector INTERFACE ${OpenSYCL_INCLUDE_DIRS})
    target_link_libraries(GenVector INTERFACE OpenSYCL::hipSYCL-rt)
    add_sycl_to_target(TARGET GenVector inc/SYCLMath/GenVector/MathUtil.h)
  endif()
endif()

add_library(GenVectorX SHARED
    inc/SYCLMath/AxisAngle.h
    inc/SYCLMath/Boost.h
    inc/SYCLMath/BoostX.h
    inc/SYCLMath/BoostY.h
    inc/SYCLMath/BoostZ.h
    inc/SYCLMath/Cartesian2D.h
    inc/SYCLMath/Cartesian3D.h
    inc/SYCLMath/Cylindrical3D.h
    inc/SYCLMath/CylindricalEta3D.h
    inc/SYCLMath/DisplacementVector2D.h
    inc/SYCLMath/DisplacementVector3D.h
    inc/SYCLMath/EulerAngles.h
    inc/SYCLMath/GenVector/3DConversions.h
    inc/SYCLMath/GenVector/3DDistances.h
    inc/SYCLMath/GenVector/AxisAnglefwd.h
    inc/SYCLMath/GenVector/AxisAngle.h
    inc/SYCLMath/GenVector/BitReproducible.h
    inc/SYCLMath/GenVector/Boostfwd.h
    inc/SYCLMath/GenVector/Boost.h
    inc/SYCLMath/GenVector/BoostXfwd.h
    inc/SYCLMath/GenVector/BoostX.h
    inc/SYCLMath/GenVector/BoostYfwd.h
    inc/SYCLMath/GenVector/BoostY.h
    inc/SYCLMath/GenVector/BoostZfwd.h
    inc/SYCLMath/GenVector/BoostZ.h
    inc/SYCLMath/GenVector/Cartesian2Dfwd.h
    inc/SYCLMath/GenVector/Cartesian2D.h
    inc/SYCLMath/GenVector/Cartesian3Dfwd.h
    inc/SYCLMath/GenVector/Cartesian3D.h
    inc/SYCLMath/GenVector/CoordinateSystemTags.h
    inc/SYCLMath/GenVector/Cylindrical3Dfwd.h
    inc/SYCLMath/GenVector/Cylindrical3D.h
    inc/SYCLMath/GenVector/CylindricalEta3Dfwd.h
    inc/SYCLMath/GenVector/CylindricalEta3D.h
    inc/SYCLMath/GenVector/DisplacementVector2Dfwd.h
    inc/SYCLMath/GenVector/DisplacementVector2D.h
    inc/SYCLMath/GenVector/DisplacementVector3Dfwd.h
    inc/SYCLMath/GenVector/DisplacementVector3D.h
    inc/SYCLMath/GenVector/eta.h
    inc/SYCLMath/GenVector/etaMax.h
    inc/SYCLMath/GenVector/EulerAnglesfwd.h
    inc/SYCLMath/GenVector/EulerAngles.h
    inc/SYCLMath/GenVector/GenVector_exception.h
    inc/SYCLMath/GenVector/GenVectorIO.h
    inc/SYCLMath/GenVector/LorentzRotationfwd.h
    inc/SYCLMath/GenVector/LorentzRotation.h
    inc/SYCLMath/GenVector/LorentzVectorfwd.h
    inc/SYCLMath/GenVector/LorentzVector.h
    inc/SYCLMath/GenVector/MathUtil.h
    inc/SYCLMath/GenVector/Plane3D.h
    inc/SYCLMath/GenVector/Polar2Dfwd.h
    inc/SYCLMath/GenVector/Polar2D.h
    inc/SYCLMath/GenVector/Polar3Dfwd.h
    inc/SYCLMath/GenVector/Polar3D.h
    inc/SYCLMath/GenVector/PositionVector2Dfwd.h
    inc/SYCLMath/GenVector/PositionVector2D.h
    inc/SYCLMath/GenVector/PositionVector3Dfwd.h
    inc/SYCLMath/GenVector/PositionVector3D.h
    inc/SYCLMath/GenVector/PtEtaPhiE4Dfwd.h
    inc/SYCLMath/GenVector/PtEtaPhiE4D.h
    inc/SYCLMath/GenVector/PtEtaPhiM4Dfwd.h
    inc/SYCLMath/GenVector/PtEtaPhiM4D.h
    inc/SYCLMath/GenVector/PxPyPzE4Dfwd.h
    inc/SYCLMath/GenVector/PxPyPzE4D.h
    inc/SYCLMath/GenVector/PxPyPzM4Dfwd.h
    inc/SYCLMath/GenVector/PxPyPzM4D.h
    inc/SYCLMath/GenVector/Quaternionfwd.h
    inc/SYCLMath/GenVector/Quaternion.h
    inc/SYCLMath/GenVector/Rotation3Dfwd.h
    inc/SYCLMath/GenVector/Rotation3D.h
    inc/SYCLMath/GenVector/RotationXfwd.h
    inc/SYCLMath/GenVector/RotationX.h
    inc/SYCLMath/GenVector/RotationYfwd.h
    inc/SYCLMath/GenVector/RotationY.h
    inc/SYCLMath/GenVector/RotationZfwd.h
    inc/SYCLMath/GenVector/RotationZ.h
    inc/SYCLMath/GenVector/RotationZYXfwd.h
    inc/SYCLMath/GenVector/RotationZYX.h
    inc/SYCLMath/GenVector/Transform3D.h
    inc/SYCLMath/GenVector/Translation3D.h
    inc/SYCLMath/GenVector/VectorUtil.h
    inc/SYCLMath/LorentzRotation.h
    inc/SYCLMath/LorentzVector.h
    inc/SYCLMath/Plane3D.h
    inc/SYCLMath/Point2Dfwd.h
    inc/SYCLMath/Point2D.h
    inc/SYCLMath/Point3Dfwd.h
    inc/SYCLMath/Point3D.h
    inc/SYCLMath/Polar2D.h
    inc/SYCLMath/Polar3D.h
    inc/SYCLMath/PositionVector2D.h
    inc/SYCLMath/PositionVector3D.h
    inc/SYCLMath/PtEtaPhiE4D.h
    inc/SYCLMath/PtEtaPhiM4D.h
    inc/SYCLMath/PxPyPzE4D.h
    inc/SYCLMath/PxPyPzM4D.h
    inc/SYCLMath/Quaternion.h
    inc/SYCLMath/Rotation3D.h
    inc/SYCLMath/RotationX.h
    inc/SYCLMath/RotationY.h
    inc/SYCLMath/RotationZ.h
    inc/SYCLMath/RotationZYX.h
    inc/SYCLMath/Transform3D.h
    inc/SYCLMath/Translation3D.h
    inc/SYCLMath/Vector2Dfwd.h
    inc/SYCLMath/Vector2D.h
    inc/SYCLMath/Vector3Dfwd.h
    inc/SYCLMath/Vector3D.h
    inc/SYCLMath/Vector4Dfwd.h
    inc/SYCLMath/Vector4D.h
    inc/SYCLMath/VectorUtil.h
)

set_target_properties(GenVectorX PROPERTIES LINKER_LANGUAGE CXX)

if (opensycl)
  #set_target_properties(GenVector PROPERTIES LINKER_LANGUAGE CXX)
  #target_compile_definitions(GenVectorX INTERFACE CL_TARGET_OPENCL_VERSION=300)
  target_compile_definitions(GenVectorX PUBLIC ROOT_MATH_SYCL)
  #target_compile_options(GenVectorX INTERFACE -O2)
  #target_include_directories(GenVectorX INTERFACE ${OpenSYCL_INCLUDE_DIRS})
  #target_link_libraries(GenVectorX INTERFACE OpenSYCL::hipSYCL-rt)
  #add_sycl_to_target(TARGET GenVectorX inc/SYCLMath/GenVector /MathUtil.h)
  add_sycl_to_root_target(TARGET GenVectorX inc/SYCLMath/GenVector/MathUtil.h)
endif()

install(TARGETS GenVectorX
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if (nocompile)
  ROOT_GENERATE_DICTIONARY(G__GenVector32
      SYCLMath/Point2D.h
      SYCLMath/Point3D.h
      SYCLMath/Vector2D.h
      SYCLMath/Vector3D.h
      SYCLMath/Vector4D.h
    MODULE
      GenVector
    MULTIDICT
    LINKDEF
      SYCLMath/LinkDef_GenVector32.h
    OPTIONS
      -writeEmptyRootPCM
    DEPENDENCIES
      Core
      MathCore
  )

  ROOT_ADD_TEST_SUBDIRECTORY(test)

  ROOT_INSTALL_HEADERS()
endif()