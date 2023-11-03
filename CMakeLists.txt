project(genvectorx )

cmake_minimum_required(VERSION 3.9)

find_package(ROOT REQUIRED COMPONENTS MathCore ROOTVecOps)
if (ROOT_FOUND)
  message(STATUS "ROOT library found")
  #message(STATUS "ROOT_INCLUDE_DIRS: ${ROOT_INCLUDE_DIRS}")
  #message(STATUS "ROOT_LIBRARIES: ${ROOT_LIBRARIES}")
  #message(STATUS "ROOT_CXX_FLAGS: ${ROOT_CXX_FLAGS}")
  #message(STATUS "ROOT_INCLUDE_DIRS: ${ROOT_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "ROOT library not found")
endif()

include_directories(${CMAKE_SOURCE_DIR}/inc)

string(TOUPPER "${CMAKE_BUILD_TYPE}" _BUILD_TYPE_UPPER)

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
    find_path(SYCL_LIB_DIR
             NAMES libsycl.so
             HINTS ${_sycl_search_dirs}
             PATH_SUFFIXES lib)

    

    if (syclcuda)
      set(_fsycl_targets "-fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64,spir64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_${CMAKE_CUDA_ARCHITECTURES} --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")
    else()
        set(_fsycl_targets "-fsycl-targets=spir64_x86_64,spir64")
    endif()

    set(SYCL_COMPILER_FLAGS "-fPIC -fsycl -fsycl-unnamed-lambda -sycl-std=2020 ${_fsycl_targets} ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_BUILD_TYPE_UPPER}}")
    message(STATUS "SYCL compiler flags: ${SYCL_COMPILER_FLAGS}")
    separate_arguments(SYCL_COMPILER_FLAGS NATIVE_COMMAND ${SYCL_COMPILER_FLAGS})

    if (SYCL_COMPILER)
      set(sycl ON)

    function(add_sycl_to_root_target_exe)
      CMAKE_PARSE_ARGUMENTS(ARG "" "TARGET" "SOURCES;HEADERS;DEPENDENCIES;INCLUDE_DIRECTORIES;COMPILE_DEFINITIONS" ${ARGN})

      
      # Get include directories for the SYCL target
      set(_inc_dirs ${ARG_INCLUDE_DIRECTORIES})
      list(APPEND _inc_dirs ${SYCL_INCLUDE_DIR})
      if (_inc_dirs)
        list(REMOVE_DUPLICATES _inc_dirs)
        list(TRANSFORM _inc_dirs PREPEND -I)
      endif()
      
      foreach(comp_def ${ARG_COMPILE_DEFINITIONS})
        list(APPEND _COMPILE_DEFINITIONS -D${comp_def})
      endforeach()

      # Compile the sycl source files with the found sycl compiler
      foreach(src ${ARG_SOURCES})
        set(_output_path ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${ARG_TARGET}.dir/${src}${CMAKE_CXX_OUTPUT_EXTENSION})
        list(APPEND _outputs ${_output_path})

        add_custom_command(OUTPUT ${_output_path}
                            COMMAND ${SYCL_COMPILER}
                                    ${SYCL_COMPILER_FLAGS} -c
                                    ${_inc_dirs}
                                    ${_COMPILE_DEFINITIONS}
                                    -o ${_output_path}
                                    ${CMAKE_CURRENT_SOURCE_DIR}/${src}
                            DEPENDS ${ARG_DEPENDENCIES}
                            COMMENT "Building SYCL object ${_output_path}"
                            MAIN_DEPENDENCY ${src}
                            )
      endforeach()

      set(SYCL_LINKER_FLAGS "-L${SYCL_LIB_DIR}  -Wl,-rpath,${SYCL_LIB_DIR}")
      foreach(lib ${ARG_DEPENDENCIES})
        list(APPEND _lib_dep_paths "$<TARGET_FILE:${lib}>")
        list(APPEND SYCL_LINKER_FLAGS ":$<TARGET_FILE_DIR:${lib}>")
      endforeach()
      string(REPLACE ";" "" SYCL_LINKER_FLAGS ${SYCL_LINKER_FLAGS})
      #message(STATUS "_lib_paths: ${_lib_paths}")
      #message(STATUS "__lib_paths: ${__lib_paths}")
      #set(SYCL_LINKER_FLAGS "-L${SYCL_LIB_DIR}  -Wl,-rpath,${SYCL_LIB_DIR}:$<TARGET_FILE_DIR:${ARG_TARGET}>")
      message(STATUS "SYCL linker flags: ${SYCL_LINKER_FLAGS}")
      separate_arguments(SYCL_LINKER_FLAGS NATIVE_COMMAND ${SYCL_LINKER_FLAGS})
      set_target_properties(${ARG_TARGET} PROPERTIES LINKER_LANGUAGE CXX)

      set_property(TARGET ${ARG_TARGET} PROPERTY SOURCES ${_outputs})  

      add_custom_command(TARGET ${ARG_TARGET}
                            COMMAND ${SYCL_COMPILER} 
                                    ${SYCL_COMPILER_FLAGS}
                                    ${SYCL_LINKER_FLAGS}
                                    ${_lib_dep_paths}
                                    ${_COMPILE_DEFINITIONS}
                                    -o ${ARG_TARGET} ${_outputs} 
                            DEPENDS ${ARG_DEPENDENCIES}
                            COMMENT "Building SYCL executable ${ARG_TARGET}"
                            MAIN_DEPENDENCY ${ARG_TARGET}
                            )
    
    endfunction()

    function(add_sycl_to_root_target)
      CMAKE_PARSE_ARGUMENTS(ARG "" "TARGET" "SOURCES;HEADERS;DEPENDENCIES;INCLUDE_DIRECTORIES;COMPILE_DEFINITIONS" ${ARGN})

      set(_library_name ${ARG_TARGET})

      # Get include directories for the SYCL target
      set(_inc_dirs ${ARG_INCLUDE_DIRECTORIES})
      if (_inc_dirs)
        list(REMOVE_DUPLICATES _inc_dirs)
        list(TRANSFORM _inc_dirs PREPEND -I)
      endif()
      list(APPEND _inc_dirs ${SYCL_INCLUDE_DIR})
      list(APPEND _inc_dirs ${SYCL_INCLUDE_DIR}/sycl)

      foreach(lib ${ARG_DEPENDENCIES})
        list(APPEND _lib_dep_paths "$<TARGET_FILE:${lib}>")
      endforeach()

      foreach(comp_def ${ARG_COMPILE_DEFINITIONS})
        list(APPEND _COMPILE_DEFINITIONS -D${comp_def})
      endforeach()

      # Compile the sycl source files with the found sycl compiler
      foreach(src ${ARG_SOURCES})
        set(_output_path ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${ARG_TARGET}.dir/${src}${CMAKE_CXX_OUTPUT_EXTENSION})
        list(APPEND _outputs ${_output_path})

        add_custom_command(OUTPUT ${_output_path}
                            COMMAND ${SYCL_COMPILER}  
                                    ${SYCL_COMPILER_FLAGS} -c
                                    ${_inc_dirs} 
                                    ${_COMPILE_DEFINITIONS}
                                    -o ${_output_path}
                                    ${CMAKE_CURRENT_SOURCE_DIR}/${src}
                            DEPENDS ${ARG_DEPENDENCIES}
                            COMMENT "Building SYCL object ${_output_path}"
                            MAIN_DEPENDENCY ${src}
                            )
      endforeach()

      set(SYCL_LINKER_FLAGS "-shared -Wl,-soname,$<TARGET_FILE:${_library_name}> -L${SYCL_LIB_DIR} ${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,${SYCL_LIB_DIR}:$<TARGET_FILE_DIR:${_library_name}>")
      separate_arguments(SYCL_LINKER_FLAGS NATIVE_COMMAND ${SYCL_LINKER_FLAGS})
      
      add_custom_command(OUTPUT ${ARG_TARGET}
                          COMMAND ${SYCL_COMPILER}
                                  ${SYCL_LINKER_FLAGS}
                                  ${_lib_dep_paths} ${_inc_dirs}
                                  ${_COMPILE_DEFINITIONS} 
                                  -o  $<TARGET_FILE:${_library_name}>
                                  ${_outputs}
                          DEPENDS ${ARG_DEPENDENCIES} ${_outputs} 
                          COMMENT "Linking shared library ${ARG_TARGET}"
                          MAIN_DEPENDENCY ${_outputs}
                          )

      set_property(TARGET ${ARG_TARGET} PROPERTY LINK_DEPENDS ${_outputs})
      set_property(TARGET ${ARG_TARGET} PROPERTY SOURCES ${_outputs})  
      set_target_properties(${ARG_TARGET} PROPERTIES LINKER_LANGUAGE CXX)
      target_include_directories(${ARG_TARGET} PUBLIC ${SYCL_INCLUDE_DIR} ${SYCL_INCLUDE_DIR}/sycl)
      target_link_directories(${ARG_TARGET} PUBLIC ${SYCL_LIB_DIR})
      target_link_libraries(${ARG_TARGET} INTERFACE sycl)
      target_compile_definitions(${ARG_TARGET} PUBLIC ${ARG_COMPILE_DEFINITIONS})

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
    #add_definitions(-DHIPSYCL_DEBUG_LEVEL=4)
    if (OpenSYCL_FOUND)
      set(sycl ON)
      function(add_sycl_to_root_target)
        CMAKE_PARSE_ARGUMENTS(ARG "" "TARGET" "SOURCES;HEADERS;DEPENDENCIES;INCLUDE_DIRECTORIES;COMPILE_DEFINITIONS" ${ARGN})
        target_compile_definitions(${ARG_TARGET} PUBLIC ${ARG_COMPILE_DEFINITIONS})
        target_link_libraries(${ARG_TARGET} PUBLIC ${ARG_DEPENDENCIES})
        target_include_directories(${ARG_TARGET} INTERFACE ${OpenSYCL_INCLUDE_DIRS})
        target_link_libraries(${ARG_TARGET} INTERFACE OpenSYCL::hipSYCL-rt)
        add_sycl_to_target(TARGET ${ARG_TARGET}  SOURCES ${ARG_SOURCES})
        message(STATUS "ARG_SOURCES ${ARG_SOURCES}")
        message(STATUS "ARG_DEPENDENCIES ${ARG_DEPENDENCIES}")
        message(STATUS "ARG_COMPILE_DEFINITIONS ${ARG_COMPILE_DEFINITIONS}")
        message(STATUS "ARG_INCLUDE_DIRECTORIES ${ARG_INCLUDE_DIRECTORIES}")
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

if (cuda)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "CUDA library not found")
  endif()
endif()

if (cuda)
  add_library(SYCLMath SHARED
  src/MathUtil.cxx
  )
  set_source_files_properties(src/MathUtil.cxx PROPERTIES LANGUAGE CUDA)
  target_compile_options(SYCLMath PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
  set_target_properties(SYCLMath PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(SYCLMath INTERFACE ROOT_MATH_CUDA)
else()
  add_library(SYCLMath SHARED
  src/MathUtil.cxx
  )
  set_target_properties(SYCLMath PROPERTIES LINKER_LANGUAGE CXX)
endif()


if (sycl)
    #set_target_properties(GenVector PROPERTIES LINKER_LANGUAGE CXX)
    target_compile_definitions(SYCLMath PUBLIC ROOT_MATH_SYCL)
    add_sycl_to_root_target(TARGET SYCLMath SOURCES src/MathUtil.cxx HEADERS inc/SYCLMath/GenVector/MathUtil.h INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc/ COMPILE_DEFINITIONS ROOT_MATH_SYCL ROOT_MEAS_TIMING )
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

target_link_libraries(GenVectorX PUBLIC SYCLMath)

if ((cuda) OR (sycl))
  add_library(VecOps SHARED
   src/VecOps.cxx
   )
  target_link_libraries(VecOps PUBLIC GenVectorX SYCLMath)

  if (cuda)
    set_source_files_properties(src/VecOps.cxx PROPERTIES LANGUAGE CUDA)
    target_compile_options(VecOps PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
    set_target_properties(VecOps PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(VecOps PUBLIC ROOT_MATH_CUDA ROOT_MEAS_TIMING)
  endif()
  if (sycl)
      set_target_properties(VecOps PROPERTIES LINKER_LANGUAGE CXX)
      #target_compile_definitions(VecOps PUBLIC ROOT_MATH_SYCL)
      add_sycl_to_root_target(TARGET VecOps SOURCES src/VecOps.cxx HEADERS inc/SYCLMath/VecOps.h INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc/ COMPILE_DEFINITIONS ROOT_MATH_SYCL ROOT_MEAS_TIMING )
  endif()

  install(TARGETS GenVectorX SYCLMath VecOps
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}   )
else()
  install(TARGETS GenVectorX SYCLMath 
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}   )
endif()

if (testing)
 add_subdirectory(testx)
endif()