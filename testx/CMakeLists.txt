if (oneapi)
  add_executable(InvariantMassesSYCL InvariantMassesSYCL.cxx)
  target_link_options(InvariantMassesSYCL PUBLIC -E)
  add_sycl_to_root_target_exe(TARGET InvariantMassesSYCL SOURCES InvariantMassesSYCL.cxx DEPENDENCIES GenVectorX SYCLMath INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc COMPILE_DEFINITIONS ROOT_MATH_SYCL COMPILE_TIMING)
endif()

if (opensycl)
   add_executable(InvariantMassesSYCL InvariantMassesSYCL.cxx)
  add_sycl_to_root_target(TARGET InvariantMassesSYCL SOURCES InvariantMassesSYCL.cxx DEPENDENCIES GenVectorX SYCLMath INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/inc COMPILE_DEFINITIONS ROOT_MATH_SYCL)
endif()


if (cuda)
  add_executable(InvariantMassesCUDA InvariantMassesCUDA.cu)
  set_target_properties(InvariantMassesCUDA PROPERTIES LINKER_LANGUAGE CUDA)
  target_compile_options(InvariantMassesCUDA PRIVATE -rdc=true -lineinfo --expt-relaxed-constexpr)
  set_target_properties(InvariantMassesCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(InvariantMassesCUDA INTERFACE ROOT_MATH_CUDA)
endif()


if ( (NOT oneapi) AND (NOT opensycl) AND (NOT cuda) )
  add_executable(Coords Coords.cpp) 
  target_link_libraries(Coords PUBLIC GenVectorX SYCLMath)
endif()
#add_executable(InvariantMasses InvariantMasses.cpp)
#target_compile_options(InvariantMasses PUBLIC `root-config --libs --cflags`)