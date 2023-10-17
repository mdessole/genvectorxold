if (oneapi)
  add_executable(CoordsOneAPI Coords.cpp)
  #target_link_libraries(CoordsOneAPI PUBLIC GenVectorX SYCLMath)
  #include_directories(../inc)
  #target_compile_definitions(CoordsOneAPI PUBLIC ROOT_MATH_SYCL)
  #add_sycl_to_root_target(TARGET CoordsOneAPI SOURCES Coords.cpp)
 
  add_custom_command(OUTPUT CoordsOneAPI
    COMMAND ${SYCL_COMPILER} 
    ARGS ${SYCL_FLAGS}
    DEPENDS ${_deps}
    COMMENT "Building SYCL object ${CoordsOneAPI}"
    MAIN_DEPENDENCY Coords.cpp
  )
  
endif()

if (opensycl)
  add_executable(CoordsOpenSYCL Coords.cpp)
  target_link_libraries(CoordsOpenSYCL PUBLIC GenVectorX)
  include_directories(../inc)
  target_compile_definitions(CoordsOpenSYCL PUBLIC ROOT_MATH_SYCL)
  #add_sycl_to_root_target(TARGET CoordsOpenSYCL SOURCES Coords.cpp)
endif()

if ( (NOT oneapi) AND (NOT opensycl) )
  add_executable(Coords Coords.cpp)
  target_link_libraries(Coords PUBLIC GenVectorX)
  include_directories(../inc)
endif()