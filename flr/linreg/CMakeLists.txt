set (SOURC_FILES main.cpp)

ADD_EXECUTABLE (linest ${SOURC_FILES})

target_link_libraries (linest ${APHID_FOUND_LIBRARY})
target_link_libraries (linest ${APHID_MATH_LIBRARY})

INCLUDE (${APHID_DIR}/cmake/boost_config.cmake)

target_link_libraries (linest ${Boost_SYSTEM_LIBRARY})

target_link_libraries (linest ${BLAS_LIBRARY})
target_link_libraries (linest ${LAPACK_LIBRARY})
target_link_libraries (linest ${F2C_LIBRARY})
