set (SOURC_FILES main.cpp)

ADD_EXECUTABLE (splmt ${SOURC_FILES})

target_link_libraries (splmt ${APHID_FOUND_LIBRARY})
target_link_libraries (splmt ${APHID_MATH_LIBRARY})
target_link_libraries (splmt ${APHID_GEOM_LIBRARY})
target_link_libraries (splmt ${APHID_OGL_LIBRARY})
target_link_libraries (splmt ${APHID_CU_LIBRARY})
target_link_libraries (splmt ${APHID_UI_LIBRARY})
target_link_libraries (splmt ${APHID_TOPO_LIBRARY})

INCLUDE (${APHID_DIR}/cmake/boost_config.cmake)

target_link_libraries (splmt ${Boost_SYSTEM_LIBRARY})

target_link_libraries (splmt ${BLAS_LIBRARY})
target_link_libraries (splmt ${LAPACK_LIBRARY})
target_link_libraries (splmt ${F2C_LIBRARY})
