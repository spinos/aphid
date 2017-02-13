set (tkm_HEADER_FILES ${APHID_DIR}/shared/Base3DView.h 
window.h
widget.h)

set (tkm_SOURCE_FILES ${APHID_DIR}/shared/Base3DView.cpp
widget.cpp
window.cpp
main.cpp)

add_executable (tkm ${tkm_HEADER_FILES} ${tkm_SOURCE_FILES})

target_link_libraries (tkm Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (tkm ${OPENGL_LIBRARIES})
target_link_libraries (tkm ${APHID_FOUND_LIBRARY})
target_link_libraries (tkm ${APHID_MATH_LIBRARY})
target_link_libraries (tkm ${APHID_GEOM_LIBRARY})
target_link_libraries (tkm ${APHID_UI_LIBRARY})
target_link_libraries (tkm ${APHID_SDB_LIBRARY})
target_link_libraries (tkm ${APHID_TOPO_LIBRARY})
target_link_libraries (tkm ${APHID_OGL_LIBRARY})

target_link_libraries (tkm ${Boost_SYSTEM_LIBRARY})
target_link_libraries (tkm ${Boost_DATE_TIME_LIBRARY})

target_link_libraries (tkm ${BLAS_LIBRARY})
target_link_libraries (tkm ${LAPACK_LIBRARY})
target_link_libraries (tkm ${F2C_LIBRARY})
