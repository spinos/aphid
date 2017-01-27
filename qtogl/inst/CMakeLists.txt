set (INST_HEADER_FILES ${APHID_DIR}/shared/Base3DView.h 
window.h
widget.h)

set (INST_SOURCE_FILES ${APHID_DIR}/shared/Base3DView.cpp
widget.cpp
window.cpp
main.cpp)

add_executable (psinst ${INST_HEADER_FILES} ${INST_SOURCE_FILES})

target_link_libraries (psinst Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (psinst ${OPENGL_LIBRARIES})
target_link_libraries (psinst ${APHID_FOUND_LIBRARY})
target_link_libraries (psinst ${APHID_MATH_LIBRARY})
target_link_libraries (psinst ${APHID_GEOM_LIBRARY})
target_link_libraries (psinst ${APHID_UI_LIBRARY})
target_link_libraries (psinst ${APHID_SDB_LIBRARY})
target_link_libraries (psinst ${APHID_TOPO_LIBRARY})
target_link_libraries (psinst ${APHID_OGL_LIBRARY})
target_link_libraries (psinst ${APHID_KD_LIBRARY})

target_link_libraries (psinst ${Boost_SYSTEM_LIBRARY})
