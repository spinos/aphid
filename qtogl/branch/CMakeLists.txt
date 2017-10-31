set (BRH_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
window.h
widget.h)

set (BRH_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
${APHID_DIR}/shared/topo/TriangleMeshCluster.cpp
widget.cpp
window.cpp
main.cpp)

add_executable (branch ${BRH_HEADER_FILES} ${BRH_SOURCE_FILES})

target_link_libraries (branch Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (branch ${OPENGL_LIBRARIES})
target_link_libraries (branch ${APHID_FOUND_LIBRARY})
target_link_libraries (branch ${APHID_MATH_LIBRARY})
target_link_libraries (branch ${APHID_GEOM_LIBRARY})
target_link_libraries (branch ${APHID_UI_LIBRARY})
target_link_libraries (branch ${APHID_SDB_LIBRARY})
target_link_libraries (branch ${APHID_TOPO_LIBRARY})
target_link_libraries (branch ${APHID_OGL_LIBRARY})
target_link_libraries (branch ${APHID_KD_LIBRARY})
target_link_libraries (branch ${APHID_SMP_LIBRARY})
target_link_libraries (branch ${APHID_GRAPH_LIBRARY})

target_link_libraries (branch ${Boost_SYSTEM_LIBRARY})
IF (WIN32)
target_link_libraries (branch ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (branch ${Boost_THREAD_LIBRARY})
target_link_libraries (branch ${Boost_CHRONO_LIBRARY})

ELSEIF (APPLE)
target_link_libraries (branch ${BLAS_LIBRARY})

ELSEIF (UNIX)
target_link_libraries (branch ${GLEW_LIBRARY})

ENDIF()
