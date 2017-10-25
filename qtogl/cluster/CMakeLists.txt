set (CLS_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
window.h
widget.h)

set (CLS_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
${APHID_DIR}/shared/topo/TriangleMeshCluster.cpp
widget.cpp
window.cpp
main.cpp)

add_executable (clust ${CLS_HEADER_FILES} ${CLS_SOURCE_FILES})

target_link_libraries (clust Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (clust ${OPENGL_LIBRARIES})
target_link_libraries (clust ${APHID_FOUND_LIBRARY})
target_link_libraries (clust ${APHID_MATH_LIBRARY})
target_link_libraries (clust ${APHID_GEOM_LIBRARY})
target_link_libraries (clust ${APHID_UI_LIBRARY})
target_link_libraries (clust ${APHID_SDB_LIBRARY})
target_link_libraries (clust ${APHID_TOPO_LIBRARY})
target_link_libraries (clust ${APHID_OGL_LIBRARY})
target_link_libraries (clust ${APHID_KD_LIBRARY})
target_link_libraries (clust ${APHID_SMP_LIBRARY})

target_link_libraries (clust ${Boost_SYSTEM_LIBRARY})
IF (WIN32)
target_link_libraries (clust ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (clust ${Boost_THREAD_LIBRARY})
target_link_libraries (clust ${Boost_CHRONO_LIBRARY})

ELSEIF (APPLE)
ELSEIF (UNIX)
target_link_libraries (clust ${GLEW_LIBRARY})
target_link_libraries (clust ${BLAS_LIBRARY})

ENDIF()
