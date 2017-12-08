set (INST_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
window.h
widget.h)

set (INST_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
widget.cpp
window.cpp
main.cpp)

add_executable (rinst ${INST_HEADER_FILES} ${INST_SOURCE_FILES})

target_link_libraries (rinst Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (rinst ${OPENGL_LIBRARIES})
target_link_libraries (rinst ${APHID_FOUND_LIBRARY})
target_link_libraries (rinst ${APHID_MATH_LIBRARY})
target_link_libraries (rinst ${APHID_GEOM_LIBRARY})
target_link_libraries (rinst ${APHID_UI_LIBRARY})
target_link_libraries (rinst ${APHID_SDB_LIBRARY})
target_link_libraries (rinst ${APHID_TOPO_LIBRARY})
target_link_libraries (rinst ${APHID_OGL_LIBRARY})
target_link_libraries (rinst ${APHID_KD_LIBRARY})
target_link_libraries (rinst ${APHID_SMP_LIBRARY})

target_link_libraries (rinst ${Boost_SYSTEM_LIBRARY})
IF (WIN32)
target_link_libraries (rinst ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (rinst ${Boost_THREAD_LIBRARY})
target_link_libraries (rinst ${Boost_CHRONO_LIBRARY})
ELSEIF (UNIX)
target_link_libraries (rinst ${GLEW_LIBRARY})
target_link_libraries (rinst ${BLAS_LIBRARY})
ENDIF ()
