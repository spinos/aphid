set (ABG_HEADER_FILES ${APHID_DIR}/shared/Base3DView.h 
window.h
widget.h)

set (ABG_SOURCE_FILES ${APHID_DIR}/shared/Base3DView.cpp
${APHID_DIR}/shared/SelectionContext.cpp
widget.cpp
window.cpp
main.cpp)

add_executable (abg ${ABG_HEADER_FILES} ${ABG_SOURCE_FILES})

target_link_libraries (abg Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (abg ${OPENGL_LIBRARIES})
target_link_libraries (abg ${APHID_FOUND_LIBRARY})
target_link_libraries (abg ${APHID_MATH_LIBRARY})
target_link_libraries (abg ${APHID_GEOM_LIBRARY})
target_link_libraries (abg ${APHID_UI_LIBRARY})
target_link_libraries (abg ${APHID_SDB_LIBRARY})
target_link_libraries (abg ${APHID_TOPO_LIBRARY})
target_link_libraries (abg ${APHID_OGL_LIBRARY})
target_link_libraries (abg ${APHID_GRAPH_LIBRARY})
target_link_libraries (abg ${APHID_TTG_LIBRARY})
target_link_libraries (abg ${APHID_KD_LIBRARY})

target_link_libraries (abg ${Boost_SYSTEM_LIBRARY})
IF (WIN32)
target_link_libraries (abg ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (abg ${Boost_THREAD_LIBRARY})
target_link_libraries (abg ${Boost_CHRONO_LIBRARY})
ENDIF ()