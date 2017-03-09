set (ABG_HEADER_FILES ${APHID_DIR}/shared/Base3DView.h 
window.h
widget.h)

set (WBG_SOURCE_FILES ${APHID_DIR}/shared/Base3DView.cpp
widget.cpp
window.cpp
main.cpp)

set(CMAKE_MACOSX_BUNDLE OFF)

add_executable (wbg ${WBG_HEADER_FILES} ${WBG_SOURCE_FILES})

target_link_libraries (wbg Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (wbg ${OPENGL_LIBRARIES})
target_link_libraries (wbg ${APHID_FOUND_LIBRARY})
target_link_libraries (wbg ${APHID_MATH_LIBRARY})
target_link_libraries (wbg ${APHID_GEOM_LIBRARY})
target_link_libraries (wbg ${APHID_UI_LIBRARY})
target_link_libraries (wbg ${APHID_SDB_LIBRARY})
target_link_libraries (wbg ${APHID_TOPO_LIBRARY})
target_link_libraries (wbg ${APHID_OGL_LIBRARY})
target_link_libraries (wbg ${APHID_GRAPH_LIBRARY})
target_link_libraries (wbg ${APHID_TTG_LIBRARY})
target_link_libraries (wbg ${APHID_KD_LIBRARY})

target_link_libraries (wbg ${Boost_SYSTEM_LIBRARY})
IF (WIN32)
target_link_libraries (wbg ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (wbg ${Boost_THREAD_LIBRARY})
target_link_libraries (wbg ${Boost_CHRONO_LIBRARY})
ENDIF ()

target_link_libraries (wbg ${BLAS_LIBRARY})
target_link_libraries (wbg ${LAPACK_LIBRARY})
target_link_libraries (wbg ${F2C_LIBRARY})
