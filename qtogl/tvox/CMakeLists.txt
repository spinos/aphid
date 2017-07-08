set(CMAKE_MACOSX_BUNDLE OFF)

set (CLS_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
window.h
widget.h)

set (CLS_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
widget.cpp
window.cpp
main.cpp)

add_executable (tvox ${CLS_HEADER_FILES} ${CLS_SOURCE_FILES})

target_link_libraries (tvox Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (tvox ${OPENGL_LIBRARIES})
target_link_libraries (tvox ${APHID_FOUND_LIBRARY})
target_link_libraries (tvox ${APHID_MATH_LIBRARY})
target_link_libraries (tvox ${APHID_GEOM_LIBRARY})
target_link_libraries (tvox ${APHID_UI_LIBRARY})
target_link_libraries (tvox ${APHID_SDB_LIBRARY})
target_link_libraries (tvox ${APHID_TOPO_LIBRARY})
target_link_libraries (tvox ${APHID_OGL_LIBRARY})
target_link_libraries (tvox ${APHID_KD_LIBRARY})

target_link_libraries (tvox ${Boost_SYSTEM_LIBRARY})
IF (WIN32)
target_link_libraries (tvox ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (tvox ${Boost_THREAD_LIBRARY})
target_link_libraries (tvox ${Boost_CHRONO_LIBRARY})
ENDIF ()

target_link_libraries (tvox ${BLAS_LIBRARY})
target_link_libraries (tvox ${LAPACK_LIBRARY})
target_link_libraries (tvox ${F2C_LIBRARY})

IF (WIN32)
ELSEIF (APPLE)
ELSEIF (UNIX)
target_link_libraries (tvox ${GLEW_LIBRARY})

ENDIF()
