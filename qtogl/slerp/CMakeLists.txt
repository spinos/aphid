set (SLERP_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
${APHID_QT_DIR}/QIconFrame.h
${APHID_QT_DIR}/ContextToolGroup.h
window.h
widget.h
slerp_common.h
ToolDlg.h)

set (SLERP_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
${APHID_QT_DIR}/QIconFrame.cpp
${APHID_QT_DIR}/ContextToolGroup.cpp
widget.cpp
window.cpp
ToolDlg.cpp
main.cpp)

set ( SLERP_RESOURCES
slerp.qrc
)

QT4_ADD_RESOURCES ( SLERP_RES_SOURCES ${SLERP_RESOURCES} )

message ("rcc output is " ${SLERP_RES_SOURCES})

add_executable (tslerp ${SLERP_HEADER_FILES} ${SLERP_SOURCE_FILES} ${SLERP_RES_SOURCES})

target_link_libraries (tslerp Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (tslerp ${OPENGL_LIBRARIES})
target_link_libraries (tslerp ${APHID_FOUND_LIBRARY})
target_link_libraries (tslerp ${APHID_MATH_LIBRARY})
target_link_libraries (tslerp ${APHID_GEOM_LIBRARY})
target_link_libraries (tslerp ${APHID_UI_LIBRARY})
target_link_libraries (tslerp ${APHID_SDB_LIBRARY})
target_link_libraries (tslerp ${APHID_TOPO_LIBRARY})
target_link_libraries (tslerp ${APHID_OGL_LIBRARY})
target_link_libraries (tslerp ${APHID_PBD_LIBRARY})

target_link_libraries (tslerp ${Boost_SYSTEM_LIBRARY})
target_link_libraries (tslerp ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (tslerp ${Boost_REGEX_LIBRARY})
target_link_libraries (tslerp ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (tslerp ${Boost_THREAD_LIBRARY})
target_link_libraries (tslerp ${Boost_CHRONO_LIBRARY})

IF(WIN32)
ELSEIF (APPLE)
ELSEIF (UNIX)
target_link_libraries (tslerp ${GLEW_LIBRARY})

ENDIF()

target_link_libraries (tslerp ${BLAS_LIBRARY})
target_link_libraries (tslerp ${LAPACK_LIBRARY})
target_link_libraries (tslerp ${F2C_LIBRARY})
