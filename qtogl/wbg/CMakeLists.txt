INCLUDE (${APHID_DIR}/cmake/exr_config.cmake)
set(CMAKE_MACOSX_BUNDLE OFF)

set (ABG_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
wbg_common.h
window.h
widget.h
HeightFieldAttrib.h
HeightFieldAssets.h
ModifyHeightField.h
DrawHeightField.h
assetdlg.h
toolBox.h)

set ( RESOURCES
wbg.qrc
)

QT4_ADD_RESOURCES ( RES_SOURCES ${RESOURCES} )

message ("rcc output is " ${RES_SOURCES})

set (WBG_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
${APHID_QT_DIR}/QIconFrame.cpp
${APHID_QT_DIR}/StateIconFrame.cpp
${APHID_QT_DIR}/ContextIconFrame.cpp
${APHID_QT_DIR}/ActionIconFrame.cpp
${APHID_QT_DIR}/IconLine.cpp
${APHID_QT_DIR}/BaseImageWidget.cpp
${APHID_QT_DIR}/Plot1DWidget.cpp
${APHID_QT_DIR}/Plot2DWidget.cpp
${APHID_QT_DIR}/NavigatorWidget.cpp
widget.cpp
window.cpp
HeightFieldAttrib.cpp
HeightFieldAssets.cpp
ModifyHeightField.cpp
DrawHeightField.cpp
assetdlg.cpp
toolBox.cpp
main.cpp)

add_executable (wbg ${WBG_HEADER_FILES} ${WBG_SOURCE_FILES} ${RES_SOURCES})

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
target_link_libraries (wbg ${APHID_IMG_LIBRARY})

target_link_libraries (wbg ${Boost_SYSTEM_LIBRARY})
IF (WIN32)
target_link_libraries (wbg ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (wbg ${Boost_THREAD_LIBRARY})
target_link_libraries (wbg ${Boost_CHRONO_LIBRARY})
ENDIF ()

target_link_libraries (wbg ${BLAS_LIBRARY})
target_link_libraries (wbg ${LAPACK_LIBRARY})
target_link_libraries (wbg ${F2C_LIBRARY})

target_link_libraries (wbg ${OpenEXR_Library})
