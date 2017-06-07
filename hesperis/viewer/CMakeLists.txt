
set(CMAKE_MACOSX_BUNDLE OFF)

set (VIE_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
window.h
glWidget.h
toolBox.h
)

set ( VIE_RESOURCES
viewer.qrc
)

QT4_ADD_RESOURCES ( VIE_RES_SOURCES ${VIE_RESOURCES} )

message ("rcc output is " ${VIE_RES_SOURCES})

set (VIE_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
${APHID_QT_DIR}/QIconFrame.cpp
${APHID_QT_DIR}/StateIconFrame.cpp
${APHID_QT_DIR}/ContextIconFrame.cpp
${APHID_QT_DIR}/ActionIconFrame.cpp
glWidget.cpp
window.cpp
toolBox.cpp
main.cpp)

add_executable (viewHes ${VIE_HEADER_FILES} ${VIE_SOURCE_FILES} ${VIE_RES_SOURCES})

target_link_libraries (viewHes Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (viewHes ${OPENGL_LIBRARIES})
target_link_libraries (viewHes ${APHID_FOUND_LIBRARY})
target_link_libraries (viewHes ${APHID_MATH_LIBRARY})
target_link_libraries (viewHes ${APHID_GEOM_LIBRARY})
target_link_libraries (viewHes ${APHID_UI_LIBRARY})
target_link_libraries (viewHes ${APHID_SDB_LIBRARY})
target_link_libraries (viewHes ${APHID_TOPO_LIBRARY})
target_link_libraries (viewHes ${APHID_OGL_LIBRARY})
target_link_libraries (viewHes ${APHID_GRAPH_LIBRARY})
target_link_libraries (viewHes ${APHID_TTG_LIBRARY})
target_link_libraries (viewHes ${APHID_KD_LIBRARY})
target_link_libraries (viewHes ${APHID_IMG_LIBRARY})
target_link_libraries (viewHes ${APHID_H5_LIBRARY})

target_link_libraries (viewHes ${HDF5_LIBRARIES})
target_link_libraries (viewHes ${ZLIB_LIBRARY})

target_link_libraries (viewHes ${Boost_SYSTEM_LIBRARY})
target_link_libraries (viewHes ${Boost_FILESYSTEM_LIBRARY})

IF (WIN32)
target_link_libraries (viewHes ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (viewHes ${Boost_THREAD_LIBRARY})
target_link_libraries (viewHes ${Boost_CHRONO_LIBRARY})
target_link_libraries (viewHes ${SZIP_LIBRARY})
ENDIF ()

target_link_libraries (viewHes ${BLAS_LIBRARY})
target_link_libraries (viewHes ${LAPACK_LIBRARY})
target_link_libraries (viewHes ${F2C_LIBRARY})
