set (PCA_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
drpcawindow.h
drpcawidget.h)

set (PCA_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
drpcawidget.cpp
drpcawindow.cpp
testDrpca.cpp)

add_executable (drpca ${PCA_HEADER_FILES} ${PCA_SOURCE_FILES})

target_link_libraries (drpca Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (drpca ${OPENGL_LIBRARIES})
target_link_libraries (drpca ${APHID_FOUND_LIBRARY})
target_link_libraries (drpca ${APHID_MATH_LIBRARY})
target_link_libraries (drpca ${APHID_GEOM_LIBRARY})
target_link_libraries (drpca ${APHID_UI_LIBRARY})
target_link_libraries (drpca ${APHID_SDB_LIBRARY})
target_link_libraries (drpca ${APHID_TOPO_LIBRARY})
target_link_libraries (drpca ${APHID_OGL_LIBRARY})

target_link_libraries (drpca ${Boost_SYSTEM_LIBRARY})
target_link_libraries (drpca ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (drpca ${Boost_REGEX_LIBRARY})
target_link_libraries (drpca ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (drpca ${Boost_THREAD_LIBRARY})
target_link_libraries (drpca ${Boost_CHRONO_LIBRARY})

target_link_libraries (drpca ${BLAS_LIBRARY})
target_link_libraries (drpca ${LAPACK_LIBRARY})
target_link_libraries (drpca ${F2C_LIBRARY})

IF (UNIX)
target_link_libraries (drpca ${GLEW_LIBRARY})
ENDIF ()
