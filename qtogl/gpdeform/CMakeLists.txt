SET (GPDF_HEADER ../cactus.h)

SET (GPDF_SOURCE ${APHID_QT_DIR}/Base3DView.cpp
${APHID_QT_DIR}/BaseImageWidget.cpp
${APHID_QT_DIR}/Plot1DWidget.cpp
gpdfwidget.cpp
gpdfwindow.cpp
gpdfxwidget.cpp
gpdfxdialog.cpp
gpdfmain.cpp)

add_executable (gpdeform ${GP_HEADER} ${GPDF_SOURCE})
target_link_libraries (gpdeform Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (gpdeform ${OPENGL_LIBRARIES})
target_link_libraries (gpdeform ${APHID_FOUND_LIBRARY})
target_link_libraries (gpdeform ${APHID_MATH_LIBRARY})
target_link_libraries (gpdeform ${APHID_GEOM_LIBRARY})
target_link_libraries (gpdeform ${APHID_UI_LIBRARY})
target_link_libraries (gpdeform ${APHID_SDB_LIBRARY})
target_link_libraries (gpdeform ${APHID_TOPO_LIBRARY})
target_link_libraries (gpdeform ${APHID_OGL_LIBRARY})

target_link_libraries (gpdeform ${BLAS_LIBRARY})
target_link_libraries (gpdeform ${LAPACK_LIBRARY})
target_link_libraries (gpdeform ${F2C_LIBRARY})

target_link_libraries (gpdeform ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (gpdeform ${Boost_SYSTEM_LIBRARY})

IF (WIN32)
ELSEIF (APPLE)
ELSEIF (UNIX)
target_link_libraries (gpdeform ${GLEW_LIBRARY})
ENDIF ()

set(CMAKE_MACOSX_BUNDLE OFF)
