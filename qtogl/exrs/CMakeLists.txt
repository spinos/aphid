INCLUDE (${APHID_DIR}/cmake/exr_config.cmake)

set (exrs_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
Parameter.h
window.h
widget.h)

set (exrs_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
Parameter.cpp
widget.cpp
window.cpp
main.cpp)

set(CMAKE_MACOSX_BUNDLE OFF)

add_executable (exrs ${exrs_HEADER_FILES} ${exrs_SOURCE_FILES})

target_link_libraries (exrs Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (exrs ${OPENGL_LIBRARIES})
target_link_libraries (exrs ${APHID_FOUND_LIBRARY})
target_link_libraries (exrs ${APHID_MATH_LIBRARY})
target_link_libraries (exrs ${APHID_GEOM_LIBRARY})
target_link_libraries (exrs ${APHID_UI_LIBRARY})
target_link_libraries (exrs ${APHID_SDB_LIBRARY})
target_link_libraries (exrs ${APHID_TOPO_LIBRARY})
target_link_libraries (exrs ${APHID_OGL_LIBRARY})
target_link_libraries (exrs ${APHID_IMG_LIBRARY})

target_link_libraries (exrs ${OpenEXR_Library})

target_link_libraries (exrs ${Boost_SYSTEM_LIBRARY})
target_link_libraries (exrs ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (exrs ${Boost_REGEX_LIBRARY})
target_link_libraries (exrs ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (exrs ${Boost_THREAD_LIBRARY})
target_link_libraries (exrs ${Boost_CHRONO_LIBRARY})
