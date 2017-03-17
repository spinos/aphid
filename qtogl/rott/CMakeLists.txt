set (ROTT_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
window.h
widget.h)

set (ROTT_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
widget.cpp
window.cpp
main.cpp)

add_executable (rott ${ROTT_HEADER_FILES} ${ROTT_SOURCE_FILES})

target_link_libraries (rott Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (rott ${OPENGL_LIBRARIES})
target_link_libraries (rott ${APHID_FOUND_LIBRARY})
target_link_libraries (rott ${APHID_MATH_LIBRARY})
target_link_libraries (rott ${APHID_GEOM_LIBRARY})
target_link_libraries (rott ${APHID_UI_LIBRARY})
target_link_libraries (rott ${APHID_SDB_LIBRARY})
target_link_libraries (rott ${APHID_TOPO_LIBRARY})
target_link_libraries (rott ${APHID_OGL_LIBRARY})

target_link_libraries (rott ${Boost_SYSTEM_LIBRARY})
target_link_libraries (rott ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (rott ${Boost_REGEX_LIBRARY})
target_link_libraries (rott ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (rott ${Boost_THREAD_LIBRARY})
target_link_libraries (rott ${Boost_CHRONO_LIBRARY})
