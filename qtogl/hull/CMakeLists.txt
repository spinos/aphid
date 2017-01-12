set (HULL_HEADER_FILES ${APHID_DIR}/shared/Base3DView.h 
window.h
glwidget.h
HullContainer.h)

set (HULL_SOURCE_FILES ${APHID_DIR}/shared/Base3DView.cpp
glwidget.cpp
window.cpp
HullContainer.cpp
main.cpp)

add_executable (cvxhull ${HULL_HEADER_FILES} ${HULL_SOURCE_FILES})

target_link_libraries (cvxhull Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (cvxhull ${OPENGL_LIBRARIES})
target_link_libraries (cvxhull ${APHID_FOUND_LIBRARY})
target_link_libraries (cvxhull ${APHID_MATH_LIBRARY})
target_link_libraries (cvxhull ${APHID_GEOM_LIBRARY})
target_link_libraries (cvxhull ${APHID_UI_LIBRARY})
target_link_libraries (cvxhull ${APHID_SDB_LIBRARY})
target_link_libraries (cvxhull ${APHID_TOPO_LIBRARY})
target_link_libraries (cvxhull ${APHID_OGL_LIBRARY})

target_link_libraries (cvxhull ${Boost_SYSTEM_LIBRARY})
target_link_libraries (cvxhull ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (cvxhull ${Boost_REGEX_LIBRARY})
target_link_libraries (cvxhull ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (cvxhull ${Boost_THREAD_LIBRARY})
target_link_libraries (cvxhull ${Boost_CHRONO_LIBRARY})
