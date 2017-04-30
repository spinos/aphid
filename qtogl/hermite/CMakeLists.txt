set (HERM_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
window.h
widget.h)

set (HERM_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
widget.cpp
window.cpp
main.cpp)

add_executable (pwhermite ${HERM_HEADER_FILES} ${HERM_SOURCE_FILES})

target_link_libraries (pwhermite Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (pwhermite ${OPENGL_LIBRARIES})
target_link_libraries (pwhermite ${APHID_FOUND_LIBRARY})
target_link_libraries (pwhermite ${APHID_MATH_LIBRARY})
target_link_libraries (pwhermite ${APHID_GEOM_LIBRARY})
target_link_libraries (pwhermite ${APHID_UI_LIBRARY})
target_link_libraries (pwhermite ${APHID_SDB_LIBRARY})
target_link_libraries (pwhermite ${APHID_TOPO_LIBRARY})
target_link_libraries (pwhermite ${APHID_OGL_LIBRARY})

target_link_libraries (pwhermite ${Boost_SYSTEM_LIBRARY})
target_link_libraries (pwhermite ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (pwhermite ${Boost_REGEX_LIBRARY})
target_link_libraries (pwhermite ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (pwhermite ${Boost_THREAD_LIBRARY})
target_link_libraries (pwhermite ${Boost_CHRONO_LIBRARY})
