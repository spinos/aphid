set (BRT_HEADER_FILES ${APHID_DIR}/shared/qt/Base3DView.h 
window.h
widget.h
../FeatherMesh.h)

set (BRT_SOURCE_FILES ${APHID_DIR}/shared/qt/Base3DView.cpp
widget.cpp
window.cpp
main.cpp
../FeatherDeformer.cpp
../FeatherMesh.cpp)

add_executable (bendrolltwist ${BRT_HEADER_FILES} ${BRT_SOURCE_FILES})

target_link_libraries (bendrolltwist Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (bendrolltwist ${OPENGL_LIBRARIES})
target_link_libraries (bendrolltwist ${APHID_FOUND_LIBRARY})
target_link_libraries (bendrolltwist ${APHID_MATH_LIBRARY})
target_link_libraries (bendrolltwist ${APHID_GEOM_LIBRARY})
target_link_libraries (bendrolltwist ${APHID_UI_LIBRARY})
target_link_libraries (bendrolltwist ${APHID_SDB_LIBRARY})
target_link_libraries (bendrolltwist ${APHID_TOPO_LIBRARY})
target_link_libraries (bendrolltwist ${APHID_OGL_LIBRARY})

target_link_libraries (bendrolltwist ${Boost_SYSTEM_LIBRARY})
target_link_libraries (bendrolltwist ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (bendrolltwist ${Boost_REGEX_LIBRARY})
target_link_libraries (bendrolltwist ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (bendrolltwist ${Boost_THREAD_LIBRARY})
target_link_libraries (bendrolltwist ${Boost_CHRONO_LIBRARY})
