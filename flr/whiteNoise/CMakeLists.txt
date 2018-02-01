set (WNOI_HEADER_FILES 
RenderWidget.h
window.h)

set (WNOI_SOURCE_FILES
${interface_SRC}
RenderWidget.cpp
window.cpp
main.cpp)

add_executable (whiteNoise ${WNOI_HEADER_FILES} ${WNOI_SOURCE_FILES})

target_link_libraries (whiteNoise Qt4::QtGui)
target_link_libraries (whiteNoise ${APHID_FOUND_LIBRARY})
target_link_libraries (whiteNoise ${APHID_MATH_LIBRARY})
target_link_libraries (whiteNoise ${APHID_GEOM_LIBRARY})
target_link_libraries (whiteNoise ${APHID_SDB_LIBRARY})
target_link_libraries (whiteNoise ${APHID_TOPO_LIBRARY})

target_link_libraries (whiteNoise ${Boost_SYSTEM_LIBRARY})
target_link_libraries (whiteNoise ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (whiteNoise ${Boost_REGEX_LIBRARY})
target_link_libraries (whiteNoise ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (whiteNoise ${Boost_THREAD_LIBRARY})
target_link_libraries (whiteNoise ${Boost_CHRONO_LIBRARY})

IF(WIN32)
ELSEIF (APPLE)
ELSEIF (UNIX)
target_link_libraries (whiteNoise ${BLAS_LIBRARY})

ENDIF()
