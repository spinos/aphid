
SET (HALTON_SRC ${APHID_DIR}/shared/qt/BaseImageWidget.cpp
${APHID_DIR}/shared/qt/Plot1DWidget.cpp
haltonwidget.cpp
window.cpp
main.cpp)

add_executable (haltont ${HALTON_SRC})
target_link_libraries (haltont ${Boost_SYSTEM_LIBRARY})
target_link_libraries (haltont ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (haltont ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (haltont ${Boost_REGEX_LIBRARY})
target_link_libraries (haltont ${Boost_THREAD_LIBRARY})
target_link_libraries (haltont ${Boost_CHRONO_LIBRARY})

target_link_libraries (haltont ${APHID_MATH_LIBRARY})
target_link_libraries (haltont ${APHID_GEOM_LIBRARY})

target_link_libraries (haltont Qt4::QtGui)
