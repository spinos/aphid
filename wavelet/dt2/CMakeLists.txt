
SET (DT_2_SRC ${APHID_DIR}/shared/qt/BaseImageWidget.cpp
${APHID_DIR}/shared/qt/Plot1DWidget.cpp
${APHID_DIR}/shared/qt/Plot2DWidget.cpp
${APHID_DIR}/wavelet/gensig.cpp
dt2widget.cpp
dt2window.cpp
main.cpp)

add_executable (dt2dwt ${DT_2_SRC})
target_link_libraries (dt2dwt ${Boost_SYSTEM_LIBRARY})
target_link_libraries (dt2dwt ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (dt2dwt ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (dt2dwt ${Boost_REGEX_LIBRARY})
target_link_libraries (dt2dwt ${Boost_THREAD_LIBRARY})
target_link_libraries (dt2dwt ${Boost_CHRONO_LIBRARY})

target_link_libraries (dt2dwt ${APHID_MATH_LIBRARY})
target_link_libraries (dt2dwt ${APHID_GEOM_LIBRARY})
target_link_libraries (dt2dwt ${APHID_WLA_LIBRARY})

target_link_libraries (dt2dwt Qt4::QtGui)
