
SET (DT_FLT_SRC ${APHID_DIR}/shared/qt/BaseImageWidget.cpp
${APHID_DIR}/shared/qt/Plot1DWidget.cpp
${APHID_DIR}/wavelet/gensig.cpp
dtftwidget.cpp
dtftwindow.cpp
main.cpp)

add_executable (dtft ${DT_FLT_SRC})
target_link_libraries (dtft ${Boost_SYSTEM_LIBRARY})
target_link_libraries (dtft ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (dtft ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (dtft ${Boost_REGEX_LIBRARY})
target_link_libraries (dtft ${Boost_THREAD_LIBRARY})
target_link_libraries (dtft ${Boost_CHRONO_LIBRARY})

target_link_libraries (dtft ${APHID_MATH_LIBRARY})
target_link_libraries (dtft ${APHID_GEOM_LIBRARY})
target_link_libraries (dtft ${APHID_WLA_LIBRARY})

target_link_libraries (dtft Qt4::QtGui)
