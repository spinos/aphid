
SET (2D_DWT_SRC ${APHID_DIR}/shared/BaseImageWidget.cpp
${APHID_DIR}/shared/Plot1DWidget.cpp
${APHID_DIR}/shared/Plot2DWidget.cpp
${APHID_DIR}/wavelet/gensig.cpp
${APHID_DIR}/wavelet/dwt2.cpp
${APHID_DIR}/wavelet/fltbanks.cpp
ft2widget.cpp
ft2window.cpp
main.cpp)

add_executable (dwt2d ${2D_DWT_SRC})
target_link_libraries (dwt2d ${Boost_SYSTEM_LIBRARY})
target_link_libraries (dwt2d ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (dwt2d ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (dwt2d ${Boost_REGEX_LIBRARY})
target_link_libraries (dwt2d ${Boost_THREAD_LIBRARY})
target_link_libraries (dwt2d ${Boost_CHRONO_LIBRARY})

target_link_libraries (dwt2d ${APHID_MATH_LIBRARY})
target_link_libraries (dwt2d ${APHID_GEOM_LIBRARY})

target_link_libraries (dwt2d Qt4::QtGui)
