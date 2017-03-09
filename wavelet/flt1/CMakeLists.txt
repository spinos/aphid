
SET (SHFT_SRC ${APHID_DIR}/shared/BaseImageWidget.cpp
${APHID_DIR}/shared/Plot1DWidget.cpp
${APHID_DIR}/wavelet/fltbanks.cpp
ft1dWidget.cpp
ftwindow.cpp
main.cpp)

add_executable (shoflt ${SHFT_SRC})
target_link_libraries (shoflt ${Boost_SYSTEM_LIBRARY})
target_link_libraries (shoflt ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (shoflt ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (shoflt ${Boost_REGEX_LIBRARY})
target_link_libraries (shoflt ${Boost_THREAD_LIBRARY})
target_link_libraries (shoflt ${Boost_CHRONO_LIBRARY})

target_link_libraries (shoflt ${APHID_MATH_LIBRARY})
target_link_libraries (shoflt ${APHID_GEOM_LIBRARY})

target_link_libraries (shoflt Qt4::QtGui)
