
add_executable (wvlt ${APHID_DIR}/wavelet/fltbanks.cpp
${APHID_DIR}/wavelet/dwt2.cpp
main.cpp)

target_link_libraries (wvlt ${Boost_SYSTEM_LIBRARY})
target_link_libraries (wvlt ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (wvlt ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (wvlt ${Boost_REGEX_LIBRARY})
target_link_libraries (wvlt ${Boost_THREAD_LIBRARY})
target_link_libraries (wvlt ${Boost_CHRONO_LIBRARY})

target_link_libraries (wvlt ${APHID_MATH_LIBRARY})
target_link_libraries (wvlt ${APHID_GEOM_LIBRARY})
