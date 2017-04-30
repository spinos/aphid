
if (WIN32)
set (OpenEXR_INCLUDE_DIR D:/usr/openexr21/include)
set (OpenEXR_Library "D:/usr/openexr21/lib/Half.lib"
"D:/usr/openexr21/lib/Iex-2_1.lib"
"D:/usr/openexr21/lib/IlmImf-2_1.lib"
"D:/usr/openexr21/lib/IlmThread-2_1.lib")
elseif (APPLE)
set (OpenEXR_INCLUDE_DIR /Users/jianzhang/Library/openexr21/include)
set (OpenEXR_Library /Users/jianzhang/Library/openexr21/lib/libHalf.dylib
 /Users/jianzhang/Library/openexr21/lib/libIex-2_1.dylib
 /Users/jianzhang/Library/openexr21/lib/libIlmImf-2_1.dylib
 /Users/jianzhang/Library/openexr21/lib/libIlmThread-2_1.dylib)
else ()

SET (ILMBASE_PACKAGE_PREFIX /usr/local)

set (OpenEXR_INCLUDE_DIR ${ILMBASE_PACKAGE_PREFIX}/include)
set (OpenEXR_Library ${ILMBASE_PACKAGE_PREFIX}/lib/libHalf.so
    ${ILMBASE_PACKAGE_PREFIX}/lib/libIex-2_2.so
    ${ILMBASE_PACKAGE_PREFIX}/lib/libImath-2_2.so
    ${ILMBASE_PACKAGE_PREFIX}/lib/libIlmImf-2_2.so
    ${ILMBASE_PACKAGE_PREFIX}/lib/libIlmThread-2_2.so)
endif ()

message (" openexr include is " ${OpenEXR_INCLUDE_DIR})
message (" openexr library is " ${OpenEXR_Library})

if (WIN32)
    set (ZLIB_INCLUDE_DIR d:/usr/zlib-1.2.5)
    set (ZLIB_LIBRARY d:/usr/zlib-1.2.5/zdll.lib)

elseif (APPLE)
	set (ZLIB_INCLUDE_DIR /usr/include)
    set (ZLIB_LIBRARY /usr/lib/libz.dylib)

else ()
    set (ZLIB_INCLUDE_DIR /usr/include)
    set (ZLIB_LIBRARY /usr/lib64/libz.so)
endif ()

message (" zlib is " ${ZLIB_LIBRARY})

include_directories (${OpenEXR_INCLUDE_DIR})

SET (DTHF_SRC ${APHID_DIR}/shared/qt/BaseImageWidget.cpp
${APHID_DIR}/shared/qt/Plot1DWidget.cpp
${APHID_DIR}/shared/qt/Plot2DWidget.cpp
${APHID_DIR}/shared/qt/NavigatorWidget.cpp
${APHID_DIR}/shared/qt/QDoubleEditSlider.cpp
SynthControl.cpp
dthfwidget.cpp
dthfwindow.cpp
main.cpp)

add_executable (hfgen ${DTHF_SRC})
target_link_libraries (hfgen ${Boost_SYSTEM_LIBRARY})
target_link_libraries (hfgen ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (hfgen ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (hfgen ${Boost_REGEX_LIBRARY})
target_link_libraries (hfgen ${Boost_THREAD_LIBRARY})
target_link_libraries (hfgen ${Boost_CHRONO_LIBRARY})

target_link_libraries (hfgen ${APHID_MATH_LIBRARY})
target_link_libraries (hfgen ${APHID_GEOM_LIBRARY})
target_link_libraries (hfgen ${APHID_IMG_LIBRARY})
target_link_libraries (hfgen ${APHID_WLA_LIBRARY})

target_link_libraries (hfgen Qt4::QtGui)
target_link_libraries (hfgen ${OpenEXR_Library})
target_link_libraries (hfgen ${ZLIB_LIBRARY})
