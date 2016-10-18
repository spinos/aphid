cmake_minimum_required (VERSION 2.8)
project (tif2exr)
# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)
set (CMAKE_BUILD_TYPE RELEASE)


add_executable (tif2exr main.cpp)

if (WIN32)
add_definitions (-DOPENEXR_DLL)
set (OpenEXR_INCLUDE_DIR D:/usr/openexr21/include)
set (OpenEXR_Library "D:/usr/openexr21/lib/Half.lib"
"D:/usr/openexr21/lib/Iex-2_1.lib"
"D:/usr/openexr21/lib/IlmImf-2_1.lib"
"D:/usr/openexr21/lib/IlmThread-2_1.lib")
else ()
set (OpenEXR_INCLUDE_DIR /Users/jianzhang/Library/openexr21/include)
set (OpenEXR_Library /Users/jianzhang/Library/openexr21/lib/libHalf.dylib
 /Users/jianzhang/Library/openexr21/lib/libIex-2_1.dylib
 /Users/jianzhang/Library/openexr21/lib/libIlmImf-2_1.dylib
 /Users/jianzhang/Library/openexr21/lib/libIlmThread-2_1.dylib)
endif ()
message (" openexr " ${OpenEXR_Library})

include_directories (${OpenEXR_INCLUDE_DIR})
target_link_libraries (tif2exr ${OpenEXR_Library})

include_directories (OpenEXR_INCLUDE_DIR)
target_link_libraries (tif2exr ${OpenEXR_Library})


SET (TIF_INC /usr/local/include)
SET (TIF_LIBRARY /usr/local/lib/libtiff.a)
include_directories (TIF_INC)

message (" tiff is" ${TIF_LIBRARY})
target_link_libraries (tif2exr ${TIF_LIBRARY})

if (WIN32)
    set (ZLIB_INCLUDE_DIR d:/usr/zlib-1.2.5)
    set (ZLIB_LIBRARY d:/usr/zlib-1.2.5/zdll.lib)
endif ()

find_package (zlib REQUIRED)

message (" zlib is " ${ZLIB_LIBRARIES})
target_link_libraries (tif2exr ${ZLIB_LIBRARIES})

SET (EXECUTABLE_OUTPUT_PATH ../bin)
