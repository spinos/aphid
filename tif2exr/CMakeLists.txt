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

include_directories (${OpenEXR_INCLUDE_DIR})
target_link_libraries (tif2exr ${OpenEXR_Library})

if (WIN32)
    
elseif (APPLE)
    
SET (TIF_INC /usr/local/include)
SET (TIF_LIBRARY /usr/local/lib/libtiff.a)
else ()

SET (TIF_INC /usr/local/include)
SET (TIF_LIBRARY /usr/lib64/libtiff.so)

endif ()

include_directories (TIF_INC)

message (" tiff is" ${TIF_LIBRARY})
target_link_libraries (tif2exr ${TIF_LIBRARY})

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

##find_package (zlib REQUIRED)

message (" zlib is " ${ZLIB_LIBRARY})
target_link_libraries (tif2exr ${ZLIB_LIBRARY})

SET (EXECUTABLE_OUTPUT_PATH ../bin)
