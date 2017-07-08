MESSAGE (" find openEXR")
if (WIN32)
add_definitions (-DOPENEXR_DLL)
set (OpenEXR_INCLUDE_DIR D:/usr/openexr21/include)
set (OpenEXR_Library "D:/usr/openexr21/lib/Half.lib"
"D:/usr/openexr21/lib/Iex-2_1.lib"
"D:/usr/openexr21/lib/IlmImf-2_1.lib"
"D:/usr/openexr21/lib/IlmThread-2_1.lib")

ELSEIF (APPLE)
set (OpenEXR_INCLUDE_DIR /Users/jianzhang/Library/openexr21/include)
set (OpenEXR_Library /Users/jianzhang/Library/openexr21/lib/libHalf.dylib
 /Users/jianzhang/Library/openexr21/lib/libIex-2_1.dylib
 /Users/jianzhang/Library/openexr21/lib/libIlmImf-2_1.dylib
 /Users/jianzhang/Library/openexr21/lib/libIlmThread-2_1.dylib)

ELSEIF (UNIX)
    SET (ILMBASE_PACKAGE_PREFIX /usr/local)
SET (OpenEXR_INCLUDE_DIR ${ILMBASE_PACKAGE_PREFIX}/include)
SET (OpenEXR_Library ${ILMBASE_PACKAGE_PREFIX}/lib/libHalf.so
    ${ILMBASE_PACKAGE_PREFIX}/lib/libIex-2_1.so
    ${ILMBASE_PACKAGE_PREFIX}/lib/libImath-2_1.so
    ${ILMBASE_PACKAGE_PREFIX}/lib/libIlmImf-2_1.so
    ${ILMBASE_PACKAGE_PREFIX}/lib/libIlmThread-2_1.so)

ENDIF ()

message (" openexr is " ${OpenEXR_Library})

include_directories (${OpenEXR_INCLUDE_DIR})

