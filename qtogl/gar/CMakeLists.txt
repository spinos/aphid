## Sat Aug 27 14:09:05 CST 2017 

INCLUDE (${APHID_DIR}/cmake/hdf5_config.cmake)
INCLUDE (${APHID_DIR}/cmake/zip_config.cmake)

include_directories (${APHID_DIR}/rhizoid)

set(CMAKE_MACOSX_BUNDLE OFF)

file (GLOB Asst_HEAD
    "asset/*.h"
)

file (GLOB Asst_SRC
    "asset/*.cpp"
)

file (GLOB Attr_HEAD
    "attr/*.h"
)

file (GLOB Attr_SRC
    "attr/*.cpp"
)

file (GLOB Chart_HEAD
    "graphchart/*.h"
)

file (GLOB Chart_SRC
    "graphchart/*.cpp"
)

file (GLOB Ui_HEAD
    "ui/*.h"
)

file (GLOB Ui_SRC
    "ui/*.cpp"
)

file (GLOB Syn_HEAD
    "syn/*.h"
)

file (GLOB Syn_SRC
    "syn/*.cpp"
)

set (GAR_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
data/grass.h
data/clover.h
data/haircap.h
data/poapratensis.h
data/monopodial.h
gar_common.h
ShrubScene.h
${APHID_QT_DIR}/GlyphPort.h
${APHID_QT_DIR}/GlyphConnection.h
${APHID_QT_DIR}/GlyphHalo.h
${Chart_HEAD}
${Asst_HEAD}
${Attr_HEAD}
${Ui_HEAD}
${Syn_HEAD}
PlantPiece.h
VegetationPatch.h
Vegetation.h
inout/ExportExample.h
inout/HGardenExample.h
inout/HVegePatch.h
GrowthSample.h
${APHID_DIR}/rhizoid/Variform.cpp
${APHID_DIR}/rhizoid/GardenExamp.cpp
${APHID_DIR}/rhizoid/CompoundExamp.cpp
${APHID_DIR}/rhizoid/SelectExmpCondition.cpp
${APHID_DIR}/rhizoid/ExampVox.cpp)

set ( GAR_RESOURCES
gar.qrc
)

QT4_ADD_RESOURCES ( GAR_RES_SOURCES ${GAR_RESOURCES} )

message ("rcc output is " ${GAR_RES_SOURCES})

set (GAR_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
${APHID_QT_DIR}/Base2DView.cpp
${APHID_QT_DIR}/QDoubleEditSlider.cpp
${APHID_QT_DIR}/QStringEditField.cpp
${APHID_QT_DIR}/QIconFrame.cpp
${APHID_QT_DIR}/StateIconFrame.cpp
${APHID_QT_DIR}/ContextIconFrame.cpp
${APHID_QT_DIR}/ActionIconFrame.cpp
${APHID_QT_DIR}/QSplineEdit.cpp
${APHID_QT_DIR}/SplineEditGroup.cpp
${APHID_QT_DIR}/IntEditGroup.cpp
${APHID_QT_DIR}/DoubleEditGroup.cpp
${APHID_QT_DIR}/QEnumCombo.cpp
${APHID_QT_DIR}/IconButtonGroup.cpp
ShrubScene.cpp
${APHID_QT_DIR}/GlyphPort.cpp
${APHID_QT_DIR}/GlyphConnection.cpp
${APHID_QT_DIR}/GlyphHalo.cpp
${Chart_SRC}
${Attr_SRC}
${Asst_SRC}
${Ui_SRC}
${Syn_SRC}
PlantPiece.cpp
VegetationPatch.cpp
Vegetation.cpp
inout/ExportExample.cpp
inout/HGardenExample.cpp
inout/HVegePatch.cpp
GrowthSample.cpp
main.cpp)

add_executable (gar ${GAR_HEADER_FILES} ${GAR_SOURCE_FILES} ${GAR_RES_SOURCES})

target_link_libraries (gar Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (gar ${OPENGL_LIBRARIES})
target_link_libraries (gar ${APHID_FOUND_LIBRARY})
target_link_libraries (gar ${APHID_MATH_LIBRARY})
target_link_libraries (gar ${APHID_GEOM_LIBRARY})
target_link_libraries (gar ${APHID_UI_LIBRARY})
target_link_libraries (gar ${APHID_SDB_LIBRARY})
target_link_libraries (gar ${APHID_TOPO_LIBRARY})
target_link_libraries (gar ${APHID_H5_LIBRARY})
target_link_libraries (gar ${APHID_OGL_LIBRARY})
target_link_libraries (gar ${APHID_GRAPH_LIBRARY})
target_link_libraries (gar ${APHID_TTG_LIBRARY})
target_link_libraries (gar ${APHID_KD_LIBRARY})
target_link_libraries (gar ${APHID_IMG_LIBRARY})
target_link_libraries (gar ${APHID_SMP_LIBRARY})

target_link_libraries (gar ${Boost_SYSTEM_LIBRARY})
target_link_libraries (gar ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (gar ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (gar ${Boost_THREAD_LIBRARY})
target_link_libraries (gar ${Boost_CHRONO_LIBRARY})
target_link_libraries (gar ${Boost_IOSTREAMS_LIBRARY})
target_link_libraries (gar ${Boost_REGEX_LIBRARY})

target_link_libraries (gar ${HDF5_LIBRARIES})
target_link_libraries (gar ${SZIP_LIBRARY})
target_link_libraries (gar ${ZLIB_LIBRARY})

IF (WIN32)
ELSEIF (APPLE)
ELSEIF (UNIX)
target_link_libraries (gar ${GLEW_LIBRARY})
ENDIF ()

target_link_libraries (gar ${BLAS_LIBRARY})
target_link_libraries (gar ${LAPACK_LIBRARY})
target_link_libraries (gar ${F2C_LIBRARY})

INCLUDE (${APHID_DIR}/cmake/exr_config.cmake)

target_link_libraries (gar ${OpenEXR_Library})

