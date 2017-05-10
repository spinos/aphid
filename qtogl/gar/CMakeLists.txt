include_directories (${APHID_DIR}/rhizoid)

set(CMAKE_MACOSX_BUNDLE OFF)

set (GAR_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
data/grass.h
window.h
widget.h
gar_common.h
assetdlg.h
AssetDescription.h
plantAssets.h
groundAssets.h
GlyphPalette.h
ShrubScene.h
PiecesList.h
ShrubChartView.h
GardenGlyph.h
GlyphPort.h
GlyphConnection.h
GlyphBuilder.h
PlantPiece.h
VegetationPatch.h
Vegetation.h
DrawVegetation.h
exportDlg.h
ExportExample.h
HGardenExample.h
HVegePatch.h
toolBox.h
${APHID_DIR}/rhizoid/ExampVox.cpp)

set ( GAR_RESOURCES
gar.qrc
)

QT4_ADD_RESOURCES ( GAR_RES_SOURCES ${GAR_RESOURCES} )

message ("rcc output is " ${GAR_RES_SOURCES})

set (GAR_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
${APHID_QT_DIR}/QIconFrame.cpp
${APHID_QT_DIR}/StateIconFrame.cpp
${APHID_QT_DIR}/ContextIconFrame.cpp
${APHID_QT_DIR}/ActionIconFrame.cpp
widget.cpp
window.cpp
assetdlg.cpp
AssetDescription.cpp
plantAssets.cpp
groundAssets.cpp
GlyphPalette.cpp
toolBox.cpp
ShrubScene.cpp
PiecesList.cpp
ShrubChartView.cpp
GardenGlyph.cpp
GlyphPort.cpp
GlyphConnection.cpp
GlyphBuilder.cpp
PlantPiece.cpp
VegetationPatch.cpp
Vegetation.cpp
DrawVegetation.cpp
exportDlg.cpp
ExportExample.cpp
HGardenExample.cpp
HVegePatch.cpp
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
target_link_libraries (gar ${APHID_OGL_LIBRARY})
target_link_libraries (gar ${APHID_GRAPH_LIBRARY})
target_link_libraries (gar ${APHID_TTG_LIBRARY})
target_link_libraries (gar ${APHID_KD_LIBRARY})
target_link_libraries (gar ${APHID_IMG_LIBRARY})
target_link_libraries (gar ${APHID_H5_LIBRARY})

target_link_libraries (gar ${HDF5_LIBRARIES})
target_link_libraries (gar ${ZLIB_LIBRARY})

target_link_libraries (gar ${Boost_SYSTEM_LIBRARY})
target_link_libraries (gar ${Boost_FILESYSTEM_LIBRARY})

IF (WIN32)
target_link_libraries (gar ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (gar ${Boost_THREAD_LIBRARY})
target_link_libraries (gar ${Boost_CHRONO_LIBRARY})
target_link_libraries (gar ${SZIP_LIBRARY})
ENDIF ()

target_link_libraries (gar ${BLAS_LIBRARY})
target_link_libraries (gar ${LAPACK_LIBRARY})
target_link_libraries (gar ${F2C_LIBRARY})
