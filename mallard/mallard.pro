win32 {
    Aphid = D:/aphid
}
mac {
    Aphid = ../aphid
}

INCLUDEPATH += ./ ../shared ../lapl ../catmullclark ../easymodel ../../ofl/core ../../ofl/opium ../fit
HEADERS       = ../shared/AllMath.h \
				../shared/Vector3F.h \
                ../shared/Vector2F.h \
                ../shared/Matrix44F.h \
                ../shared/Matrix33F.h \
                ../shared/Ray.h \
                ../shared/RayIntersectionContext.h \
                ../shared/IntersectionContext.h \
                ../shared/BaseCamera.h \
                ../shared/PerspectiveCamera.h \
                ../shared/Base3DView.h \
                ../shared/BaseDrawer.h \
                ../shared/KdTreeDrawer.h \
                ../shared/SpaceHandle.h \
                ../shared/TypedEntity.h \
                ../shared/Geometry.h \
                ../shared/BaseMesh.h \
                ../shared/BoundingBox.h \
                ../shared/TriangleMesh.h \
                ../shared/GeoElement.h \
                ../shared/Vertex.h \
                ../shared/Edge.h \
                ../shared/Facet.h \
                ../shared/BaseField.h \
                ../shared/BaseDeformer.h \
                ../shared/KdTreeNode.h \
                ../shared/KdTree.h \
                ../shared/Primitive.h \
                ../shared/PrimitiveFilter.h \
                ../shared/BuildKdTreeContext.h \
                ../shared/KdTreeNodeArray.h \
                ../shared/PrimitiveArray.h \
                ../shared/IndexArray.h \
                ../shared/BaseArray.h \
                ../shared/ClassificationStorage.h \
                ../shared/KdTreeBuilder.h \
                ../shared/SplitEvent.h \
                ../shared/MinMaxBins.h \
                ../shared/BuildKdTreeStream.h \
                ../shared/IndexList.h \
                ../shared/BoundingBoxList.h \
                ../shared/SelectionArray.h \
                ../shared/ComponentConversion.h \
                ../shared/BarycentricCoordinate.h \
                ../shared/EasemodelUtil.h \
                ../shared/GeodesicSphereMesh.h \
				../shared/PyramidMesh.h \
				../shared/CubeMesh.h \
                ../shared/AnchorGroup.h \
                ../shared/BaseCurve.h \
                ../shared/VertexPath.h \
                ../shared/MeshTopology.h \
                ../shared/PatchMesh.h \
				../shared/BezierCurve.h \
				../shared/ContextIconFrame.h \
				../shared/ActionIconFrame.h \
				../shared/QIconFrame.h \
				../shared/ToolContext.h \
				../shared/BiLinearInterpolate.h \
				../shared/LinearInterpolate.h \
                ../shared/Plane.h \
                ../shared/PointInsidePolygonTest.h \
				../shared/AccPatchMesh.h \
				../shared/CircleCurve.h \
				../shared/InverseBilinearInterpolate.h \
				../shared/Segment.h \
				../shared/Patch.h \
				../shared/BaseBrush.h \
				../shared/QuickSort.h \
				../shared/GLight.h \
				../shared/GMaterial.h \
				../shared/GProfile.h \
				../shared/BaseTessellator.h \
				../shared/CollisionRegion.h \
				../shared/DrawBuffer.h \
				../shared/QIntEditSlider.h \
                ../shared/QDoubleEditSlider.h \
				../shared/BlockDrawBuffer.h \
                ../shared/BaseScene.h \
    $${Aphid}/shared/AllHdf.h \
    $${Aphid}/shared/HObject.h \
    $${Aphid}/shared/HDocument.h \
    $${Aphid}/shared/HGroup.h \
    $${Aphid}/shared/HDataset.h \
    $${Aphid}/shared/HAttribute.h \
    $${Aphid}/shared/VerticesHDataset.h \
    $${Aphid}/shared/IndicesHDataset.h \
    $${Aphid}/shared/HIntAttribute.h \
    $${Aphid}/shared/HFloatAttribute.h \
    $${Aphid}/shared/HBase.h \
    $${Aphid}/shared/HWorld.h \
    $${Aphid}/shared/HMesh.h \
				../lapl/VertexAdjacency.h \
                ../lapl/Anchor.h \
                ../../ofl/core/BaseImage.cpp \
                ../../ofl/core/zEXRImage.h \
                ../catmullclark/BaseQuad.h \
                ../catmullclark/LODQuad.h \
                ../catmullclark/accPatch.h \
                ../catmullclark/accStencil.h \
                ../catmullclark/bezierPatch.h \
                ../catmullclark/tessellator.h \
                ../catmullclark/AccCorner.h \
                ../catmullclark/AccEdge.h \
                ../catmullclark/AccInterior.h \
                ../catmullclark/BezierDrawer.h \
                ../fit/SingleModelView.h \
                ToolBox.h \
                glwidget.h \
                MlFeather.h \
				MlCalamus.h \
				MlCalamusArray.h \
				MlDrawer.h \
				MlTessellate.h \
				MlSkin.h \
                MlRachis.h \
                BrushControl.h \
                MlScene.h \
                window.h
                
SOURCES       = ../shared/Vector3F.cpp \  
                ../shared/Vector2F.cpp \
                ../shared/Matrix44F.cpp \
                ../shared/Matrix33F.cpp \
                ../shared/Ray.cpp \
                ../shared/RayIntersectionContext.cpp \
                ../shared/IntersectionContext.cpp \
                ../shared/BaseCamera.cpp \
                ../shared/PerspectiveCamera.cpp \
                ../shared/Base3DView.cpp \
                ../shared/BaseDrawer.cpp \
                ../shared/KdTreeDrawer.cpp \
                ../shared/SpaceHandle.cpp \
                ../shared/TypedEntity.cpp \
                ../shared/Geometry.cpp \
                ../shared/BoundingBox.cpp \
                ../shared/BaseMesh.cpp \
                ../shared/TriangleMesh.cpp \
                ../shared/GeoElement.cpp \
                ../shared/Vertex.cpp \
                ../shared/Edge.cpp \
                ../shared/Facet.cpp \
                ../shared/BaseField.cpp \
                ../shared/BaseDeformer.cpp \
                ../shared/KdTreeNode.cpp \
                ../shared/KdTree.cpp \
                ../shared/Primitive.cpp \
                ../shared/PrimitiveFilter.cpp \
                ../shared/BuildKdTreeContext.cpp \
                ../shared/KdTreeNodeArray.cpp \
                ../shared/PrimitiveArray.cpp \
                ../shared/IndexArray.cpp \
                ../shared/BaseArray.cpp \
                ../shared/ClassificationStorage.cpp \
                ../shared/KdTreeBuilder.cpp \
                ../shared/SplitEvent.cpp \
                ../shared/MinMaxBins.cpp \
                ../shared/BuildKdTreeStream.cpp \
                ../shared/IndexList.cpp \
                ../shared/BoundingBoxList.cpp \
                ../shared/SelectionArray.cpp \
                ../shared/ComponentConversion.cpp \
                ../shared/BarycentricCoordinate.cpp \
                ../shared/EasemodelUtil.cpp \
                ../shared/GeodesicSphereMesh.cpp \
                ../shared/PyramidMesh.cpp \
                ../shared/CubeMesh.cpp \
                ../shared/AnchorGroup.cpp \
                ../shared/BaseCurve.cpp \
                ../shared/VertexPath.cpp \
                ../shared/MeshTopology.cpp \
                ../shared/PatchMesh.cpp \
				../shared/BezierCurve.cpp \
				../shared/ContextIconFrame.cpp \
				../shared/ActionIconFrame.cpp \
				../shared/QIconFrame.cpp \
				../shared/ToolContext.cpp \
				../shared/BiLinearInterpolate.cpp \
				../shared/LinearInterpolate.cpp \
                ../shared/Plane.cpp \
                ../shared/PointInsidePolygonTest.cpp \
				../shared/AccPatchMesh.cpp \
				../shared/CircleCurve.cpp \
				../shared/InverseBilinearInterpolate.cpp \
				../shared/Segment.cpp \
				../shared/Patch.cpp \
				../shared/BaseBrush.cpp \
				../shared/QuickSort.cpp \
				../shared/GLight.cpp \
				../shared/GMaterial.cpp \
				../shared/GProfile.cpp \
				../shared/BaseTessellator.cpp \
				../shared/CollisionRegion.cpp \
				../shared/DrawBuffer.cpp \
				../shared/QIntEditSlider.cpp \
                ../shared/QDoubleEditSlider.cpp \
				../shared/BlockDrawBuffer.cpp \
                ../shared/BaseScene.cpp \
    $${Aphid}/shared/HObject.cpp \
    $${Aphid}/shared/HDocument.cpp \
    $${Aphid}/shared/HGroup.cpp \
    $${Aphid}/shared/HDataset.cpp \
    $${Aphid}/shared/HAttribute.cpp \
    $${Aphid}/shared/VerticesHDataset.cpp \
    $${Aphid}/shared/IndicesHDataset.cpp \
    $${Aphid}/shared/HIntAttribute.cpp \
    $${Aphid}/shared/HFloatAttribute.cpp \
    $${Aphid}/shared/HBase.cpp \
    $${Aphid}/shared/HWorld.cpp \
    $${Aphid}/shared/HMesh.cpp \
				../lapl/VertexAdjacency.cpp \
                ../lapl/Anchor.cpp \
                ../../ofl/core/BaseImage.cpp \
                ../../ofl/core/zEXRImage.cpp \
                ../catmullclark/BaseQuad.cpp \
                ../catmullclark/LODQuad.cpp \
                ../catmullclark/accPatch.cpp \
                ../catmullclark/accStencil.cpp \
                ../catmullclark/bezierPatch.cpp \
                ../catmullclark/tessellator.cpp \
                ../catmullclark/AccCorner.cpp \
                ../catmullclark/AccEdge.cpp \
                ../catmullclark/AccInterior.cpp \
                ../catmullclark/BezierDrawer.cpp \
                ../fit/SingleModelView.cpp \
                ToolBox.cpp \
                glwidget.cpp \
                MlFeather.cpp \
				MlCalamus.cpp \
				MlCalamusArray.cpp \
				MlDrawer.cpp \
				MlTessellate.cpp \
				MlSkin.cpp \
                MlRachis.cpp \
                window.cpp \
                BrushControl.cpp \
                MlScene.cpp \
                main.cpp
                
INCLUDEPATH += /usr/local/include/OpenEXR
LIBS += -L../easymodel -leasymodel -lIlmImf -lHalf -lhdf5 -lhdf5_hl
macx {
    INCLUDEPATH += ../../Library/boost_1_44_0 \
                    ../../Library/hdf5/include
    QMAKE_LIBDIR += ../../Library/boost_1_44_0/stage/lib \
                    ../../Library/hdf5/lib
    LIBS += -lboost_date_time -lboost_thread -lboost_filesystem -lboost_system -framework libxml
}
win32 {
    HEADERS += ../shared/gExtension.h
    SOURCES += ../shared/gExtension.cpp
    INCLUDEPATH += D:/usr/local/include D:/ofl/shared \
                   D:/usr/hdf5/include \
                   D:/usr/libxml2x64/include D:/usr/eigen3
    QMAKE_LIBDIR += D:/usr/local/lib64 
    LIBS += -LD:/usr/libxml2x64/lib -llibxml2 \
            -LD:/usr/hdf5/lib -lszip
    DEFINES += OPENEXR_DLL NDEBUG
CONFIG += console
}
QT           += opengl
RESOURCES += ../icons/mallard.qrc
