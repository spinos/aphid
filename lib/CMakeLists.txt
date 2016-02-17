cmake_minimum_required (VERSION 2.8)
project (aphid)
set (CMAKE_BUILD_TYPE RELEASE)
include_directories(../mallard ../shared ../lapl ../catmullclark ../easymodel ../../ofl/opium ../fit ../powell ../../ofl/shared
../btree
../kdtree
../hesperis)
IF (WIN32)
include_directories(D:/usr/libxml2/include D:/usr/HDF5/include)
ADD_DEFINITIONS( -DNDEBUG -D_WIN32_WINDOWS 
## http://www.boost.org/doc/libs/1_49_0/libs/config/doc/html/boost_config/boost_macro_reference.html
-DBOOST_HAS_TR1)
## http://stackoverflow.com/questions/3499487/replacing-native-vs-2010-vc10-tr1-libraries-with-boost-tr1
set (_HAS_CPP0X 0)
set (AttributeNoninline "__attribute__ ((noinline))")
add_definitions (-DBOOST_NOINLINE=${AttributeNoninline})

ELSE ()
set (AttributeNoninline "__attribute__ ((noinline))")
add_definitions (-DBOOST_NOINLINE=${AttributeNoninline})

ENDIF()

SET (INSTALLED_HDF5 OFF)
if (WIN32)
	IF (EXISTS "C:/Program Files/HDF5/cmake/hdf5")
# location of configure file FindHDF5.cmake
		SET (HDF5_DIR "C:/Program Files/HDF5/cmake/hdf5")
        SET (INSTALLED_HDF5 ON)
    ELSEIF (EXISTS "D:/usr/hdf5")
        SET (HDF5_INCLUDE_DIRS "D:/usr/hdf5/include")
        SET (HDF5_LIBRARIES "D:/usr/hdf5/lib/libhdf5.lib")
	ENDIF ()	
endif ()

IF (INSTALLED_HDF5)
FIND_PACKAGE (HDF5 REQUIRED)
IF (WIN32)
SET (HDF5_LIBRARIES "C:/Program Files/HDF5/lib/libhdf5.lib")	
ENDIF ()
ENDIF ()

MESSAGE (" hdf5 version is " ${HDF5_VERSION} )
MESSAGE (" hdf5 include is " ${HDF5_INCLUDE_DIRS} )
MESSAGE (" hdf5 library is " ${HDF5_LIBRARIES} )

include_directories (${HDF5_INCLUDE_DIRS})

if(WIN32)
	IF (EXISTS "D:/usr/boost_1_51_0")
		SET (Boost_INCLUDE_DIR "D:/usr/boost_1_51_0")
	
	ENDIF ()
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	
ELSEIF (APPLE)
	IF (EXISTS "/users/jianzhang/Library/boost_1_55_0")
		SET (Boost_INCLUDE_DIR "/users/jianzhang/Library/boost_1_55_0")
	ELSEIF (EXISTS "/users/jollyroger/Library/boost_1_43_0")
		SET (Boost_INCLUDE_DIR "/users/jollyroger/Library/boost_1_43_0")
	ENDIF ()
	
endif()

FIND_PACKAGE (Boost REQUIRED COMPONENTS system filesystem date_time regex)	
	
MESSAGE (" boost version is " ${Boost_VERSION})
MESSAGE (" boost include is " ${Boost_INCLUDE_DIR} )
MESSAGE (" boost system is " ${Boost_SYSTEM_LIBRARY})
MESSAGE (" boost date_time is " ${Boost_DATE_TIME_LIBRARY})
MESSAGE (" boost regex is " ${Boost_REGEX_LIBRARY})
MESSAGE (" boost filesystem is " ${Boost_FILESYSTEM_LIBRARY})

include_directories (${Boost_INCLUDE_DIR})
include_directories (${Boost_INCLUDE_DIR}/boost/tr1/tr1)

add_library(aphid STATIC ../shared/line_math.h
		../shared/tetrahedron_math.h
                ../shared/Vector3F.cpp 
                ../shared/Vector2F.cpp 
                ../shared/Matrix44F.cpp 
                ../shared/Matrix33F.cpp 
                ../shared/Quaternion.cpp 
                ../shared/Ray.cpp 
                ../shared/RayIntersectionContext.cpp 
                ../shared/IntersectionContext.cpp 
                ../shared/BaseCamera.cpp 
                ../shared/PerspectiveCamera.cpp 
                ../shared/BaseDrawer.cpp 
                ../shared/KdTreeDrawer.cpp 
                ../shared/SpaceHandle.cpp 
                ../shared/TypedEntity.cpp 
				../shared/NamedEntity.cpp 
				../shared/BaseTransform.cpp 
                ../shared/Geometry.cpp 
                ../shared/BaseMesh.cpp 
				../shared/PlaneMesh.cpp
                ../shared/Triangle.cpp 
                ../shared/GeoElement.cpp 
                ../shared/Vertex.cpp 
                ../shared/Edge.cpp 
                ../shared/Facet.cpp 
                ../shared/BaseField.cpp 
                ../shared/BaseDeformer.cpp 
    ../shared/SkeletonSubspaceDeformer.cpp 
                ../shared/KdTreeNode.cpp 
                ../shared/Primitive.cpp 
                ../shared/PrimitiveFilter.cpp 
                ../shared/BuildKdTreeContext.cpp 
                ../shared/KdTreeNodeArray.cpp 
                ../shared/KdTree.cpp 
                ../shared/PrimitiveArray.cpp 
                ../shared/IndexArray.cpp 
                ../shared/BaseArray.cpp 
                ../shared/ClassificationStorage.cpp 
                ../shared/KdTreeBuilder.cpp 
                ../shared/SplitEvent.cpp 
                ../shared/MinMaxBins.cpp 
                ../shared/BuildKdTreeStream.cpp 
                ../shared/IndexList.cpp 
                ../shared/BoundingBoxList.cpp 
                ../shared/SelectionArray.cpp 
                ../shared/ComponentConversion.cpp 
                ../shared/BarycentricCoordinate.cpp 
                ../shared/EasemodelUtil.cpp 
                ../shared/GeodesicSphereMesh.cpp 
                ../shared/PyramidMesh.cpp 
                ../shared/CubeMesh.cpp 
				../shared/DiscMesh.cpp 
                ../shared/AnchorGroup.cpp 
                ../shared/BaseCurve.cpp 
                ../shared/VertexPath.cpp 
                ../shared/MeshTopology.cpp 
                ../shared/PatchMesh.cpp 
                                ../shared/BezierCurve.cpp 
                                ../shared/ToolContext.cpp 
                                ../shared/BiLinearInterpolate.cpp 
                                ../shared/LinearInterpolate.cpp 
                ../shared/Plane.cpp 
                ../shared/PointInsidePolygonTest.cpp 
                                ../shared/AccPatchMesh.cpp 
								../shared/PatchNeighborRec.cpp 
								../shared/PatchTexture.cpp 
                                ../shared/CircleCurve.cpp 
                                ../shared/InverseBilinearInterpolate.cpp 
                                ../shared/Segment.cpp 
                                ../shared/Patch.cpp 
                                ../shared/BaseBrush.cpp 
                                ../shared/QuickSort.cpp 
                                ../shared/GLight.cpp 
                                ../shared/GMaterial.cpp 
                                ../shared/GProfile.cpp 
                                ../shared/BaseTessellator.cpp 
                                ../shared/CollisionRegion.cpp 
                                ../shared/DrawBuffer.cpp 
                                ../shared/BlockDrawBuffer.cpp 
                ../shared/BaseScene.cpp 
    ../shared/BoundingBox.cpp 
    ../shared/BoundingRectangle.cpp 
    ../shared/HObject.cpp 
    ../shared/HDocument.cpp 
    ../shared/HGroup.cpp 
    ../shared/HDataset.cpp 
    ../shared/HAttribute.cpp 
    ../shared/VerticesHDataset.cpp 
    ../shared/IndicesHDataset.cpp 
        ../shared/FloatsHDataset.cpp 
../shared/XformHDataset.cpp 
    ../shared/HIntAttribute.cpp 
    ../shared/HFloatAttribute.cpp 
    ../shared/HStringAttribute.cpp 
    ../shared/HCharData.cpp 
    ../shared/HBase.cpp 
    ../shared/HWorld.cpp 
    ../shared/HMesh.cpp 
    ../shared/BaseImage.cpp 
        ../shared/BaseFile.cpp
    ../shared/BaseFileGroup.cpp
    ../shared/BakeDeformer.cpp 
    ../shared/PlaybackControl.cpp 
    ../shared/CacheFile.cpp 
    ../shared/BaseState.cpp 
    ../shared/HFile.cpp 
    ../shared/HRFile.cpp 
    ../shared/TransformManipulator.cpp 
    ../shared/MeshManipulator.cpp 
	../shared/SkeletonJoint.cpp 
	../shared/SkeletonSystem.cpp 
	../shared/SkeletonPose.cpp 
	../shared/LineBuffer.cpp 
	../shared/SelectCondition.cpp 
    ../shared/FloodCondition.cpp 
	../shared/ActiveRegion.cpp 
	../shared/AccPatchGroup.cpp 
	../shared/VertexDataGroup.cpp 
	../shared/BezierPatchHirarchy.cpp 
	../shared/KMeansClustering.cpp 
	../shared/BaseSphere.cpp 
	../shared/PseudoNoise.cpp 
## ../shared/BaseServer.cpp 
    ../shared/BaseClient.cpp 
	../shared/RenderEngine.cpp 
	../shared/AdaptableStripeBuffer.cpp 
	../shared/BlockStripeBuffer.cpp 
	../shared/LODFn.cpp 
	../shared/NoisePlot.cpp 
	../shared/FractalPlot.cpp 
	../shared/BaseLight.cpp 
	../shared/DistantLight.cpp 
	../shared/PointLight.cpp 
	../shared/SquareLight.cpp 
../shared/LightGroup.cpp 
../shared/LineDrawer.cpp 
../shared/MeshDrawer.cpp 
../shared/GeoDrawer.cpp 
../shared/LightDrawer.cpp 
../shared/RenderOptions.cpp 
../shared/SelectionContext.cpp 
../shared/BaseTexture.cpp 
../shared/TextureGroup.cpp 
../shared/ColorBlend.cpp 
../shared/TexturePainter.cpp 
../shared/BaseShader.cpp 
../shared/ShaderGroup.cpp 
##../shared/RayMarch.cpp 
../shared/GLHUD.cpp
../shared/BaseLog.cpp
../shared/CurveGroup.cpp 
../shared/HCurveGroup.cpp
../shared/BaseBuffer.cpp
../shared/HFrameRange.cpp
../shared/HTransform.cpp
../shared/HTriangleMesh.cpp
../shared/HTetrahedronMesh.cpp
../shared/HAttributeEntry.cpp
../shared/AFrameRange.cpp
../shared/APlayback.cpp
../shared/APlaybackFile.cpp
../shared/MultiPlaybackFile.cpp
../shared/AGenericMesh.cpp
../shared/CurveBuilder.cpp
../shared/GeometryArray.cpp
../shared/Boundary.cpp
../shared/SHelper.cpp
../shared/AStripedModel.cpp
../shared/ATriangleMeshGroup.cpp
../shared/HTriangleMeshGroup.cpp
../shared/APolygonalMesh.cpp
../shared/HPolygonalMesh.cpp
../shared/APolygonalUV.cpp
../shared/HPolygonalUV.cpp
../shared/AVerbose.cpp
../shared/ModelDifference.cpp
../shared/TriangleDifference.cpp
../shared/ADeformer.cpp
../shared/TriangleAnchorDeformer.cpp
../shared/SplineMap1D.cpp
../shared/ATetrahedronMeshGroup.cpp
../shared/HTetrahedronMeshGroup.cpp
                                ../lapl/VertexAdjacency.cpp 
                ../lapl/Anchor.cpp 
                ../catmullclark/BaseQuad.cpp 
                ../catmullclark/LODQuad.cpp 
                ../catmullclark/accPatch.cpp 
                ../catmullclark/accStencil.cpp 
                ../catmullclark/bezierPatch.cpp 
                ../catmullclark/tessellator.cpp 
                ../catmullclark/AccCorner.cpp 
                ../catmullclark/AccEdge.cpp 
                ../catmullclark/AccInterior.cpp 
                ../catmullclark/BezierDrawer.cpp 
				../powell/PowellMethod.cpp 
				../powell/BaseFunction.cpp
../kdtree/KdCluster.cpp
../kdtree/KdIntersection.cpp
../shared/APointCloud.cpp
../shared/ATriangleMesh.cpp
../shared/ATetrahedronMesh.cpp
../shared/GjkIntersection.cpp
../shared/MersenneTwister.cpp
../shared/MeshSeparator.cpp
../shared/PrincipalComponents.cpp
../shared/AOrientedBox.cpp
../shared/DynamicBody.cpp
../shared/ConvexShape.cpp
../shared/ViewFrame.cpp
../btree/Types.cpp
../btree/Entity.cpp
../btree/BNode.cpp
../btree/BTree.cpp
../btree/Entity.cpp
../btree/MortonHash.cpp
../shared/BaseSampler.cpp
../shared/AField.cpp
../shared/CartesianGrid.cpp
../shared/HField.cpp
../shared/TetrahedronMeshBuilder.cpp
../shared/AAnimationCurve.cpp
../shared/HAnimationCurve.cpp
../shared/AAttribute.cpp
../shared/HAttributeGroup.cpp
../shared/AFrustum.cpp
../shared/ViewCull.cpp
../shared/DrawBox.cpp
../shared/DrawCircle.cpp
../shared/UniformGrid.cpp
../hesperis/HAdaptiveField.cpp
../hesperis/H5FieldIn.cpp
../hesperis/H5FieldOut.cpp
../hesperis/AdaptiveGrid.cpp
../hesperis/AdaptiveField.cpp
../hesperis/H5IO.cpp
../hesperis/H5Availability.cpp)

