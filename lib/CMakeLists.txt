cmake_minimum_required (VERSION 2.8)
project (aphid)
set (CMAKE_BUILD_TYPE RELEASE)
include_directories(../mallard ../shared ../lapl ../catmullclark 
../../ofl/opium ../fit ../powell ../../ofl/shared
../btree
../kdtree
../ntree
../hesperis)

IF (UNIX)
SET (CMAKE_CXX_FLAGS -fPIC)
ENDIF ()

IF (WIN32)
include_directories(D:/usr/libxml2/include D:/usr/HDF5/include)
ADD_DEFINITIONS( -DNDEBUG -D_WIN32_WINDOWS 
## http://www.boost.org/doc/libs/1_49_0/libs/config/doc/html/boost_config/boost_macro_reference.html
-DBOOST_HAS_TR1)
## http://stackoverflow.com/questions/3499487/replacing-native-vs-2010-vc10-tr1-libraries-with-boost-tr1
set (_HAS_CPP0X 0)
set (AttributeNoninline "__attribute__ ((noinline))")
add_definitions (-DBOOST_NOINLINE=${AttributeNoninline})

ELSEIF (APPLE)
	set (AttributeNoninline "__attribute__ ((noinline))")
add_definitions (-DBOOST_NOINLINE=${AttributeNoninline})

ELSEIF (UNIX)
    ADD_DEFINITIONS(-DBOOST_HAS_TR1)
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
ELSEIF (UNIX)
	SET (HDF5_VERSION "1.8.17")
	SET (HDF5_INCLUDE_DIRS "~/Public/hdf5-1.8.17/include")
        SET (HDF5_LIBRARIES "~/Public/hdf5-1.8.17/lib/libhdf5.so")
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

ELSEIF (UNIX)
	SET (BOOST_ROOT "~/Public/boost_1_51_0")
	
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

IF (WIN32) 
	SET (LIB_TYP STATIC)
ELSEIF (APPLE)
	SET (LIB_TYP STATIC)
ELSE ()
	SET (LIB_TYP SHARED)
ENDIF ()

MESSAGE (" library type is " ${LIB_TYP})

ADD_LIBRARY (aphidMath ${LIB_TYP}
../shared/Vector3F.cpp 
                ../shared/Vector2F.cpp 
                ../shared/Matrix44F.cpp 
                ../shared/Matrix33F.cpp 
                ../shared/Quaternion.cpp 
    ../shared/BoundingBox.cpp 
    ../shared/BoundingRectangle.cpp
../shared/AOrientedBox.cpp
../shared/Ray.cpp
../shared/MersenneTwister.cpp
../shared/PseudoNoise.cpp
../shared/QuickSort.cpp
../shared/Plane.cpp
../shared/ViewCull.cpp
../shared/ANoise3.cpp
../shared/Calculus.cpp
../shared/plots.cpp
)

ADD_LIBRARY (aphidGeom ${LIB_TYP} 
../shared/SHelper.cpp 
../shared/AVerbose.cpp
../shared/BaseArray.cpp 
../shared/AStripedModel.cpp
../shared/TypedEntity.cpp 
../shared/NamedEntity.cpp 
../shared/Geometry.cpp
../shared/Primitive.cpp
../shared/BaseCurve.cpp
../shared/CircleCurve.cpp 
../shared/BezierCurve.cpp
../shared/AGenericMesh.cpp 
../shared/ATriangleMesh.cpp
../shared/ATetrahedronMesh.cpp
../shared/ATetrahedronMeshGroup.cpp
../shared/GjkIntersection.cpp
../shared/CurveGroup.cpp 
../shared/BaseBuffer.cpp
../shared/GeometryArray.cpp
../shared/Boundary.cpp
../shared/AQuadMesh.cpp
../shared/ATriangleMeshGroup.cpp
../shared/APolygonalMesh.cpp
../shared/APolygonalUV.cpp
../shared/Triangle.cpp
../shared/BarycentricCoordinate.cpp  
../shared/IntersectionContext.cpp 
../shared/BaseTransform.cpp
../shared/Segment.cpp
../shared/CurveBuilder.cpp 
../shared/BaseMesh.cpp
../shared/GeodesicSphereMesh.cpp 
../shared/PyramidMesh.cpp 
../shared/CubeMesh.cpp 
../shared/DiscMesh.cpp
../shared/PlaneMesh.cpp
../shared/ConvexShape.cpp
../catmullclark/bezierPatch.cpp 
../shared/AFrustum.cpp
../shared/TriangleRaster.cpp
../shared/PrimitiveFilter.cpp
../shared/BoxNeighbors.cpp
../shared/PatchMesh.cpp
../shared/Patch.cpp  
../shared/PointInsidePolygonTest.cpp  
../shared/InverseBilinearInterpolate.cpp 
../shared/BiLinearInterpolate.cpp 
../shared/LinearInterpolate.cpp  
../shared/VertexDataGroup.cpp
)

ADD_LIBRARY (aphidH5 ${LIB_TYP}
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
../shared/HCurveGroup.cpp
../shared/AFrameRange.cpp 
../shared/HFrameRange.cpp
../shared/HTransform.cpp
../shared/HTriangleMesh.cpp
../shared/HTetrahedronMesh.cpp
../shared/HAttributeEntry.cpp
../shared/BaseFile.cpp
../shared/HFile.cpp
../shared/HTetrahedronMeshGroup.cpp 
../shared/HTriangleMeshGroup.cpp
../shared/HPolygonalMesh.cpp
../shared/HPolygonalUV.cpp
../shared/AAnimationCurve.cpp
../shared/HAnimationCurve.cpp
../shared/AAttribute.cpp
../shared/HAttributeGroup.cpp
../shared/HNumericBundle.cpp
../hesperis/H5IO.cpp
../hesperis/H5Availability.cpp
../hesperis/HesperisFile.cpp
)

ADD_LIBRARY (aphidUI ${LIB_TYP}
../shared/BaseCamera.cpp
../shared/PerspectiveCamera.cpp 
../shared/BaseDrawer.cpp 
##../shared/DrawBuffer.cpp
../shared/LineDrawer.cpp 
../shared/MeshDrawer.cpp 
../shared/GeoDrawer.cpp 
##../shared/LightDrawer.cpp 
../shared/LineBuffer.cpp
../shared/AdaptableStripeBuffer.cpp
../shared/GLHUD.cpp
../shared/GLight.cpp 
../shared/GMaterial.cpp 
../shared/GProfile.cpp
../shared/BaseBrush.cpp
../shared/ToolContext.cpp
../shared/SelectionArray.cpp
../shared/ComponentConversion.cpp
../shared/GeoElement.cpp 
../shared/Vertex.cpp 
../shared/Edge.cpp 
../shared/Facet.cpp 
../shared/VertexPath.cpp
../lapl/VertexAdjacency.cpp
../shared/MeshTopology.cpp 
##../shared/BaseField.cpp
../shared/BaseState.cpp
../shared/BaseDeformer.cpp  
../shared/DrawDistanceField.cpp
../shared/DrawBox.cpp
../shared/PerspectiveView.cpp)

ADD_LIBRARY (aphidSdb ${LIB_TYP}
../btree/Types.cpp
../btree/Entity.cpp
../btree/BNode.cpp
../btree/MortonHash.cpp
)

ADD_LIBRARY (aphidKd ${LIB_TYP}
../shared/KdTreeNode.cpp
../shared/CartesianGrid.cpp
../shared/PrincipalComponents.cpp
../shared/MinMaxBins.cpp
../shared/SplitEvent.cpp
../shared/BaseBinSplit.cpp 
../btree/GridClustering.cpp
../kdtree/TreeProperty.cpp
../kdtree/PrimBoundary.cpp
)

ADD_LIBRARY (aphidGraph ${LIB_TYP}
../shared/ADistanceField.cpp
../shared/BDistanceFunction.cpp
)

add_library(aphid ${LIB_TYP}                 
##../shared/SpaceHandle.cpp                
    ../shared/SkeletonSubspaceDeformer.cpp                                  
                ../shared/PrimitiveArray.cpp                 
                ../shared/ClassificationStorage.cpp 
		../shared/IndexArray.cpp 
                ##../shared/BoundingBoxList.cpp  
                ##../shared/EasemodelUtil.cpp 
                ../shared/AnchorGroup.cpp                    
                               ## ../shared/AccPatchMesh.cpp 
				../shared/PatchNeighborRec.cpp 
				##../shared/PatchTexture.cpp 
                                ##../shared/BaseTessellator.cpp 
                                ##../shared/CollisionRegion.cpp 
                                ## 
                                ##../shared/BlockDrawBuffer.cpp 
                ##../shared/BaseScene.cpp   
    ##../shared/BaseFileGroup.cpp
    ##../shared/BakeDeformer.cpp 
    ##../shared/PlaybackControl.cpp 
    ##../shared/CacheFile.cpp  
    ##../shared/HRFile.cpp 
    ##../shared/TransformManipulator.cpp 
    ##../shared/MeshManipulator.cpp 
	##../shared/SkeletonJoint.cpp 
	##../shared/SkeletonSystem.cpp 
	##../shared/SkeletonPose.cpp  
	##../shared/SelectCondition.cpp 
    ##../shared/FloodCondition.cpp 
	##../shared/ActiveRegion.cpp 
	##../shared/AccPatchGroup.cpp  
	##../shared/BezierPatchHirarchy.cpp 
	##../shared/KMeansClustering.cpp 
	##../shared/BaseSphere.cpp   
    ##../shared/BaseClient.cpp 
	##../shared/RenderEngine.cpp 
	## 
	##../shared/BlockStripeBuffer.cpp 
	##../shared/LODFn.cpp 
	##../shared/NoisePlot.cpp 
	##../shared/FractalPlot.cpp 
	##../shared/BaseLight.cpp 
	##../shared/DistantLight.cpp 
	##../shared/PointLight.cpp 
	##../shared/SquareLight.cpp 
##../shared/LightGroup.cpp 
##../shared/RenderOptions.cpp  
##../shared/BaseTexture.cpp 
##../shared/TextureGroup.cpp 
##../shared/ColorBlend.cpp 
##../shared/TexturePainter.cpp 
##../shared/BaseShader.cpp 
##../shared/ShaderGroup.cpp 
##../shared/RayMarch.cpp 
##../shared/BaseLog.cpp
##../shared/APlayback.cpp
##../shared/APlaybackFile.cpp
##../shared/MultiPlaybackFile.cpp
##../shared/ModelDifference.cpp
##../shared/TriangleDifference.cpp
##../shared/ADeformer.cpp
##../shared/TriangleAnchorDeformer.cpp
##../shared/SplineMap1D.cpp
##../lapl/Anchor.cpp 
##../catmullclark/BaseQuad.cpp 
##../catmullclark/LODQuad.cpp 
##../catmullclark/accPatch.cpp 
##../catmullclark/accStencil.cpp  
##../catmullclark/tessellator.cpp 
##../catmullclark/AccCorner.cpp 
##../catmullclark/AccEdge.cpp 
##../catmullclark/AccInterior.cpp 
               ## ../catmullclark/BezierDrawer.cpp 
		##		../powell/PowellMethod.cpp 
		##		../powell/BaseFunction.cpp
##../shared/KdTreeNode.cpp 
                ##../shared/BuildKdTreeContext.cpp 
                ##../shared/KdTreeNodeArray.cpp 
                ##../shared/KdTree.cpp 
                ##../shared/KdTreeBuilder.cpp 
                ##../shared/SplitEvent.cpp 
                ##../shared/MinMaxBins.cpp 
                ##../shared/BuildKdTreeStream.cpp 
                ## ../shared/BaseServer.cpp
##../kdtree/KdCluster.cpp
##../kdtree/KdIntersection.cpp
##../shared/APointCloud.cpp
##../shared/MeshSeparator.cpp
##../shared/PrincipalComponents.cpp
##../shared/DynamicBody.cpp
##../shared/ViewFrame.cpp
##../btree/BTree.cpp
##../btree/MortonHash.cpp
##../shared/BaseSampler.cpp
##../shared/AField.cpp
##../shared/HField.cpp
##../shared/TetrahedronMeshBuilder.cpp
##../shared/DrawCircle.cpp
##../shared/UniformGrid.cpp
##../hesperis/HAdaptiveField.cpp
##../hesperis/H5FieldIn.cpp
##../hesperis/H5FieldOut.cpp
##../hesperis/AdaptiveGrid.cpp
##../hesperis/AdaptiveField.cpp
)

add_library(aphidImg ${LIB_TYP}
../shared/BaseImage.cpp
../shared/ExrImage.cpp
)

find_package (CUDA QUIET)

IF (${CUDA_FOUND})

message ("cuda found ? " ${CUDA_FOUND})
message ("cuda version " ${CUDA_VERSION})
message ("cuda sdk root dir " ${CUDA_SDK_ROOT_DIR})
message ("cuda librarie " ${CUDA_CUDA_LIBRARY})
message ("cuda art librarie " ${CUDA_CUDART_LIBRARY})
message ("cuda blas librarie " ${CUDA_cublas_LIBRARY})
message ("cuda sparse librarie " ${CUDA_cusparse_LIBRARY})

CUDA_ADD_LIBRARY (aphidCu ${LIB_TYP}
../shared/CudaBase.cpp
../shared/CudaGLBase.cpp
../shared/CUDABuffer.cpp
../shared/CudaRender.cpp
)

ELSE ()
	MESSAGE (" cuda not found ")
ENDIF ()
