## Fri Feb 10 05:53:07 CST 2018 

cmake_minimum_required (VERSION 2.8)
project (aphid)
set (CMAKE_BUILD_TYPE Release)
include_directories(../shared ../lapl ../catmullclark
../hesperis)

IF (APPLE)
ELSEIF (UNIX)
SET (CMAKE_CXX_FLAGS -fPIC)
ADD_DEFINITIONS(-DLINUX)
ENDIF ()

IF (WIN32)
include_directories(D:/usr/libxml2/include D:/usr/HDF5/include)
ADD_DEFINITIONS(-DWIN32 -DNDEBUG -D_WIN32_WINDOWS 
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

if(WIN32)
	IF (EXISTS "D:/usr/boost_1_51_0")
		SET (Boost_INCLUDE_DIR "D:/usr/boost_1_51_0")
	
	ENDIF ()
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	
ELSEIF (APPLE)
	IF (EXISTS "/users/jianzhang/Library/boost_1_55_0")
		SET (Boost_INCLUDE_DIR "/users/jianzhang/Library/boost_1_55_0")
	ELSEIF (EXISTS "/users/jollyroger/Library/boost_1_55_0")
		SET (Boost_INCLUDE_DIR "/users/jollyroger/Library/boost_1_55_0")
	ENDIF ()

ELSEIF (UNIX)
	SET (BOOST_ROOT "/home/td21/usr/boost_1_51_0")
	
endif()

FIND_PACKAGE (Boost REQUIRED COMPONENTS system filesystem date_time regex)	
	
MESSAGE (" boost version is    " ${Boost_VERSION})
MESSAGE (" boost include is    " ${Boost_INCLUDE_DIR} )
MESSAGE (" boost system is     " ${Boost_SYSTEM_LIBRARY})
MESSAGE (" boost date_time is  " ${Boost_DATE_TIME_LIBRARY})
MESSAGE (" boost regex is      " ${Boost_REGEX_LIBRARY})
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



SET (APHID_GEOM_PATH ../shared/geom)

file (GLOB Geom_SRC
    "../shared/geom/*.h"
    "../shared/geom/*.cpp"
)

ADD_LIBRARY (aphidGeom ${LIB_TYP} 
${Geom_SRC}
../shared/Primitive.cpp
../shared/BaseCurve.cpp
../shared/CircleCurve.cpp 
../shared/BezierCurve.cpp 
../shared/GjkIntersection.cpp
../shared/Boundary.cpp
../shared/Triangle.cpp
../shared/IntersectionContext.cpp 
../shared/BaseTransform.cpp
../shared/Segment.cpp
../shared/CurveBuilder.cpp 
../shared/BaseMesh.cpp
../shared/PyramidMesh.cpp 
../shared/CubeMesh.cpp 
../shared/PlaneMesh.cpp
../catmullclark/bezierPatch.cpp 
../shared/TriangleRaster.cpp
../shared/PrimitiveFilter.cpp
../shared/PatchMesh.cpp
../shared/Patch.cpp  
../shared/PointInsidePolygonTest.cpp  
../shared/InverseBilinearInterpolate.cpp 
../shared/BiLinearInterpolate.cpp 
../shared/LinearInterpolate.cpp  
../shared/VertexDataGroup.cpp
../shared/ViewCull.cpp)

INCLUDE ("../cmake/hdf5_config.cmake")

file (GLOB H5_SRC
    "../shared/h5/*.h"
    "../shared/h5/*.cpp")

ADD_LIBRARY (aphidH5 ${LIB_TYP}
    ${H5_SRC} 
../shared/AFrameRange.cpp  
../shared/BaseFile.cpp
../shared/AAnimationCurve.cpp
)

set (APH_HES_SRC
../hesperis/SampleFrame.cpp
../hesperis/H5VCache.cpp
../hesperis/H5Holder.cpp
)

ADD_LIBRARY (aphidHes ${LIB_TYP}
    ${APH_HES_SRC}
)

ADD_LIBRARY (aphidUI ${LIB_TYP}
../shared/BaseDrawer.cpp 
../shared/LineDrawer.cpp 
../shared/MeshDrawer.cpp 
../shared/GeoDrawer.cpp  
../shared/LineBuffer.cpp
../shared/GLHUD.cpp
../shared/BaseBrush.cpp
../shared/ToolContext.cpp
../shared/SelectionArray.cpp
../shared/BaseDeformer.cpp  
../shared/PerspectiveView.cpp)

set (APH_SDB_PATH ../shared/sdb)

INCLUDE (../cmake/clapack_config.cmake)

INCLUDE (../cmake/ogl_config.cmake)

add_library(aphid ${LIB_TYP}                                
    ../shared/SkeletonSubspaceDeformer.cpp                                  
                ../shared/PrimitiveArray.cpp                 
                ../shared/ClassificationStorage.cpp 
		../shared/IndexArray.cpp 
                ##../shared/BoundingBoxList.cpp   
                ../shared/AnchorGroup.cpp 
../shared/Anchor.cpp                   
                               ## ../shared/AccPatchMesh.cpp 
				../shared/PatchNeighborRec.cpp 
				##../shared/PatchTexture.cpp 
                                ##../shared/BaseTessellator.cpp 
                                ##../shared/CollisionRegion.cpp  
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
                ## ../shared/BaseServer.cpp
##../shared/APointCloud.cpp
##../shared/MeshSeparator.cpp
##../shared/DynamicBody.cpp
##../shared/ViewFrame.cpp
##../btree/MortonHash.cpp
##../shared/BaseSampler.cpp
##../shared/AField.cpp
##../shared/HField.cpp
##../shared/UniformGrid.cpp
)

INCLUDE (../cmake/exr_config.cmake)

INCLUDE (../cmake/cuda_config.cmake)

IF (${CUDA_FOUND})

message (" aphid has cuda ")

include_directories (${CUDA_TOOLKIT_INCLUDE} ${CUDA_SDK_ROOT_DIR}/common/inc)

CUDA_ADD_LIBRARY (aphidCu ${LIB_TYP}
../shared/CudaBase.cpp
../shared/CudaGLBase.cpp
../shared/CUDABuffer.cpp
../shared/CudaRender.cpp
)

ELSE ()
	MESSAGE (" cuda not found ")
ENDIF ()

add_subdirectory(found)
add_subdirectory(math)
add_subdirectory(pbd)
add_subdirectory(sdb)
add_subdirectory(ogl)
add_subdirectory(wla)
add_subdirectory(img)
add_subdirectory(gpr)
add_subdirectory(smp)
add_subdirectory(ttg)
add_subdirectory(topo)
add_subdirectory(graph)
add_subdirectory(kd)
add_subdirectory(lbm)

