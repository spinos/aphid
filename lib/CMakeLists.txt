cmake_minimum_required (VERSION 2.8)
project (aphid)
set (CMAKE_BUILD_TYPE Release)
include_directories(../shared ../lapl ../catmullclark 
../hesperis)

IF (UNIX)
SET (CMAKE_CXX_FLAGS -fPIC)
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
	ELSEIF (EXISTS "/users/jollyroger/Library/boost_1_55_0")
		SET (Boost_INCLUDE_DIR "/users/jollyroger/Library/boost_1_55_0")
	ENDIF ()

ELSEIF (UNIX)
	SET (BOOST_ROOT "~/Public/boost_1_51_0")
	
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

include_directories (${OpenEXR_INCLUDE_DIR})

if(WIN32)
    include_directories (D:/usr/CLAPACK/include)
    set(BLAS_LIBRARY D:/usr/CLAPACK/lib/blas.lib)
	set(LAPACK_LIBRARY D:/usr/CLAPACK/lib/lapack.lib)
    set(F2C_LIBRARY D:/usr/CLAPACK/lib/libf2c.lib)

elseif(APPLE)
	set(F2C_LIBRARY libf2c.a)
	set(BLAS_LIBRARY libblas.a)
	set(LAPACK_LIBRARY liblapack.a)
else ()
include_directories (/usr/local/include)
	set(F2C_LIBRARY /usr/local/lib/libf2c.so)
    set(BLAS_LIBRARY /usr/local/lib/libblas.so)
	set(LAPACK_LIBRARY /usr/local/lib/liblapack.so)
	
endif()

message ("blas " ${BLAS_LIBRARY})
message ("lapack " ${LAPACK_LIBRARY})
message ("f2c " ${F2C_LIBRARY})

IF (WIN32) 
	SET (LIB_TYP STATIC)
ELSEIF (APPLE)
	SET (LIB_TYP STATIC)
ELSE ()
	SET (LIB_TYP SHARED)
ENDIF ()

MESSAGE (" library type is " ${LIB_TYP})

file (GLOB Foundation_SRC
"../shared/foundation/*.h"
    "../shared/foundation/*.cpp"
)

ADD_LIBRARY (aphidFoundation ${LIB_TYP}
${Foundation_SRC})

file (GLOB Math_SRC
    "../shared/math/*.h"
    "../shared/math/*.cpp"
)

ADD_LIBRARY (aphidMath ${LIB_TYP} 
${Math_SRC}
../shared/plots.cpp)

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
../shared/DiscMesh.cpp
../shared/PlaneMesh.cpp
../catmullclark/bezierPatch.cpp 
../shared/AFrustum.cpp
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

file (GLOB H5_SRC
    "../shared/h5/*.h"
    "../shared/h5/*.cpp")

ADD_LIBRARY (aphidH5 ${LIB_TYP}
    ${H5_SRC} 
../shared/AFrameRange.cpp  
../shared/BaseFile.cpp
../shared/AAnimationCurve.cpp
)

ADD_LIBRARY (aphidUI ${LIB_TYP}
../shared/BaseCamera.cpp
../shared/PerspectiveCamera.cpp 
../shared/BaseDrawer.cpp 
../shared/LineDrawer.cpp 
../shared/MeshDrawer.cpp 
../shared/GeoDrawer.cpp  
../shared/LineBuffer.cpp
../shared/AdaptableStripeBuffer.cpp
../shared/GLHUD.cpp
../shared/BaseBrush.cpp
../shared/ToolContext.cpp
../shared/SelectionArray.cpp
../shared/BaseState.cpp
../shared/BaseDeformer.cpp  
../shared/PerspectiveView.cpp)

set (APH_SDB_PATH ../shared/sdb)

file (GLOB Sdb_SRC
    "../shared/sdb/*.h"
    "../shared/sdb/*.cpp"
)

ADD_LIBRARY (aphidSdb ${LIB_TYP}
${Sdb_SRC})

file (GLOB APH_Kd_SRC
	"../shared/kd/*.h"
    "../shared/kd/*.cpp"
)

ADD_LIBRARY (aphidKd ${LIB_TYP}
${APH_Kd_SRC})

file (GLOB graph_SRC
	"../shared/graph/*.h"
    "../shared/graph/*.cpp"
)

ADD_LIBRARY (aphidGraph ${LIB_TYP}
${graph_SRC})

file (GLOB Topo_SRC
    "../shared/topo/*.h"
    "../shared/topo/*.cpp"
)

ADD_LIBRARY (aphidTopology ${LIB_TYP}
${Topo_SRC})

file (GLOB Gpr_SRC
    "../shared/gpr/*.h"
    "../shared/gpr/*.cpp"
)

ADD_LIBRARY (aphidGp ${LIB_TYP}
${Gpr_SRC})

file (GLOB Ogl_SRC
    "../shared/ogl/*.h"
    "../shared/ogl/*.cpp"
)

ADD_LIBRARY (aphidOgl ${LIB_TYP}
${Ogl_SRC})

file (GLOB ttg_SRC
    "../shared/ttg/*.h"
    "../shared/ttg/*.cpp"
)

ADD_LIBRARY (aphidTtg ${LIB_TYP}
${ttg_SRC})

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
