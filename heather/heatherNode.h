/*
 *  heatherNode.h
 *  heather
 *
 *  Created by jian zhang on 2/10/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <math.h>
#include <maya/MIOStream.h>
#include <maya/MPxLocatorNode.h> 
#include <maya/MString.h> 
#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MVector.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MColor.h>
#include <maya/M3dView.h>
#include <maya/MDistance.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MMatrix.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MPoint.h>
#include <maya/MString.h>
#include <maya/MDagPath.h>
#include <maya/MSelectionList.h>
#include <maya/MFnCamera.h>
#include <maya/MGlobal.h>
#include <maya/MObjectArray.h>
#include <maya/MDagPathArray.h>
#include <Plane.h>
#include <GlFramebuffer.h>
#include "ExrImgData.h"
#include "ClampShader.h"
#include "DepthShader.h"
class BaseBuffer;
class CUDABuffer;
class CudaTexture;
class StripeCompressedRGBAZImage;
class heatherNode : public MPxLocatorNode
{
public:
	heatherNode();
	virtual ~heatherNode(); 

    virtual MStatus   		compute( const MPlug& plug, MDataBlock& data );

	virtual void            draw( M3dView & view, const MDagPath & path, 
								  M3dView::DisplayStyle style,
								  M3dView::DisplayStatus status );

	virtual bool            isBounded() const;
	
	static  void *          creator();
	static  MStatus         initialize();

	static MObject ainimages;
	static MObject ablockSetName;
	static MObject acameraName;
	static MObject acompressRatio;
	static MObject outValue;
	
public: 
	static	MTypeId		id;
	
private:
    void addImage(ExrImgData::DataDesc * desc);
    void computeCombinedBufs();
    void cacheBlocks(const MString & setname);
	void cacheMeshFromNode(const MString & name);
	void cacheMeshFromNode(const MObject & node);
	void drawBackPlane(double farPlane, const GLdouble * mproj, const Matrix44F & mmvinv, const float & aspectRatio, 
	                   const double & overscan,
	                   const float & gatePortRatioW, const float & gatePortRatioH, char isHorizontalFit = true,
	                   float gateSqueezeX = 1.f, float gateSqueezeY = 1.f);
	void drawBlocks();
	void cacheMeshes();
private:
	MDagPathArray m_meshes;
	MString m_carmeraName;
	GlFramebuffer * m_framebuffer;
	ClampShader m_clamp;
	DepthShader m_depth;
    GLuint m_bgdCImg;
	int m_portWidth, m_portHeight;
	int m_compressRatio;
    bool m_needLoadImage;
    static StripeCompressedRGBAZImage * m_compressedImages[32];
    static CUDABuffer * m_colorBuf;
    static CUDABuffer * m_depthBuf;
    static CUDABuffer * m_combinedColorBuf;
    static CUDABuffer * m_combinedDepthBuf;
    static CudaTexture * m_combinedColorTex;
    static CudaTexture * m_combinedDepthTex;
    static BaseBuffer * m_decompressedColor;
    static BaseBuffer * m_decompressedDepth;
    unsigned m_numImages;
};
