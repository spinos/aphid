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
#include <CUDABuffer.h>
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
    GLuint m_bgdCImg, m_depthImg, m_colorImg;
	int m_portWidth, m_portHeight;
    bool m_needLoadImage;
    ZEXRImage * m_images[32];
    CUDABuffer * m_colorBuf[2];
    CUDABuffer * m_depthBuf[2];
    CUDABuffer * m_combinedColorBuf;
    CUDABuffer * m_combinedDepthBuf;
    BaseBuffer * m_hostCombinedColorBuf;
    BaseBuffer * m_hostCombinedDepthBuf;
    unsigned m_numImages;
};
