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
#include <Plane.h>
#include <GlslBase.h>
#include "zEXRImage.h"
class heatherNode : public MPxLocatorNode, public GLSLBase
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

	static  MObject         amatrix;
	static  MObject         anear;
	static  MObject         afar;
	static	MObject		ahapeture;
	static	MObject		avapeture;
	static	MObject		afocallength;
	static	MObject		aorthographic;
	static	MObject		aorthographicwidth;
	static MObject adepthImageName;
	static MObject aframeNumber;
	static MObject outValue;
public: 
	static	MTypeId		id;
	
protected:
    virtual const char* vertexProgramSource() const;
	virtual const char* fragmentProgramSource() const;
	virtual void updateShaderParameters() const;
private:
    void preLoadImage(const char * name, int frame);
private:
    ZEXRImage * m_exr;
    GLuint m_depthImg;
    bool m_needLoadImage;
};
