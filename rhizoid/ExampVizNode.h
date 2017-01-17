/*
 *  ExampVizNode.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include "ExampVox.h"
#include <ogl/DrawCircle.h>
#include <maya/MPxLocatorNode.h> 
#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/M3dView.h>
#include <maya/MPointArray.h>

namespace aphid {

class BoundingBox;

template<typename T>
class DenseMatrix;

}

class ExampViz : public MPxLocatorNode, public aphid::ExampVox, public aphid::DrawCircle
{
	float m_transBuf[16];
	
public:
	ExampViz();
	virtual ~ExampViz(); 

    virtual MStatus   		compute( const MPlug& plug, MDataBlock& data );

	virtual void            draw( M3dView & view, const MDagPath & path, 
								  M3dView::DisplayStyle style,
								  M3dView::DisplayStatus status );

	virtual bool            isBounded() const;
	virtual MBoundingBox    boundingBox() const; 
	virtual MStatus connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc );
	virtual MStatus connectionBroken ( const MPlug & plug, const MPlug & otherPlug, bool asSrc );
	
	static  void *          creator();
	static  MStatus         initialize();

	static  MObject         abboxminv;
	static  MObject         abboxmaxv;
	static  MObject         adoplen;
	static  MObject         adopPBuf;
	static  MObject         adopNBuf;
	static MObject adrawColorR;
	static MObject adrawColorG;
	static MObject adrawColorB;
	static MObject adrawColor;
	static MObject adrawDopSizeX;
	static MObject adrawDopSizeY;
	static MObject adrawDopSizeZ;
	static MObject adrawDopSize;
	static MObject aradiusMult;
	static MObject aininstspace;
	static MObject outValue;
	static	MTypeId		id;
	
	void setTriangleMesh(const aphid::DenseMatrix<float> & pnts,
						const MIntArray & triangleVertices,
						const aphid::BoundingBox & bbox);
						
	virtual void voxelize2(aphid::sdb::VectorArray<aphid::cvx::Triangle> * tri,
							const aphid::BoundingBox & bbox);
	
private:
	void updateGeomBox(MObject & node);
	bool loadTriangles(MDataBlock & data);
	bool loadTriangles(MObject & node);
	
};