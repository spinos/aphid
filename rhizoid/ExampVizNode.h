/*
 *  ExampVizNode.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include "ExampVox.h"
#include <DrawCircle.h>
#include <maya/MPxLocatorNode.h> 
#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/M3dView.h>
#include <maya/MPointArray.h>

class ExampViz : public MPxLocatorNode, public ExampVox, public DrawCircle
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
	static  MObject         ancells;
	static  MObject         acellBuf;
	static MObject adrawColorR;
	static MObject adrawColorG;
	static MObject adrawColorB;
	static MObject adrawColor;
	static MObject aradiusMult;
	static MObject outValue;
	static	MTypeId		id;
	
	virtual void voxelize(KdTree * tree);
	
private:
	void loadBoxes(MObject & node);
	void updateGeomBox(MObject & node);
	void loadBoxes(MDataBlock & data);
	void setBoxes(const MPointArray & src, const unsigned & num);
	
};