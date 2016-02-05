/*
 *  ExampVizNode.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include "ExampVox.h"
#include <maya/MPxLocatorNode.h> 
#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/M3dView.h>

class ExampViz : public MPxLocatorNode, public ExampVox
{
	
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
	static MObject adrawColor;
	static MObject outValue;
	static	MTypeId		id;
	
	virtual void voxelize(KdIntersection * tree);
	
private:
	void loadBoxes(MObject & node);
	
};