#ifndef VEG_EXAMPLE_NODE_H
#define VEG_EXAMPLE_NODE_H

/*
 *  VegExampleNode.h
 *  proxyPaint
 *
 *	n examples, 1 instance each
 *
 *  Created by jian zhang on 3/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#include <maya/MPxLocatorNode.h> 
#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/M3dView.h>
#include <maya/MDagPath.h>
#include <maya/MSceneMessage.h>
#include <maya/MIntArray.h>
#include <maya/MVectorArray.h>
#include <ogl/DrawCircle.h>
#include "GardenExamp.h"

namespace aphid {

class Matrix44F;
class BoundingBox;

class VegExampleNode : public MPxLocatorNode, public GardenExamp, public DrawCircle
{
	Matrix44F * m_cameraSpace;
	
public:
	VegExampleNode();
	virtual ~VegExampleNode(); 

    virtual MStatus   		compute( const MPlug& plug, MDataBlock& data );

	virtual void            draw( M3dView & view, const MDagPath & path, 
								  M3dView::DisplayStyle style,
								  M3dView::DisplayStatus status );

	virtual bool            isBounded() const;
	virtual MBoundingBox    boundingBox() const; 
	
	static  void *          creator();
	static  MStatus         initialize();

	static MTypeId id;
/// overall bbox
	static MObject ashrubbox;
/// bbox of each group, length n*2, n is # groups
	static MObject ainstbbox;
/// begin of each group, length (n+1), n is # groups
	static MObject ainstrange;
/// ind to instance
	static MObject ainstexamp;
/// space to instance
	static MObject ainsttrans;
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
	static MObject avoxactive;
	static MObject avoxvisible;
	static MObject avoxpriority;
    static MObject outValue;
	
	void saveInternal();
	
protected:
	void getBBox(BoundingBox & bbox) const;
	   
private:
	bool loadInstances(const MVectorArray & instvecs,
						const MIntArray & instexmps);
	bool loadInternal();
	bool loadInternal(MDataBlock& block);
	void saveBBox(const BoundingBox & bbox);
	void saveGroupBBox();
	int saveGroupRange();
	
};

}
#endif        //  #ifndef VegExampleNode_H
