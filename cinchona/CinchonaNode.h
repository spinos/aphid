#ifndef CINCHONA_NODE_H
#define CINCHONA_NODE_H

/*
 *  CinchonaNode.h
 *  proxyPaint
 *
 *  Created by jian zhang on 3/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#include <maya/MPxLocatorNode.h> 
#include <maya/MTypeId.h> 
#include <maya/M3dView.h>
#include <maya/MDagPath.h>
#include <maya/MSceneMessage.h>
#include <maya/MIntArray.h>
#include <maya/MVectorArray.h>
#include "MAvianArm.h"

namespace aphid {

class BoundingBox;

}

class CinchonaNode : public MPxLocatorNode, public MAvianArm
{

public:
	CinchonaNode();
	virtual ~CinchonaNode(); 

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

	static MTypeId id;
	static MObject ahumerusmat;
	static MObject aulnamat;
	static MObject aradiusmat;
	static MObject acarpusmat;
	static MObject aseconddigitmat;
	static MObject aligament0x;
	static MObject aligament0y;
	static MObject aligament0z;
	static MObject aligament0;
	static MObject aligament1x;
	static MObject aligament1y;
	static MObject aligament1z;
	static MObject aligament1;
	static MObject aelbowos1x;
	static MObject aelbowos1y;
	static MObject aelbowos1z;
	static MObject aelbowos1;
	static MObject awristos0x;
	static MObject awristos0y;
	static MObject awristos0z;
	static MObject awristos0;
	static MObject awristos1x;
	static MObject awristos1y;
	static MObject awristos1z;
	static MObject awristos1;
	static MObject adigitos0x;
	static MObject adigitos0y;
	static MObject adigitos0z;
	static MObject adigitos0;
	static MObject adigitos1x;
	static MObject adigitos1y;
	static MObject adigitos1z;
	static MObject adigitos1;
	static MObject adigitl;
	static MObject outValue;
	
protected:
		   
private:
	void attachSceneCallbacks();
	void detachSceneCallbacks();
	static void releaseCallback(void* clientData);

	MCallbackId fBeforeSaveCB;
	void saveInternal();
	bool loadInternal();
	bool loadInternal(MDataBlock& block);
	
};
#endif        //  #ifndef CinchonaNode_H
