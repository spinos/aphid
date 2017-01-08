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
	static MObject anumfeather0;
	static MObject anumfeather1;
	static MObject anumfeather2;
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
	static MObject adigitl;
	static MObject ainboardmat;
	static MObject amidsectmat0;
	static MObject amidsectmat1;
	static MObject achord0;
	static MObject achord1;
	static MObject achord2;
	static MObject achord3;
	static MObject athickness0;
	static MObject athickness1;
	static MObject athickness2;
	static MObject athickness3;
	static MObject abrt0mat;
	static MObject abrt1mat;
	static MObject abrt2mat;
	static MObject abrt3mat;
	static MObject aup0n0;
	static MObject aup0n1;
	static MObject aup0n2;
	static MObject aup0n3;
	static MObject aup0c0;
	static MObject aup0c1;
	static MObject aup0c2;
	static MObject aup0c3;
	static MObject aup0c4;
	static MObject aup0t0;
	static MObject aup0t1;
	static MObject aup0t2;
	static MObject aup0t3;
	static MObject aup0t4;
	static MObject aup0rz;
	static MObject aup1n0;
	static MObject aup1n1;
	static MObject aup1n2;
	static MObject aup1n3;
	static MObject aup1c0;
	static MObject aup1c1;
	static MObject aup1c2;
	static MObject aup1c3;
	static MObject aup1c4;
	static MObject aup1t0;
	static MObject aup1t1;
	static MObject aup1t2;
	static MObject aup1t3;
	static MObject aup1t4;
	static MObject aup1rz;
	static MObject alow0n0;
	static MObject alow0n1;
	static MObject alow0n2;
	static MObject alow0n3;
	static MObject alow0c0;
	static MObject alow0c1;
	static MObject alow0c2;
	static MObject alow0c3;
	static MObject alow0c4;
	static MObject alow0t0;
	static MObject alow0t1;
	static MObject alow0t2;
	static MObject alow0t3;
	static MObject alow0t4;
	static MObject alow0rz;
	static MObject alow1n0;
	static MObject alow1n1;
	static MObject alow1n2;
	static MObject alow1n3;
	static MObject alow1c0;
	static MObject alow1c1;
	static MObject alow1c2;
	static MObject alow1c3;
	static MObject alow1c4;
	static MObject alow1t0;
	static MObject alow1t1;
	static MObject alow1t2;
	static MObject alow1t3;
	static MObject alow1t4;
	static MObject alow1rz;
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
